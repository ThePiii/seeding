import logging
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import json

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)
# from seeding.ops.models2 import TSN
# from mmbt_example.utils_mmimdb import ImageEncoder, get_image_transforms
import torchvision.transforms as transforms
import torchvision
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd
from utils import url_to_pilimg_train, preprocess, Logger
import gc
import warnings

gc.collect()
torch.cuda.empty_cache()

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

logger = Logger(f'logs/mmbt_v1.log', resume=True)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# V1版本，多张图像

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model_path = r'D:\研究生\毕业论文\MMBT\cache\torch\hub\checkpoints\resnet152-394f9c45.pth'
        if os.path.exists(model_path):
            model = torchvision.models.resnet152(pretrained=False)
            model.load_state_dict(torch.load(r'D:\研究生\毕业论文\MMBT\cache\torch\hub\checkpoints\resnet152-394f9c45.pth'))
        else:
            model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.conv = nn.Conv2d(in_channels=18, out_channels=3, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[3])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        # 单张图像 Bx3X224X224，六张图像Bx18x224x224，故降低通道数后再传入模型
        out = self.conv(x)      # 图像拼接后，降低通道数
        out = self.model(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # 矩阵转置
        return out  # BxNx2048


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            )
        ]
    )


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, labels, max_seq_length):
        data = pd.read_csv(data_path, lineterminator='\n').sample(100)
        data = data.fillna(' ')
        self.data = data[['content', 'trendImageUrl', 'is_commercial']]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data.iloc[index]["content"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        # label = torch.zeros(self.n_classes)
        # label[[self.labels.index(tgt) for tgt in self.data.iloc[index]["label"]]] = 1
        label = self.data.iloc[index]['is_commercial']

        # image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        urls = self.data.iloc[index]['trendImageUrl'].split(',')[0: 6]
        imgs = []
        for url in urls:

            img = url_to_pilimg_train(url)
            imgs.append(img)

        image = preprocess(imgs, 6)

        # try:
        #     image.save(f'image/{url.split("/")[-1]}')
        # except:
        #     pass

        # image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        d = self.data.to_dict('records')
        for row in d:
            label_freqs.update(str(row["is_commercial"]))  # 转为一个列表，列表内每个元素为字典，与源代码中数据格式一致
        return label_freqs


def load_examples(data_dir, tokenizer):
    path = data_dir
    transform = get_image_transforms()
    labels = ['0', '1']
    dataset = JsonlDataset(path, tokenizer, transform, labels, 256)
    return dataset


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    # tgt_tensor = torch.stack(torch.tensor([row["label"] for row in batch]))
    tgt_tensor = torch.tensor([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


def evaluate(model, eval_dataset, criterion, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_output_dir = 'output'

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    batch_size = 4

    logger.append("***** Running evaluation {} *****".format(prefix))
    logger.append("  Num examples = {}".format(len(eval_dataset)))
    logger.append("  Batch size = {}".format(batch_size))

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0, shuffle=True
    )
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):  # tqdm用于可视化展示进度
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)

        with torch.no_grad():
            batch = tuple(t.to('cuda') for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": True
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            # preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            _, preds = torch.max(logits, dim=1)  # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            preds = preds.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            # preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
            _, temp_preds = torch.max(logits, dim=1)
            preds = np.append(preds, temp_preds.detach().cpu().numpy())
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps

    result = {
        "loss": eval_loss,
        "micro_f1": f1_score(out_label_ids, preds, average="micro"),
    }
    logger.append(classification_report(out_label_ids, preds, digits=4))
    return result


def train(train_dataset, evaluate_dataset, model, criterion, EPOCHS, lr):
    logger.append("***** Running training *****")
    logger.append("  Num examples = {}".format(len(train_dataset)))
    logger.append("  Num Epochs = {}".format(len(train_dataset)))
    batch_size = 4

    tb_writer = SummaryWriter('./logs')

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=collate_fn,
                                   num_workers=2,
                                   shuffle=True)

    total_steps = len(train_data_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps
    )

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1, n_no_improve = 0, 0
    model.zero_grad()
    train_iterator = trange(int(EPOCHS), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_data_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to('cuda') for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": True
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss = criterion(logits, labels)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # 定期记录日志
            logging_steps = 100
            if global_step % logging_steps == 0:
                logs = {}
                results = evaluate(model, evaluate_dataset, criterion)
                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value

                loss_scalar = (tr_loss - logging_loss) / logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                logger.append(json.dumps({**logs, **{"step": global_step}}))

            # 定期保存模型以防万一
            save_steps = 1000
            if global_step % save_steps == 0:
                output_dir = os.path.join('model', "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'mmbt_base.pth'))
                logger.append("Saving model checkpoint to {}".format(output_dir))

        # 每轮迭代后，evaluate，并保存模型
        torch.save(model.state_dict(), os.path.join('model', f'mmbt_base_{_}.pth'))
        results = evaluate(model, evaluate_dataset, criterion)

        if results["micro_f1"] > best_f1:
            best_f1 = results["micro_f1"]
            n_no_improve = 0
        else:
            n_no_improve += 1

        # 设定一个耐心值，当超出耐心值且模型无性能提升，则停止训练
        patience = 100
        if n_no_improve > patience:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename='./log/mmbt_v1.log'
    )

    logger.append("***** Initialize *****")
    cache_dir = 'cache'
    config = AutoConfig.from_pretrained('bert-base-chinese', cache_dir=cache_dir)
    mmbt_config = MMBTConfig(config, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', cache_dir=cache_dir)
    transformer = AutoModel.from_pretrained('bert-base-chinese', cache_dir=cache_dir)

    img_encoder = ImageEncoder()

    output_dir = 'output'
    model_name = 'mmbt_base.pth'

    model = MMBTForClassification(mmbt_config, transformer, img_encoder)
    model = model.to('cuda')

    data_path = './data/train.csv'
    # 先抽样1000条跑通
    train_dataset = load_examples(data_path, tokenizer)
    #
    eval_path = './data/test.csv'
    eval_dataset = load_examples(eval_path, tokenizer)

    criterion = nn.CrossEntropyLoss()

    EPOCHS = 10

    lr = 5e-5

    global_step, tr_loss = train(train_dataset, eval_dataset, model, criterion, EPOCHS, lr)
    logger.append(f" global_step = {global_step}, average loss = {tr_loss}")

    # 保存最优模型
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, model_name))
    tokenizer.save_pretrained(output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = MMBTForClassification(config, transformer, img_encoder)
    model.load_state_dict(torch.load(os.path.join(output_dir, model_name)))
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.to(device)


if __name__ == '__main__':
    main()
