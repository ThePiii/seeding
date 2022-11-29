import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
import os
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
# import torchvision.transforms as transforms
import torchvision
from utils import url_to_pilimg_train, Logger
from mmbt_base import get_image_transforms
from imgProcess import ImageProcess

# 3.x更新到4以后，不false会有一直有个log，不方便看输出
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (
    2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

img = ImageProcess()

logger = Logger(f'logs/bert_cat.log', resume=True)


# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         model_path = r'D:\研究生\毕业论文\MMBT\cache\torch\hub\checkpoints\resnet152-394f9c45.pth'
#         if os.path.exists(model_path):
#             model = torchvision.models.resnet152(pretrained=False)
#             model.load_state_dict(torch.load(r'D:\研究生\毕业论文\MMBT\cache\torch\hub\checkpoints\resnet152-394f9c45.pth'))
#         else:
#             model = torchvision.models.resnet50(pretrained=True)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         # self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[1])
#
#     def forward(self, x):
#         # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
#         # out = self.pool(self.model(x))
#         out = self.model(x)
#         out = torch.flatten(out, start_dim=2)
#         # out = out.transpose(1, 2).contiguous()
#         return out  # BxNx2048


class bert_dataset(Dataset):
    def __init__(self, contents, targets, tokenizer, url, transform, max_len=256):
        # self.dataset = pd.read_csv(dataroot).dropna()
        self.contents = contents
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.url = url
        self.transforms = transform

    def __getitem__(self, item):
        content = str(self.contents[item])
        target = self.targets[item]
        encoding = self.tokenizer(
            content,  # 分词文本
            padding='max_length',  # 保证长度一致才可以torch.stack
            max_length=self.max_len,  # 分词最大长度
            add_special_tokens=True,  # 添加特殊tokens['cls'], ['sep']
            return_token_type_ids=False,  # 返回是前一句还是后一句
            return_attention_mask=True,  # 返回attention_mask
            return_tensors='pt',  # 返回pytorch类型的tensor
            truncation=True  # 若长度大于max则切段
        )
        image = [url_to_pilimg_train(self.url[item].split(',')[0])]
        image = img.preprocess(image)
        image = img.run(image)

        return {
            'content_text': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
            'image': image
        }

    def __len__(self):
        return len(self.contents)


def data_loader(df, tokenizer, transform, max_len, batch_size):
    ds = bert_dataset(
        contents=df.content.to_numpy(),
        targets=df.is_commercial.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        url=df.trendImageUrl.to_numpy(),
        transform=transform
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-xlnet-base',
                                              return_dict=False, cache_dir='cache', output_hidden_states=True)  # 不加return_dict=False的话，pooled_output返回的是str
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size +
                             768, n_classes)  # hidden_size = 768
        self.project = nn.Linear(2048*1, 768)

    def forward(self, input_ids, attention_mask, image):
        # with torch.no_grad():
        # 微调Bert
        with torch.no_grad():
            _, pooled_output, hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        # first_last_avg
        if True:
            pooled_output = []
            for i in range(2):
                seq = hidden[-i]
                pooled_output += [torch.mean(seq, dim=1, keepdim=True)]
            pooled_output = torch.sum(torch.cat(pooled_output, dim=1), 1)

        image_feature = self.project(image)
        image_feature = image_feature.reshape(-1, 768)

        output = torch.cat((pooled_output, image_feature),
                           dim=-1).type(torch.FloatTensor).to(device)
        output = self.drop(output).to(device)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model.train()
    losses = []
    correct_pred = 0
    prob_all = []
    label_all = []
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        image = d['image'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, image=image).to(device)

        prob = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets).to(device)

        label_all.extend(targets.cpu().numpy())
        prob_all.extend(prob[:, 1].detach().cpu().numpy())
        correct_pred += torch.sum(preds == targets)
        losses.append(loss.item())
        if len(losses) % 500 == 0:
            logger.append(f'当前batch的损失为：{loss.item()}')

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度截断
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    auc = roc_auc_score(label_all, prob_all)
    logger.append("AUC:{:.4f}".format(auc))

    return correct_pred.double() / n_examples, np.mean(losses), auc


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            image = d["image"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    real_values = torch.stack(real_values)

    return predictions, prediction_probs, real_values


if __name__ == '__main__':
    # 训练太慢了，先抽样训练

    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-chinese', cache_dir='cache')
    batch_size = 16
    max_len = 256
    EPOCHS = 10
    train, test = True, True
    df_train = pd.read_csv('./data/train_5w.csv', lineterminator='\n')
    df_train = df_train.fillna('')
    # df_train = df_train.rename(columns={'trendImageUrl\r': 'trendImageUrl'})
    transform = get_image_transforms()

    if train:
        train_data_loader = data_loader(
            df_train, tokenizer, transform, max_len, batch_size)
        model = BertClassifier(2).to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(device)

        best_accuracy = 0
        for epoch in range(EPOCHS):
            start = time.time()
            logger.append(f'Epoch {epoch + 1}/{EPOCHS}')
            logger.append('-' * 30)

            train_acc, train_loss, auc = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                scheduler,
                len(df_train)
            )

            logger.append(f'Train loss {train_loss} accuracy {train_acc}')

            end = time.time()
            logger.append("本轮训练时间{}".format(end - start))

            torch.save(model.state_dict(),
                       './model/bert_cat.pth')
            best_accuracy = train_acc

    # if test:
    #     for epoch in range(EPOCHS):
    #         model = BertClassifier(2).to(device)
    #         model.load_state_dict(torch.load(
    #             f'./model/Bert-base/bert_base_{epoch}.pth'))
            df_test = pd.read_csv('./data/test_big.csv', lineterminator='\n')
            test_data_loader = data_loader(
                df_test, tokenizer, transform, max_len, batch_size)

            y_pred, y_pred_probs, y_test = get_predictions(
                model, test_data_loader)

            y_pred = y_pred.cpu()
            y_test = y_test.cpu()
            y_pred_probs = y_pred_probs[:, 1].cpu()

            logger.append("Test_AUC:{:.4f}".format(roc_auc_score(y_test, y_pred_probs)))
            logger.append(classification_report(y_test, y_pred, digits=4))

            cm = confusion_matrix(y_test, y_pred)
            class_names = ['0', '1']
            df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            # show_confusion_matrix(df_cm)
            logger.append(df_cm)
