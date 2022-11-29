import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup, ErnieModel
import os
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from utils import Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 3.x更新到4以后，不false会有一直有个log，不方便看输出
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = Logger(f'logs/ERNIE.log', resume=True)


class bert_dataset(Dataset):
    def __init__(self, contents, targets, tokenizer, max_len=256):
        # self.dataset = pd.read_csv(dataroot).dropna()
        self.contents = contents
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        return {
            'content_text': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

    def __len__(self):
        return len(self.contents)


def data_loader(df, tokenizer, max_len, batch_size):
    ds = bert_dataset(
        contents=df.content.to_numpy(),
        targets=df.is_commercial.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = ErnieModel.from_pretrained("nghuyong/ernie-3.0-base-zh",
                                                     return_dict=False, cache_dir='cache')  # 不加return_dict=False的话，pooled_output返回的是str
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear((self.bert.config.hidden_size)//2, n_classes)  # hidden_size = 768
        self.linear = nn.Linear(self.bert.config.hidden_size, (self.bert.config.hidden_size)//2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            _, pooled_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        output = self.drop(pooled_output)
        output = self.linear(output)
        output = self.drop(output)
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).to(device)

        prob = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets).to(device)
        
        prob_all.extend(prob[:,1].cpu().numpy()) 
        label_all.extend(targets.cpu().numpy())
        correct_pred += torch.sum(preds == targets)
        losses.append(loss.item())
        if len(losses) % 1000 == 0:
            logger.append(f'当前batch的损失为：{loss.item()}')

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度截断
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    auc = roc_auc_score(label_all,prob_all)
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
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

    tokenizer = BertTokenizerFast.from_pretrained("nghuyong/ernie-3.0-base-zh", cache_dir='cache')
    batch_size = 16
    max_len = 256
    EPOCHS = 10
    train, test = True, True

    if train:
        df_train = pd.read_csv('./data/train_5w.csv', lineterminator='\n').sample(50)
        df_train = df_train.fillna('')

        train_data_loader = data_loader(df_train, tokenizer, max_len, batch_size)

        # data = next(iter(train_data_loader))
        # logger.append(data.keys())
        # input_ids = train_data_loader['input_ids']
        # attention_mask = train_data_loader['attention_mask']

        model = BertClassifier(2).to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(device)

        best_auc = 0
        for epoch in range(EPOCHS):
            start = time.time()
            logger.append(f'Epoch {epoch + 1}/{EPOCHS}')
            logger.append('-' * 30)

            train_acc, train_loss, train_auc = train_epoch(
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

            if train_auc > best_auc:
                torch.save(model.state_dict(), f'./model/ERNIE.pth')
                best_auc = train_auc

    # if test:
    #     for epoch in range(EPOCHS):
    #         model = BertClassifier(2).to(device)
    #         model.load_state_dict(torch.load(f'./model/Bert-base/ERNIE_{epoch}.pth'))
            logger.append('----- test -----')
            df_test = pd.read_csv('./data/test.csv', lineterminator='\n').sample(50)
            test_data_loader = data_loader(df_test, tokenizer, max_len, batch_size)

            y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

            y_pred = y_pred.cpu()
            y_test = y_test.cpu()
            y_pred_probs = y_pred_probs[:,1].cpu()
            
            logger.append("AUC:{:.4f}".format(roc_auc_score(y_test, y_pred_probs)))
            
            logger.append(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            class_names = ['0', '1']
            df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

            logger.append(df_cm)
