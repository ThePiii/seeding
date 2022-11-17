import torch
import torchvision
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import nn
from utils import url_to_pilimg_train, preprocess, get_image_transforms
from transformers import AdamW, get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_image_transforms()


class image_dataset(Dataset):
    def __init__(self, targets, urls):
        self.targets = targets
        self.urls = urls

    def __getitem__(self, item):
        target = self.targets[item]
        urls = self.urls[item].split(',')

        image = url_to_pilimg_train(np.random.choice(urls))  # 一张图像
        image = transform(image)

        # 多图拼接
        # imgs = []
        # for url in urls:
        #     img = url_to_pilimg_train(url)
        #     imgs.append(img)
        # image = preprocess(imgs)

        return {
            'targets': torch.tensor(target, dtype=torch.long),
            'image': image
        }

    def __len__(self):
        return len(self.targets)


class ResNetClassfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        self.model = torchvision.models.resnet152(pretrained=False)
        self.model.load_state_dict(torch.load(r'D:\研究生\毕业论文\MMBT\cache\torch\hub\checkpoints\resnet152-394f9c45.pth'))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(1000, 2)

    def forward(self, image):
        with torch.no_grad():
            output = self.model(image)

        output = self.dropout(output)
        output = self.classifier(output)

        return output


def data_loader(df, batch_size):

    targets = df['is_commercial']
    urls = df['trendImageUrl']

    ds = image_dataset(targets, urls)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )


def train(model, data_loader, loss_fn, optimizer, scheduler):
    model.train()
    losses = []
    correct_pred = 0

    for batch_idx, d in enumerate(data_loader):

        targets = d['targets'].to(device)
        image = d['image'].to(device)

        outputs = model(image)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        correct_pred += torch.sum(preds == targets)
        losses.append(loss.item())
        if len(losses) % 1 == 0:
            print(f'当前batch的损失为：{loss.item()}')

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度截断
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_pred.double() / len(data_loader), np.mean(losses)


def eval_model(model, data_loader, loss_fn):
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

    return correct_predictions.double() / len(data_loader), np.mean(losses)


if __name__ == '__main__':

    batch_size = 16
    EPOCHS = 10

    df_train = pd.read_csv('./data/train.csv', lineterminator='\n').sample(100).reset_index(drop=True)
    train_data_loader = data_loader(df_train, batch_size=16)

    model = ResNetClassfier()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        start = time.time()
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 30)

        train_acc, train_loss = train(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        end = time.time()
        print("本轮训练时间{}".format(end - start))

        torch.save(model.state_dict(), f'./model/Bert-base/bert_base_{epoch}.pth')