import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
import torchvision.transforms as transforms
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
    BertModel,
    AutoConfig,
    MMBTConfig
)
from transformers.trainer_utils import is_main_process
from mmbt_example.utils_mmimdb import ImageEncoder

model = MMBTModel()


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def load_datasets():
    transforms = get_image_transforms()
    # 可以将图像统一处理后，保存为npy文件，以免每次迭代的时候再重新infer图像
    image_feature = np.load('')
    # 文本内容保存在csv文件即可
    text = pd.read_csv('')
    # 最后返回的dataset应该为dataframe格式，一列sentence保存文本，一列img保存图像
    dataset = []

    return dataset


def collect_fn(batch):
    lens = [len(row['setence']) for row in batch]
    batch_size, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    # 注意collate_fn所得到的tensor列表，与train/eval中的inputs 模型输入[5]相对应，这里再次给出注释如下，会与后面的模型侧使用对应。
    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


transformer = BertModel.from_pretrained('bert-base-chinese')
encoder = ImageEncoder()
# MMBTConfig会进一步在原有的transformer_config中，增加modal_hidden_size=2048，以及（可选添加）num_labels标签数量，补充了更多信息。
config = MMBTConfig(AutoConfig.from_pretrained('bert-base-chinese'), num_labels=2)
mmbt = MMBTModel(config, transformer, encoder)