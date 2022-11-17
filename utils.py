from func_timeout import func_set_timeout
import requests
from io import BytesIO
# from PIL import Image
# import torch
# import torchvision
from torch import nn
# import random
from ops.transforms import *
import torchvision.transforms as transforms
import os
import traceback

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}


def url_to_pilimg_train(inputurl, timeout=10):
    @func_set_timeout(timeout)
    def run(url):
        try:
            response = requests.get(url, timeout=timeout)
            image = Image.open(BytesIO(response.content))
            image = image.convert('RGB')
            # print('图像正常下载')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        return image

    try:
        image = run(inputurl)
    except:
        print('图像下载超时')
        image = Image.new('RGB', (224, 224), (255, 255, 255))
    return image


# 图像拼接预处理
def preprocess(imgs, num_segments):
    '''
    :param imgs: 图像列表
    :param num_segments: 最大图像个数
    :param transform: 图像归一化
    :return: 拼接后的图像
    '''
    if len(imgs) < num_segments:
        padimg = Image.new('RGB', (224, 224), (0, 0, 0))
        pad_front = (num_segments - len(imgs)) // 2
        pad_end = (num_segments - len(imgs)) - pad_front
        imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
    else:
        random.shuffle(imgs)
        imgs = imgs[:num_segments]
    process_data = transform(imgs)

    return process_data


# 图像拼接用
cropping = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
        ])

transform = torchvision.transforms.Compose([
    cropping,
    Stack(roll=('resnet152' in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=('resnet152' not in ['BNInception', 'InceptionV3'])),
    GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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


# 尝试定义一个简单的图像拼接函数
def img_concat(imgs):
    '''

    :param imgs: list of images
    :return: concated image
    '''


class Logger(object):   # 保存各种日志信息到本地
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath,resume=False):
        self.file = None
        self.resume = resume
        if os.path.isfile(fpath):
            if resume:
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
