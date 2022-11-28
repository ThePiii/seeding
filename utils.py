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
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss