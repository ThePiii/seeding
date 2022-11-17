from torch.nn import functional as F
from seeding.ops.models2 import TSN
from seeding.ops.transforms import *


class ImageProcess():
    def __init__(self):
        this_weights = r'D:\研究生\毕业论文\MMBT\seeding\checkpoint\45_ckpt.pth.tar'
        self.num_segments = 1
        is_shift = True
        shift_div = 6
        shift_place = 'blockres'
        modality = "RGB"
        this_arch = 'resnet50'
        self.num_class = 2
        input_size = 224

        self.net = TSN(self.num_class, self.num_segments if is_shift else 1, modality,
                       base_model=this_arch,
                       consensus_type='avg',
                       img_feature_dim=256,
                       pretrain='imagenet',
                       is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
                       non_local=False,
                       )
        checkpoint = torch.load(this_weights, map_location='cpu')
        checkpoint = checkpoint['state_dict']
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight', 'base_model.classifier.bias': 'new_fc.bias', }
        for k, v in replace_dict.items():
            if k in base_dict:
                base_dict[v] = base_dict.pop(k)
        self.net.load_state_dict(base_dict)

        cropping = torchvision.transforms.Compose([
            GroupScale(self.net.scale_size),
            GroupCenterCrop(input_size),
        ])
        self.transform = torchvision.transforms.Compose([
            cropping,
            Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
            GroupNormalize(self.net.input_mean, self.net.input_std),
        ])

    def preprocess(self, imgs):
        if len(imgs) < self.num_segments:
            padimg = Image.new('RGB', (224, 224), (0, 0, 0))
            pad_front = (self.num_segments - len(imgs)) // 2
            pad_end = (self.num_segments - len(imgs)) - pad_front
            imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
        else:
            random.shuffle(imgs)
            imgs = imgs[:self.num_segments]
        process_data = self.transform(imgs)
        return process_data

    def run(self, data):
        batch_size = 1
        num_crop = 1
        length = 3
        softmax = False
        with torch.no_grad():
            data_in = data.view(-1, length, data.size(1), data.size(2))
            data_in = data_in.view(batch_size * num_crop, self.num_segments, length, data_in.size(2), data_in.size(3))
            self.net.eval()
            rst = self.net(data_in)
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)
            if softmax:
                rst = F.softmax(rst, dim=1)
            rst = rst.data.cpu().numpy().copy()
            # print(rst.shape)
            # pred = rst.argmax()
            # print(pred)

        return rst


if __name__ == '__main__':
    import pandas as pd
    from PIL import Image

    file = pd.read_csv('./data/mml_expdata.csv')

    paths = file['img']

    im = ImageProcess()

    for path in paths:
        imgs = []
        for i in eval(path):
            img = Image.open(i)
            imgs.append(img)

        data = im.preprocess(imgs)
        rst = im.run(data)
