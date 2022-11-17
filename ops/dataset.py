# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


from collections import *
import json
import random
import torch



class Filelist3DDataset(data.Dataset):
    '''read data path and label from a filelist'''
    def __init__(self, root, annofile, num_segments=3, transform=None) -> None:
        self.num_segments = num_segments
        self.datas = []
        # spus = []
        # spu_paths = defaultdict(list)

        trendids_pos = []
        trendids_neg = []
        trendid_paths = defaultdict(list)
        trendid_label = {}

        # import pdb; pdb.set_trace()
        with open(annofile) as f:
            for line in f.readlines():
                trendid, filepath, label = line.strip().split(',')
                # trendid = int(trendid)
                label = int(label)
                if label > 0:
                    trendids_pos.append(trendid)
                elif label == 0:
                    trendids_neg.append(trendid)
                # trendids.append(trendid)
                trendid_paths[trendid].append(filepath)
                trendid_label[trendid] = label
                # spus.append(int(spu))
                # spu_paths[int(spu)].append(os.path.join(root, filename))
                # spu_label[int(spu)] = label
        trendids_pos = list(set(trendids_pos))
        # trendids_pos = trendids_pos * 2
        trendids_neg = list(set(trendids_neg))
        trendids = trendids_pos + trendids_neg
        random.shuffle(trendids)
        for trendid in trendids:
            self.datas.append((trendid_paths[trendid], trendid_label[trendid]))
        self.transform = transform

    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        try:
            paths, label = self.datas[index]
            imgs = []
            for filepath in paths:
                try:
                    img=Image.open(filepath).convert('RGB')
                    imgs.append(img)
                except Exception as e:
                    index = random.randint(0, self.__len__()-1)
                    print('file broken: {}'.format(filepath))
            if len(imgs) < self.num_segments:
                # padimg = Image.new('RGB', imgs[-1].size, (0,0,0))
                # pad_front = (self.num_segments - len(imgs)) // 2
                # pad_end = (self.num_segments - len(imgs)) - pad_front
                # imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
                imgs = imgs * (self.num_segments//len(imgs)) ##帧数不够的话，复制帧
                padimg = Image.new('RGB', imgs[-1].size, (0,0,0))
                pad_front = (self.num_segments - len(imgs)) // 2
                pad_end = (self.num_segments - len(imgs)) - pad_front
                imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
            else:
                random.shuffle(imgs)
                imgs = imgs[:self.num_segments]
            process_data = self.transform(imgs)
            return process_data, label
        except Exception as e:
            img=Image.open('test.jpeg').convert('RGB')
            imgs = [img] * self.num_segments
            label = 0
            print(e)
            process_data = self.transform(imgs)
            return process_data, label



class Filelist3DPathDataset(data.Dataset):
    '''read data path and label from a filelist'''
    def __init__(self, root, annofile, num_segments=3, transform=None) -> None:
        self.num_segments = num_segments
        self.datas = []
        # spus = []
        # spu_paths = defaultdict(list)

        trendids_pos = []
        trendids_neg = []
        trendid_paths = defaultdict(list)
        trendid_label = {}

        # import pdb; pdb.set_trace()
        with open(annofile) as f:
            for line in f.readlines():
                trendid, filepath, label = line.strip().split(',')
                trendid = int(trendid)
                label = int(label)
                if label > 0:
                    trendids_pos.append(trendid)
                elif label == 0:
                    trendids_neg.append(trendid)
                # trendids.append(trendid)
                trendid_paths[trendid].append(filepath)
                trendid_label[trendid] = label
                # spus.append(int(spu))
                # spu_paths[int(spu)].append(os.path.join(root, filename))
                # spu_label[int(spu)] = label
        trendids_pos = list(set(trendids_pos))
        trendids_neg = list(set(trendids_neg))
        trendids = trendids_pos + trendids_neg
        for trendid in trendids:
            self.datas.append((trendid_paths[trendid], trendid_label[trendid]))
        self.transform = transform

    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        try:
            paths, label = self.datas[index]
            imgs = []
            for filepath in paths:
                try:
                    img=Image.open(filepath).convert('RGB')
                    imgs.append(img)
                except Exception as e:
                    index = random.randint(0, self.__len__()-1)
                    print('file broken: {}'.format(filepath))
            if len(imgs) < self.num_segments:
                imgs = imgs * (self.num_segments//len(imgs)) ##帧数不够的话，复制帧
                padimg = Image.new('RGB', imgs[-1].size, (0,0,0))
                pad_front = (self.num_segments - len(imgs)) // 2
                pad_end = (self.num_segments - len(imgs)) - pad_front
                imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
            else:
                random.shuffle(imgs)
                imgs = imgs[:self.num_segments]
            process_data = self.transform(imgs)
            return process_data, label, '-'.join(paths)
        except Exception as e:
            img=Image.open('test.jpeg').convert('RGB')
            imgs = [img] * self.num_segments
            label = 0
            print(e)
            process_data = self.transform(imgs)
            return process_data, label, ""





# class Filelist3DPathDataset(data.Dataset):
#     '''read data path and label from a filelist'''
#     def __init__(self, root, annofile, num_segments=3, transform=None) -> None:
#         self.num_segments = num_segments
#         self.datas = []
#         spus = []
#         spu_paths = defaultdict(list)
#         spu_label = {}

#         dictfilepath = os.path.join(os.path.dirname(annofile), "labeldict.txt")
#         if os.path.exists(dictfilepath):
#             with open(dictfilepath) as f:
#                 tag_label = json.load(f)
#         # import pdb; pdb.set_trace()
#         with open(annofile) as f:
#             for line in f.readlines():
#                 spu, filename, label = line.strip().split(',')
#                 spus.append(int(spu))
#                 spu_paths[int(spu)].append(os.path.join(root, filename))
#                 spu_label[int(spu)] = label
#         spus = list(set(spus))
#         for spu in spus:
#             if spu_label[spu] in tag_label:
#                 self.datas.append((spu_paths[spu], tag_label[spu_label[spu]]))
#         self.transform = transform

    
#     def __len__(self):
#         return len(self.datas)

#     def __getitem__(self, index):
#         try:
#             paths, label = self.datas[index]
#             imgs = []
#             for filepath in paths:
#                 try:
#                     img=Image.open(filepath).convert('RGB')
#                     # if self.transform:
#                         # img = self.transform(img)
#                     # img = torch.unsqueeze(img, 0)
#                     imgs.append(img)
#                 except Exception as e:
#                     index = random.randint(0, self.__len__()-1)
#                     print('file broken: {}'.format(filepath))
#                     print(e)
#             # print(len(imgs))
#             if len(imgs) < self.num_segments:
#                 # padimg = torch.zeros_like(imgs[-1])
#                 padimg = Image.new('RGB', imgs[-1].size, (0,0,0))
#                 pad_front = (self.num_segments - len(imgs)) // 2
#                 pad_end = (self.num_segments - len(imgs)) - pad_front
#                 imgs = [padimg] * pad_front + imgs + [padimg] * pad_end
#             else:
#                 random.shuffle(imgs)
#                 imgs = imgs[:self.num_segments]
#             # img = torch.stack(imgs, 1)
#             process_data = self.transform(imgs)
#             return process_data, label, '-'.join(paths)
#         except Exception as e:
#             img=Image.open('test.jpeg').convert('RGB')
#             # img = self.transform(img)
#             imgs = [img] * self.num_segments
#             label = 0
#             print(e)
#             process_data = self.transform(imgs)
#             return process_data, label, 'test.jpg'













class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
