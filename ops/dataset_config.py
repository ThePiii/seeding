# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/ssd/video/'  # '/data/jilin/'


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_lxys(dodality):
    filename_categories = 21
    filename_imglist_train = 'data/data_1014/lxys3d/label_lxys_train.txt'
    filename_imglist_val = 'data/data_1014/lxys3d/label_lxys_val.txt'
    root_data = '/mnt/user/xuliang/data/commodity/data_tag/images_lxys'
    prefix = 'lxys'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_lxys29(dodality):
    filename_categories = 29
    filename_imglist_train = 'data/data_1014/lxys3d/label_lxys_train.txt'
    filename_imglist_val = 'data/data_1014/lxys3d/label_lxys_val.txt'
    # root_data = '/mnt/user/xuliang/data/commodity/data_tag/images_lxys'
    root_data = '/mnt/ssd/data/commodity/data_tag/images_lxys'
    prefix = 'lxys29'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy(modality):
    filename_categories = 2
    filename_imglist_train = 'data/anno_train.csv'
    filename_imglist_val = 'data/anno_val.csv'
    root_data = ''
    prefix = 'sexy'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy2(modality):
    filename_categories = 2
    filename_imglist_train = 'data/anno_train2.csv'
    filename_imglist_val = 'data/anno_val.csv'
    root_data = ''
    prefix = 'sexy2'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy_v2(modality):
    filename_categories = 2
    filename_imglist_train = 'data/anno_train_v2.csv'
    filename_imglist_val = 'data/anno_val_v2.csv'
    root_data = ''
    prefix = 'sexy_v2'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexyv2(modality):
    filename_categories = 2
    filename_imglist_train = 'data/anno_train_v2.csv'
    filename_imglist_val = 'data/anno_val_v2.csv'
    root_data = ''
    prefix = 'sexyv2'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexyv4(modality):
    filename_categories = 2
    filename_imglist_train = 'data/anno_train_v2.csv'
    filename_imglist_val = 'data/anno_val_v2.csv'
    root_data = ''
    prefix = 'sexyv4'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexyvideo(modality):
    filename_categories = 2
    filename_imglist_train = 'data/video/anno_train.csv'
    filename_imglist_val = 'data/video/anno_val.csv'
    root_data = ''
    prefix = 'sexyvideo'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexyvideo0616(modality):
    filename_categories = 2
    filename_imglist_train = 'data/video_0616/anno_train.csv'
    filename_imglist_val = 'data/video_0616/anno_val.csv'
    root_data = ''
    prefix = 'sexyvideo0616'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy05120608(modality):
    filename_categories = 2
    filename_imglist_train = 'data/image_0512_0608/anno_train.csv'
    filename_imglist_val = 'data/image_0512_0608/anno_val.csv'
    root_data = ''
    prefix = 'sexy05120608'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy06160626(modality):
    filename_categories = 2
    filename_imglist_train = 'data/image_0616_0626/anno_train.csv'
    filename_imglist_val = 'data/image_0616_0626/anno_val.csv'
    root_data = ''
    prefix = 'sexy06160626'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexy06160710(modality):
    filename_categories = 2
    filename_imglist_train = 'data/image_0616_0710/anno_train.csv'
    filename_imglist_val = 'data/image_0616_0710/anno_val.csv'
    root_data = ''
    prefix = 'sexy06160710'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_sexyvideo0710(modality):
    filename_categories = 2
    filename_imglist_train = 'data/video_0616_0710/anno_train.csv'
    filename_imglist_val = 'data/video_0616_0710/anno_val.csv'
    root_data = ''
    prefix = 'sexyvideo0710'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_sexy05120608pos2(modality):
    filename_categories = 2
    filename_imglist_train = 'data/image_0512_0608/anno_train.csv'
    filename_imglist_val = 'data/image_0512_0608/anno_val.csv'
    root_data = ''
    prefix = 'sexy05120608pos2'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_shangdan(modality):
    filename_categories = 2
    filename_imglist_train = 'data/shangdan/anno_train.csv'
    filename_imglist_val = 'data/shangdan/anno_val.csv'
    root_data = ''
    prefix = 'shangdan'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'lxys': return_lxys, 'lxys29': return_lxys29, 
                   'shangdan':return_shangdan
                   }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)
    return file_categories, file_imglist_train, file_imglist_val, root_data, prefix
    # if dataset == 'lxys' or dataset == 'lxys29':
    #     return file_categories, file_imglist_train, file_imglist_val, root_data, prefix
    
    # file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    # file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    # if isinstance(file_categories, str):
    #     file_categories = os.path.join(ROOT_DATASET, file_categories)
    #     with open(file_categories) as f:
    #         lines = f.readlines()
    #     categories = [item.rstrip() for item in lines]
    # else:  # number of categories
    #     categories = [None] * file_categories
    # n_class = len(categories)
    # print('{}: {} classes'.format(dataset, n_class))
    # return n_class, file_imglist_train, file_imglist_val, root_data, prefix
