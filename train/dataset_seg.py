from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, PadIfNeeded, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, RandomGamma, LongestMaxSize, GaussNoise, Resize)
import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

class Dataset_train(Dataset):
    def __init__(self, data_root='data', split_file='', size=(256, 256), fold=0, resize=False):
        self.data_root = data_root
        pkl_data = pickle.load(open(split_file, 'rb'))
        if fold == -1:
            self.path_list = pkl_data[0]['train']
            self.path_list.extend(pkl_data[0]['val'])
        else:
            self.path_list = pkl_data[fold]['train']
        self.len = len(self.path_list)
        if resize:
            self.transforms = Compose([Resize(height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST),
                                       ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7,
                                                        border_mode=cv2.BORDER_CONSTANT, value=0),
                                       HorizontalFlip(p=0.5),
                                       OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                               border_mode=cv2.BORDER_CONSTANT, value=0),
                                              OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                                border_mode=cv2.BORDER_CONSTANT, value=0)], p=0.5),
                                       RandomGamma(gamma_limit=(80, 120), p=0.5),
                                       GaussNoise(var_limit=(0.02, 0.1), mean=0, p=0.5)
                                       ])
        else:
            self.transforms = Compose([LongestMaxSize(max_size=max(size)),
                                       PadIfNeeded(min_height=size[0], min_width=size[1], value=0,
                                                   border_mode=cv2.BORDER_CONSTANT),
                                       ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7,
                                                        border_mode=cv2.BORDER_CONSTANT, value=0),
                                       HorizontalFlip(p=0.5),
                                       OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                               border_mode=cv2.BORDER_CONSTANT, value=0),
                                              OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                                border_mode=cv2.BORDER_CONSTANT, value=0)], p=0.5),
                                       RandomGamma(gamma_limit=(80, 120), p=0.5),
                                       GaussNoise(var_limit=(0.02, 0.1), mean=0, p=0.5)
                                       ])

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img_name = path.split('/')[-1]
        dirname = os.path.dirname(path)
        img = io.imread(os.path.join(self.data_root, dirname, 'image', img_name), as_gray=True)
        seg = io.imread(os.path.join(self.data_root, dirname, 'SegmentationClass', os.path.splitext(img_name)[0] +
                                         '.png'), as_gray=True)
        _, seg = cv2.threshold(seg, 127, 1, 0)
        augmented = self.transforms(image=img, mask=seg)
        img, seg = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().unsqueeze(0)
        seg = torch.from_numpy(seg).float().unsqueeze(0)
        return img, seg

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, data_root='data', split_file='', size=(256, 256), fold=0, resize=False):
        self.data_root = data_root
        pkl_data = pickle.load(open(split_file, 'rb'))
        self.path_list = pkl_data[fold]['val']
        self.len = len(self.path_list)
        if resize:
            self.transforms = Resize(height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST)
        else:
            self.transforms = Compose([LongestMaxSize(max_size=max(size)),
                                       PadIfNeeded(min_height=size[0], min_width=size[1], value=0,
                                                   border_mode=cv2.BORDER_CONSTANT)
                                       ])

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img_name = path.split('/')[-1]
        dirname = os.path.dirname(path)
        img = io.imread(os.path.join(self.data_root, dirname, 'image', img_name), as_gray=True)
        seg = io.imread(os.path.join(self.data_root, dirname, 'SegmentationClass', os.path.splitext(img_name)[0] +
                                     '.png'), as_gray=True)
        _, seg = cv2.threshold(seg, 127, 1, 0)
        augmented = self.transforms(image=img, mask=seg)
        img, seg = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().unsqueeze(0)
        seg = torch.from_numpy(seg).float().unsqueeze(0)
        return img, seg

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_train(data_root=r'I:\public_data\NIR-ISL2021',
                               split_file=r'E:\competition\NIR-ISL2021\preprocess/CASIA_Iris_Africa_split_20210305.pkl',
                               size=(384, 640), fold=0, resize=True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, (img, seg) in enumerate(train_dataloader):
        img = img[0, 0, :, :].numpy() * 255
        seg = seg[0, 0, :, :].numpy() * 255
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.imshow(seg, cmap='gray')
        plt.show()