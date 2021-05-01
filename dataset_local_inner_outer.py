from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, PadIfNeeded, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, RandomGamma, LongestMaxSize, GaussNoise, Resize, VerticalFlip)
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
        local_inner = io.imread(os.path.join(self.data_root, dirname, 'pupil_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        local_outer = io.imread(os.path.join(self.data_root, dirname, 'iris_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        _, local_inner = cv2.threshold(local_inner, 127, 1, 0)
        _, local_outer = cv2.threshold(local_outer, 127, 1, 0)
        local = np.concatenate((local_inner[:, :, np.newaxis],
                                    local_outer[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=img, mask=local)
        img, local = augmented['image'], augmented['mask']
        local_inner = local[:, :, 0]
        local_outer = local[:, :, 1]
        img = torch.from_numpy(img).float().unsqueeze(0)
        local_inner = torch.from_numpy(local_inner).float().unsqueeze(0)
        local_outer = torch.from_numpy(local_outer).float().unsqueeze(0)
        return img, local_inner, local_outer

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
        local_inner = io.imread(os.path.join(self.data_root, dirname, 'pupil_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        local_outer = io.imread(os.path.join(self.data_root, dirname, 'iris_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        _, local_inner = cv2.threshold(local_inner, 127, 1, 0)
        _, local_outer = cv2.threshold(local_outer, 127, 1, 0)
        local = np.concatenate((local_inner[:, :, np.newaxis],
                                    local_outer[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=img, mask=local)
        img, local = augmented['image'], augmented['mask']
        local_inner = local[:, :, 0]
        local_outer = local[:, :, 1]
        img = torch.from_numpy(img).float().unsqueeze(0)
        local_inner = torch.from_numpy(local_inner).float().unsqueeze(0)
        local_outer = torch.from_numpy(local_outer).float().unsqueeze(0)
        return img, local_inner, local_outer

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_val(data_root=r'I:\public_data\NIR-ISL2021',
                               split_file=r'E:\competition\NIR-ISL2021\preprocess/CASIA_Iris_Africa_split_20210305.pkl',
                               size=(384, 640), fold=0, resize=True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, (img, local_inner, local_outer) in enumerate(train_dataloader):
        img = img[0, 0, :, :].numpy() * 255
        local_inner = local_inner[0, 0, :, :].numpy() * 255
        local_outer = local_outer[0, 0, :, :].numpy() * 255
        plt.subplot(221)
        plt.imshow(img, cmap='gray')
        plt.subplot(223)
        plt.imshow(local_inner, cmap='gray')
        plt.subplot(224)
        plt.imshow(local_outer, cmap='gray')
        plt.show()