import os
import numpy as np
import torch
from Networks.Local_network import Local_UNet
import shutil
import argparse
import pickle
import pandas as pd
from albumentations import LongestMaxSize
import skimage.io as io
import cv2
from skimage import exposure
from utils import padding_2D, keep_large_area, fit_Ellipse
from medpy.metric import hd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--net', type=str, default='baseline', help='net')
parser.add_argument('--model_path', type=str, help='trained model path')
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--dataset', type=str, default='Asia', help='Asia/Africa/M1/All')
parser.add_argument('--TTA', type=int, default=1, help='1/0')
parser.add_argument('--ellipse', type=int, default=1, help='1/0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
if args.dataset.lower() == 'asia':
    split_file = 'data/CASIA_Iris_Asia_split_20210305.pkl'
    input_size = (480, 640)
    resize = True
elif args.dataset.lower() == 'africa':
    split_file = 'data/CASIA_Iris_Africa_split_20210305.pkl'
    input_size = (384, 640)
    resize = True
elif args.dataset.lower() == 'm1':
    split_file = 'data/CASIA_Iris_M1_split_20210305.pkl'
    input_size = (416, 416)
    resize = True
elif args.dataset.lower() == 'all':
    split_file = 'data/CASIA_Iris_All_split_20210305.pkl'
    input_size = (512, 512)
    resize = False

pkl_data = pickle.load(open(split_file, 'rb'))
path_list = pkl_data[args.fold]['test']

new_dir = args.model_path.rstrip('.pth') + '_TTA{}_Ellipse{}'.format(args.TTA, args.ellipse)
print(new_dir)
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.makedirs(os.path.join(new_dir, 'inner_predictions'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'outer_predictions'), exist_ok=True)
excel_path = os.path.join(new_dir, 'evaluation.xlsx')

if args.net.lower() == 'unet_resnet34':
    net = Local_UNet(encoder_name='resnet34', in_channels=1, out_channels_1=1, out_channels_2=1).cuda()
if args.net.lower() == 'unet_resnet101':
    net = Local_UNet(encoder_name='resnet101', in_channels=1, out_channels_1=1, out_channels_2=1).cuda()

net.load_state_dict(torch.load(args.model_path))
net.eval()

def compute_dice(pred, real):
    intersection = pred * real
    dice_sco = (2 * intersection.sum() + 1e-5) / (pred.sum() + real.sum() + 1e-5)
    return dice_sco

name_all = []
Dice_all = []
HD_all = []

with torch.no_grad():
    for i, path in enumerate(path_list):
        img_name = path.split('/')[-1]
        dirname = os.path.dirname(path)
        img = io.imread(os.path.join(data_root, dirname, 'image', img_name), as_gray=True)
        local_inner = io.imread(os.path.join(data_root, dirname, 'pupil_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        local_outer = io.imread(os.path.join(data_root, dirname, 'iris_edge_mask', os.path.splitext(img_name)[0]
                                             + '.png'), as_gray=True)
        _, local_inner = cv2.threshold(local_inner, 127, 1, 0)
        _, local_outer = cv2.threshold(local_outer, 127, 1, 0)
        original_shape = img.shape
        if resize:
            img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            transforms = LongestMaxSize(max_size=max(input_size))
            img = transforms(image=img)['image']
            transformed_shape = img.shape
            img, pads = padding_2D(img)
        inner_prediction = np.zeros(img.shape)
        outer_prediction = np.zeros(img.shape)
        tta = 1
        if args.TTA:
            tta = 4
        for t in range(tta):
            if t == 0:
                img_tta = img
            elif t == 1:
                img_tta = np.flip(img, axis=1)
            elif t == 2:
                img_tta = exposure.adjust_gamma(img, 1.2)
            elif t == 3:
                img_tta = exposure.adjust_gamma(img, 0.8)
            img_tta = np.ascontiguousarray(img_tta)
            img_tta = torch.from_numpy(img_tta).unsqueeze(0).unsqueeze(0).float().cuda()
            inner_one, outer_one = net(img_tta)
            inner_one = torch.sigmoid(inner_one).cpu().squeeze(0).squeeze(0).detach().numpy()
            outer_one = torch.sigmoid(outer_one).cpu().squeeze(0).squeeze(0).detach().numpy()
            if t == 1:
                inner_one = np.flip(inner_one, axis=1)
                outer_one = np.flip(outer_one, axis=1)
            inner_prediction += inner_one
            outer_prediction += outer_one
        inner_prediction /= tta
        outer_prediction /= tta
        if resize:
            inner_prediction = cv2.resize(inner_prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            outer_prediction = cv2.resize(outer_prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            inner_prediction = inner_prediction[pads[0]: pads[0] + transformed_shape[0],
                             pads[1]: pads[1] + transformed_shape[1]]
            inner_prediction = cv2.resize(inner_prediction, (original_shape[1], original_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            outer_prediction = outer_prediction[pads[0]: pads[0] + transformed_shape[0],
                             pads[1]: pads[1] + transformed_shape[1]]
            outer_prediction = cv2.resize(outer_prediction, (original_shape[1], original_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
        _, inner_prediction = cv2.threshold(inner_prediction, 0.5, 1, 0)
        _, outer_prediction = cv2.threshold(outer_prediction, 0.5, 1, 0)
        inner_prediction = keep_large_area(inner_prediction, top_n_large=1)
        outer_prediction = keep_large_area(outer_prediction, top_n_large=1)
        if args.ellipse:
            inner_prediction = fit_Ellipse(inner_prediction).astype(np.uint8)
            outer_prediction = fit_Ellipse(outer_prediction).astype(np.uint8)
        Dice = compute_dice(outer_prediction - inner_prediction, local_outer - local_inner)
        HD = hd(outer_prediction - inner_prediction, local_outer - local_inner)
        print(i, img_name, Dice, HD)
        name_all.append(path)
        Dice_all.append(Dice)
        HD_all.append(HD)
        inner_save = np.zeros(inner_prediction.shape)
        contours, _ = cv2.findContours(inner_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(inner_save, contours, -1, 255, 1)
        outer_save = np.zeros(outer_prediction.shape)
        contours, _ = cv2.findContours(outer_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(outer_save, contours, -1, 255, 1)
        io.imsave(os.path.join(new_dir, 'inner_predictions', os.path.splitext(img_name)[0] + '.png'),
                  inner_save.astype(np.uint8))
        io.imsave(os.path.join(new_dir, 'outer_predictions', os.path.splitext(img_name)[0] + '.png'),
                  outer_save.astype(np.uint8))

Dice_mean = np.mean(Dice_all)
HD_mean = np.mean(HD_all)
print('mean', Dice_mean, HD_mean)
name_all.append('mean')
Dice_all.append(Dice_mean)
HD_all.append(HD_mean)
df = pd.DataFrame({'image': name_all, 'Dice': Dice_all, 'HD': HD_all})
df.to_excel(excel_path, index=False)