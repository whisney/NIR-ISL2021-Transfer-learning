import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import shutil
import argparse
import pickle
import pandas as pd
from albumentations import LongestMaxSize
import skimage.io as io
import cv2
from skimage import exposure
from utils import padding_2D, keep_large_area

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--net', type=str, default='unet_resnet34', help='net')
parser.add_argument('--model_path', type=str, help='trained model path')
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--dataset', type=str, default='Asia', help='Asia/Africa/M1/All')
parser.add_argument('--TTA', type=int, default=1, help='1/0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
if args.dataset.lower() == 'asia':
    split_file = 'data/CASIA_Iris_Asia_split.pkl'
    input_size = (480, 640)
    resize = True
elif args.dataset.lower() == 'africa':
    split_file = 'data/CASIA_Iris_Africa_split.pkl'
    input_size = (384, 640)
    resize = True
elif args.dataset.lower() == 'm1':
    split_file = 'data/CASIA_Iris_M1_split.pkl'
    input_size = (416, 416)
    resize = True
elif args.dataset.lower() == 'all':
    split_file = 'data/CASIA_Iris_All_split.pkl'
    input_size = (512, 512)
    resize = False

pkl_data = pickle.load(open(split_file, 'rb'))
path_list = pkl_data[args.fold]['test']

new_dir = args.model_path.rstrip('.pth') + '_TTA{}'.format(args.TTA)
print(new_dir)
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)
excel_path = os.path.join(new_dir, 'evaluation.xlsx')

if args.net.lower() == 'unet_resnet34':
    net = smp.Unet('resnet34', in_channels=1, classes=1, activation=None).cuda()
if args.net.lower() == 'unet_resnet101':
    net = smp.Unet('resnet101', in_channels=1, classes=1, activation=None).cuda()

net.load_state_dict(torch.load(args.model_path))
net.eval()

def compute_E1(pred, real):
    return (pred != real).sum() / pred.size

def compute_E2(pred, real):
    true_positives = np.copy(real)
    true_negatives = 1 - real
    error = (pred != real).astype(np.int)
    FPR = (error * true_negatives).sum() / pred.size
    FNR = (error * true_positives).sum() / pred.size
    return 0.5 * FPR + 0.5 * FNR

name_all = []
E1_all = []
E2_all = []

with torch.no_grad():
    for i, path in enumerate(path_list):
        img_name = path.split('/')[-1]
        dirname = os.path.dirname(path)
        img = io.imread(os.path.join(data_root, dirname, 'image', img_name), as_gray=True)
        seg = io.imread(os.path.join(data_root, dirname, 'SegmentationClass', os.path.splitext(img_name)[0] +
                                     '.png'), as_gray=True)
        _, seg = cv2.threshold(seg, 127, 1, 0)
        original_shape = img.shape
        if resize:
            img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            transforms = LongestMaxSize(max_size=max(input_size))
            img = transforms(image=img)['image']
            transformed_shape = img.shape
            img, pads = padding_2D(img)
        prediction = np.zeros(img.shape)
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
            prediction_one = net(img_tta)
            prediction_one = torch.sigmoid(prediction_one).cpu().squeeze(0).squeeze(0).detach().numpy()
            if t == 1:
                prediction_one = np.flip(prediction_one, axis=1)
            prediction += prediction_one
        prediction /= tta
        if resize:
            prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            prediction = prediction[pads[0]: pads[0] + transformed_shape[0], pads[1]: pads[1] + transformed_shape[1]]
            prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        _, prediction = cv2.threshold(prediction, 0.5, 1, 0)
        prediction = keep_large_area(prediction, top_n_large=1)
        E1 = compute_E1(prediction, seg)
        E2 = compute_E2(prediction, seg)
        print(i, img_name, E1, E2)
        name_all.append(path)
        E1_all.append(E1)
        E2_all.append(E2)
        io.imsave(os.path.join(new_dir, 'predictions', os.path.splitext(img_name)[0] + '.png'),
                  (prediction * 255).astype(np.uint8))
E1_mean = np.mean(E1_all)
E2_mean = np.mean(E2_all)
print('mean', E1_mean, E2_mean)
name_all.append('mean')
E1_all.append(E1_mean)
E2_all.append(E2_mean)
df = pd.DataFrame({'image': name_all, 'E1': E1_all, 'E2': E2_all})
df.to_excel(excel_path, index=False)