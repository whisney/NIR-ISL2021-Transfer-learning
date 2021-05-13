import os
import numpy as np
import torch
from Networks.Local_network import Local_UNet
import segmentation_models_pytorch as smp
import argparse
import skimage.io as io
import cv2
from skimage import exposure
from utils import keep_large_area, fit_Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='none', help='which gpu is used, if cpu fill in none')
parser.add_argument('--img_path', type=str, help='input image path')
parser.add_argument('--save_dir', type=str, help='save output dir')
parser.add_argument('--model', type=str, default='Asia',
                    help='choose which model to predict image, options are Asia, Africa and M1')
args = parser.parse_args()
if args.gpu.lower() != 'none':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.model.lower() == 'asia':
    seg_model_path = [
        'trained_models/Asia/Seg/baseline_UNet_ResNet34/bs16_epoch500_fold0/best_dice.pth',
        'trained_models/Asia/Seg/finetune_UNet_ResNet34/bs16_epoch200_fold1/best_acc.pth',
        'trained_models/Asia/Seg/finetune_UNet_ResNet34/bs16_epoch200_fold2/best_acc.pth',
        'trained_models/Asia/Seg/baseline_UNet_ResNet34/bs16_epoch500_fold3/best_dice.pth',
        'trained_models/Asia/Seg/baseline_UNet_ResNet34/bs16_epoch500_fold4/best_dice.pth'
    ]
    local_model_path = [
        'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold0/best_model.pth',
        'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold1/best_model.pth',
        'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold2/best_model.pth',
        'trained_models/Asia/Local_inner_outer/baseline_UNet_ResNet34/bs16_epoch500_fold3/best_inner_dice.pth',
        'trained_models/Asia/Local_inner_outer/baseline_UNet_ResNet34/bs16_epoch500_fold4/best_inner_dice.pth'
    ]
    input_size = (480, 640)
elif args.model.lower() == 'africa':
    seg_model_path = [
        'trained_models/Africa/Seg/finetune_UNet_ResNet34/bs16_epoch200_fold0/best_acc.pth',
        'trained_models/Africa/Seg/baseline_UNet_ResNet34/bs16_epoch500_fold1/best_dice.pth',
        'trained_models/Africa/Seg/finetune_UNet_ResNet34/bs16_epoch200_fold2/best_acc.pth',
        'trained_models/Africa/Seg/baseline_UNet_ResNet34/bs16_epoch500_fold3/best_dice.pth',
        'trained_models/Africa/Seg/finetune_UNet_ResNet34/bs16_epoch200_fold4/best_acc.pth'
    ]
    local_model_path = [
        'trained_models/Africa/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold0/best_model.pth',
        'trained_models/Africa/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold1/best_model.pth',
        'trained_models/Africa/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold2/best_model.pth',
        'trained_models/Africa/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold3/best_model.pth',
        'trained_models/Africa/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold4/best_model.pth'
    ]
    input_size = (384, 640)
elif args.model.lower() == 'm1':
    seg_model_path = [
        'trained_models/M1/Seg/finetune_UNet_ResNet34/bs28_epoch100_fold0/best_acc.pth',
        'trained_models/M1/Seg/finetune_UNet_ResNet34/bs28_epoch100_fold1/best_acc.pth',
        'trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold2/best_acc.pth',
        'trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold3/best_acc.pth',
        'trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold4/best_acc.pth'
    ]
    local_model_path = [
        'trained_models/M1/Local_inner_outer/finetune_UNet_ResNet34/bs24_epoch150_fold0/best_model.pth',
        'trained_models/M1/Local_inner_outer/finetune_UNet_ResNet34/bs24_epoch150_fold1/best_model.pth',
        'trained_models/M1/Local_inner_outer/finetune_UNet_ResNet34/bs24_epoch150_fold2/best_model.pth',
        'trained_models/M1/Local_inner_outer/finetune_UNet_ResNet34/bs24_epoch150_fold3/best_model.pth',
        'trained_models/M1/Local_inner_outer/finetune_UNet_ResNet34/bs24_epoch150_fold4/best_model.pth'
    ]
    input_size = (416, 416)

os.makedirs(args.save_dir, exist_ok=True)

if args.gpu.lower() != 'none':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

seg_net_all = []
for path in seg_model_path:
    net = smp.Unet('resnet34', in_channels=1, classes=1, activation=None).to(device)
    if args.gpu.lower() == 'none':
        net.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(path))
    net.eval()
    seg_net_all.append(net)

local_net_all = []
for path in local_model_path:
    net = Local_UNet(encoder_name='resnet34', in_channels=1, out_channels_1=1, out_channels_2=1).to(device)
    if args.gpu.lower() == 'none':
        net.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(path))
    net.eval()
    local_net_all.append(net)

basename = os.path.splitext(os.path.basename(args.img_path))[0]

img = io.imread(args.img_path, as_gray=True).astype(np.float)
if img.max() > 1:
    img /= 255.
original_shape = img.shape
img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)
seg_prediction = np.zeros(img.shape)
inner_prediction = np.zeros(img.shape)
outer_prediction = np.zeros(img.shape)
for t in range(4):
    if t == 0:
        img_tta = img
    elif t == 1:
        img_tta = np.flip(img, axis=1)
    elif t == 2:
        img_tta = exposure.adjust_gamma(img, 1.2)
    elif t == 3:
        img_tta = exposure.adjust_gamma(img, 0.8)
    img_tta = np.ascontiguousarray(img_tta)
    img_tta = torch.from_numpy(img_tta).unsqueeze(0).unsqueeze(0).float().to(device)
    for net in seg_net_all:
        prediction_one = net(img_tta)
        prediction_one = torch.sigmoid(prediction_one).cpu().squeeze(0).squeeze(0).detach().numpy()
        if t == 1:
            prediction_one = np.flip(prediction_one, axis=1)
        seg_prediction += prediction_one
    for net in local_net_all:
        inner_one, outer_one = net(img_tta)
        inner_one = torch.sigmoid(inner_one).cpu().squeeze(0).squeeze(0).detach().numpy()
        outer_one = torch.sigmoid(outer_one).cpu().squeeze(0).squeeze(0).detach().numpy()
        if t == 1:
            inner_one = np.flip(inner_one, axis=1)
            outer_one = np.flip(outer_one, axis=1)
        inner_prediction += inner_one
        outer_prediction += outer_one
seg_prediction /= 20
seg_prediction = cv2.resize(seg_prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
_, seg_prediction = cv2.threshold(seg_prediction, 0.5, 1, 0)
seg_prediction = keep_large_area(seg_prediction, top_n_large=1)
io.imsave(os.path.join(args.save_dir, '{}_seg.png'.format(basename)), (seg_prediction * 255).astype(np.uint8))

inner_prediction /= 20
outer_prediction /= 20
inner_prediction = cv2.resize(inner_prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
outer_prediction = cv2.resize(outer_prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
_, inner_prediction = cv2.threshold(inner_prediction, 0.5, 1, 0)
_, outer_prediction = cv2.threshold(outer_prediction, 0.5, 1, 0)
inner_prediction = keep_large_area(inner_prediction, top_n_large=1)
outer_prediction = keep_large_area(outer_prediction, top_n_large=1)
inner_prediction = fit_Ellipse(inner_prediction).astype(np.uint8)
outer_prediction = fit_Ellipse(outer_prediction).astype(np.uint8)
inner_save = np.zeros(inner_prediction.shape)
contours, _ = cv2.findContours(inner_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(inner_save, contours, -1, 255, 1)
outer_save = np.zeros(outer_prediction.shape)
contours, _ = cv2.findContours(outer_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(outer_save, contours, -1, 255, 1)
io.imsave(os.path.join(args.save_dir, '{}_inner_boundary.png'.format(basename)), inner_save.astype(np.uint8))
io.imsave(os.path.join(args.save_dir, '{}_outer_boundary.png'.format(basename)), outer_save.astype(np.uint8))