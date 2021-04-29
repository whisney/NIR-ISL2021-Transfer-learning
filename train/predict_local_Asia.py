import os
import numpy as np
import torch
from Networks.Local_network import Local_UNet
import argparse
import skimage.io as io
import cv2
from skimage import exposure
from utils import keep_large_area, fit_Ellipse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
input_size = (480, 640)

Distance_name_list = os.listdir('data/testing/CASIA-Iris-Asia/CASIA-distance/test/image')
Occlusion_name_list = os.listdir('data/testing/CASIA-Iris-Asia/CASIA-Iris-Complex/Occlusion/test/image')
Off_angle_name_list = os.listdir('data/testing/CASIA-Iris-Asia/CASIA-Iris-Complex/Off_angle/test/image')

model_path = [
    'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold0/best_model.pth',
    'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold1/best_model.pth',
    'trained_models/Asia/Local_inner_outer/finetune_UNet_ResNet34/bs16_epoch200_fold2/best_model.pth',
    'trained_models/Asia/Local_inner_outer/baseline_UNet_ResNet34/bs16_epoch500_fold3/best_inner_dice.pth',
    'trained_models/Asia/Local_inner_outer/baseline_UNet_ResNet34/bs16_epoch500_fold4/best_inner_dice.pth'
]

new_dir = 'trained_models/NIR-ISL2021030401_1/CASIA-Iris-Asia'
Distance_save_dir = os.path.join(new_dir, 'CASIA-Iris-Distance')
Occlusion_save_dir = os.path.join(new_dir, 'CASIA-Iris-Complex-Occlusion')
Off_angle_save_dir = os.path.join(new_dir, 'CASIA-Iris-Complex-Off-angle')
os.makedirs(os.path.join(Distance_save_dir, 'Inner_Boundary'), exist_ok=True)
os.makedirs(os.path.join(Occlusion_save_dir, 'Inner_Boundary'), exist_ok=True)
os.makedirs(os.path.join(Off_angle_save_dir, 'Inner_Boundary'), exist_ok=True)
os.makedirs(os.path.join(Distance_save_dir, 'Outer_Boundary'), exist_ok=True)
os.makedirs(os.path.join(Occlusion_save_dir, 'Outer_Boundary'), exist_ok=True)
os.makedirs(os.path.join(Off_angle_save_dir, 'Outer_Boundary'), exist_ok=True)

img_name_dicts = [{'data_dir': 'data/testing/CASIA-Iris-Asia/CASIA-distance/test/image', 'save_dir': Distance_save_dir,
                   'name_list': Distance_name_list},
                  {'data_dir': 'data/testing/CASIA-Iris-Asia/CASIA-Iris-Complex/Occlusion/test/image',
                   'save_dir': Occlusion_save_dir, 'name_list': Occlusion_name_list},
                  {'data_dir': 'data/testing/CASIA-Iris-Asia/CASIA-Iris-Complex/Off_angle/test/image',
                   'save_dir': Off_angle_save_dir, 'name_list': Off_angle_name_list}]

net_all = []
for path in model_path:
    net = Local_UNet(encoder_name='resnet34', in_channels=1, out_channels_1=1, out_channels_2=1).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    net_all.append(net)


with torch.no_grad():
    for j, img_name_dict in enumerate(img_name_dicts):
        data_dir = img_name_dict['data_dir']
        save_dir = img_name_dict['save_dir']
        name_list = img_name_dict['name_list']
        for i, name in enumerate(name_list):
            print('{}/{} {}/{}: {}'.format(j + 1, len(img_name_dicts), i + 1, len(name_list), name))
            img = io.imread(os.path.join(data_dir, name), as_gray=True)
            original_shape = img.shape
            img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)
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
                img_tta = torch.from_numpy(img_tta).unsqueeze(0).unsqueeze(0).float().cuda()
                for net in net_all:
                    inner_one, outer_one = net(img_tta)
                    inner_one = torch.sigmoid(inner_one).cpu().squeeze(0).squeeze(0).detach().numpy()
                    outer_one = torch.sigmoid(outer_one).cpu().squeeze(0).squeeze(0).detach().numpy()
                    if t == 1:
                        inner_one = np.flip(inner_one, axis=1)
                        outer_one = np.flip(outer_one, axis=1)
                    inner_prediction += inner_one
                    outer_prediction += outer_one
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
            io.imsave(os.path.join(save_dir, 'Inner_Boundary', os.path.splitext(name)[0] + '.png'),
                      inner_save.astype(np.uint8))
            io.imsave(os.path.join(save_dir, 'Outer_Boundary', os.path.splitext(name)[0] + '.png'),
                      outer_save.astype(np.uint8))
