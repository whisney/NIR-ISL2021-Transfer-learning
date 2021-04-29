import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import argparse
import skimage.io as io
import cv2
from skimage import exposure
from utils import keep_large_area

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_root = 'data'
input_size = (416, 416)

M1_name_list = os.listdir('data/testing/CASIA-Iris-Mobile-V1.0/test/image')

model_path = [
    '/home/zyw/NIR_ISL2021/trained_models/M1/Seg/finetune_UNet_ResNet34/bs28_epoch100_fold0/best_acc.pth',
    '/home/zyw/NIR_ISL2021/trained_models/M1/Seg/finetune_UNet_ResNet34/bs28_epoch100_fold1/best_acc.pth',
    '/home/zyw/NIR_ISL2021/trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold2/best_acc.pth',
    '/home/zyw/NIR_ISL2021/trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold3/best_acc.pth',
    '/home/zyw/NIR_ISL2021/trained_models/M1/Seg/baseline_UNet_ResNet34/bs28_epoch400_fold4/best_acc.pth'
]

new_dir = 'trained_models/NIR-ISL2021030401_1/CASIA-Iris-M1'
M1_save_dir = os.path.join(new_dir, 'CASIA-Iris-M1', 'SegmentationClass')
os.makedirs(M1_save_dir, exist_ok=True)

img_name_dicts = [{'data_dir': 'data/testing/CASIA-Iris-Mobile-V1.0/test/image', 'save_dir': M1_save_dir,
                   'name_list': M1_name_list}]

net_all = []
for path in model_path:
    net = smp.Unet('resnet34', in_channels=1, classes=1, activation=None).cuda()
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
            prediction = np.zeros(img.shape)
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
                    prediction_one = net(img_tta)
                    prediction_one = torch.sigmoid(prediction_one).cpu().squeeze(0).squeeze(0).detach().numpy()
                    if t == 1:
                        prediction_one = np.flip(prediction_one, axis=1)
                    prediction += prediction_one
            prediction /= 20
            prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            _, prediction = cv2.threshold(prediction, 0.5, 1, 0)
            prediction = keep_large_area(prediction, top_n_large=1)

            io.imsave(os.path.join(save_dir, os.path.splitext(name)[0] + '.png'), (prediction * 255).astype(np.uint8))