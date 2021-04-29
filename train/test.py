from moviepy.editor import *
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorboardX import SummaryWriter

# iris_edge_mask = io.imread(r'I:\public_data\NIR-ISL2021\CASIA-Iris-Asia\CASIA-distance\train\iris_edge_mask/S4000L02_00003.png', as_gray=True).astype(np.float)
# iris_edge = io.imread(r'I:\public_data\NIR-ISL2021\CASIA-Iris-Asia\CASIA-distance\train\iris_edge/S4000L02_00003.png', as_gray=True).astype(np.float)
#
# print(np.unique(iris_edge_mask))
# print(np.unique(iris_edge))
# print(np.unique(iris_edge_mask - iris_edge))

Writer = SummaryWriter(log_dir=r'E:\competition\NIR-ISL2021\postprocess\trained_models\Africa\Local_inner_outer\baseline_UNet_ResNet34\bs16_epoch500_fold1\log')
