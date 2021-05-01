from skimage.morphology import remove_small_objects
import numpy as np
from skimage.measure import *
import cv2

def remove_small_areas(img, min_area):
    img = img.astype(np.bool)
    img = remove_small_objects(img, min_area, connectivity=1)
    img = img.astype(np.uint8)
    return img

def keep_large_area(img, top_n_large):
    post_img = np.zeros(img.shape)
    img = img.astype(np.bool)
    connect_regions = label(img, connectivity=1, background=0)
    props = regionprops(connect_regions)
    regions_area = []
    if len(props) > top_n_large:
        for n in range(len(props)):
            regions_area.append(props[n].area)
        index = np.argsort(np.array(regions_area))
        index = np.flip(index)
        for i in range(top_n_large):
            index_one = index[i]
            filled_value = props[index_one].label
            post_img[connect_regions == filled_value] = 1
    else:
        post_img = img
    post_img = post_img.astype(np.uint8)
    return post_img

def fit_Ellipse(mask):
    Ellipse_mask = np.zeros(mask.shape)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0 and len(contours[0]) > 5:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(Ellipse_mask, ellipse, 1, -1)
    else:
        Ellipse_mask = mask
    return Ellipse_mask

def adjust_lr(optimizer, lr_max, epoch, all_epochs):
    # if epoch < 0.1 * all_epochs:
    #     lr = (0.99 * lr_max * epoch) / (0.1 * all_epochs) + 0.01 * lr_max
    if epoch < 0.1 * all_epochs:
        lr = lr_max
    elif epoch < 0.6 * all_epochs:
        lr = lr_max
    elif epoch < 0.9 * all_epochs:
        lr = lr_max * 0.1
    else:
        lr = lr_max * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def padding_2D(image, size):
    img = np.copy(image)
    img_shape = img.shape
    left_pad = (size[0] - img_shape[0]) // 2
    right_pad = size[0] - img_shape[0] - left_pad
    top_pad = (size[1] - img_shape[1]) // 2
    down_pad = size[1] - img_shape[1] - top_pad
    paded_img = np.pad(img, ((left_pad, right_pad), (top_pad, down_pad)), mode='constant')
    return paded_img, (left_pad, right_pad, top_pad, down_pad)