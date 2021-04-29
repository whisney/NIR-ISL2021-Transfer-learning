from skimage.morphology import remove_small_objects
import numpy as np
from skimage.measure import *
import cv2
import matplotlib.pyplot as plt
import torch
import nibabel as nib

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

def crop_mask_expand(mask, expand_Percentage):
    mask_copy = np.copy(mask)
    img_shape = mask_copy.shape
    x_, y_ = mask_copy.nonzero()
    x1 = x_.min()
    x2 = x_.max()
    y1 = y_.min()
    y2 = y_.max()
    expand_x = int(expand_Percentage * 100) if x2 - x1 <= 100 else int((x2 - x1) * expand_Percentage)
    expand_y = int(expand_Percentage * 100) if y2 - y1 <= 100 else int((y2 - y1) * expand_Percentage)
    x1_new = 0 if x1 - expand_x <= 0 else x1 - expand_x
    x2_new = img_shape[0] if x2 + expand_x >= img_shape[0] else x2 + expand_x
    y1_new = 0 if y1 - expand_y <= 0 else y1 - expand_y
    y2_new = img_shape[1] if y2 + expand_y >= img_shape[1] else y2 + expand_y
    return x1_new, x2_new, y1_new, y2_new

def roi_extend(img_shape, size, x1_new, x2_new, y1_new, y2_new):
    if x2_new - x1_new >= size[0]:
        x1_roi, x2_roi = x1_new, x2_new
    else:
        left_extend = int((size[0] - (x2_new - x1_new)) / 2)
        right_extend = size[0] - (x2_new - x1_new) - left_extend
        x1_roi = x1_new - left_extend
        x2_roi = x2_new + right_extend
        if x1_roi < 0:
            x1_roi = 0
            x2_roi = size[0]
        if x2_roi > img_shape[0]:
            x2_roi = img_shape[0]
            x1_roi = img_shape[0] - size[0]
    if y2_new - y1_new >= size[1]:
        y1_roi, y2_roi = y1_new, y2_new
    else:
        top_extend = int((size[1] - (y2_new - y1_new)) / 2)
        down_extend = size[1] - (y2_new - y1_new) - top_extend
        y1_roi = y1_new - top_extend
        y2_roi = y2_new + down_extend
        if y1_roi < 0:
            y1_roi = 0
            y2_roi = size[1]
        if y2_roi > img_shape[1]:
            y2_roi = img_shape[1]
            y1_roi = img_shape[1] - size[1]
    return x1_roi, x2_roi, y1_roi, y2_roi

def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img

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

def calc_gradient_penalty(real_data, fake_data, D_net, mask, lambda_=10, gpu_mode=True):
    # gradient penalty
    data_size = real_data.size()
    alpha = torch.rand(data_size[0], 1)
    alpha = alpha.expand(data_size[0], int(real_data.nelement() /
                         data_size[0])).contiguous().view(data_size[0], data_size[1], data_size[2], data_size[3])
    if gpu_mode:
        alpha = alpha.cuda()

    x_hat = alpha * real_data + (1 - alpha) * fake_data
    x_hat = x_hat * mask

    x_hat.requires_grad = True

    pred_hat = D_net(x_hat).view(-1)
    if gpu_mode:
        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

    return gradient_penalty

def mu_updata(syn_net, seg_net, syn_mu, seg_mu):
    with torch.no_grad():
        momentum = 0.9
        syn_mu = syn_mu.mean(dim=0, keepdim=True)
        syn_net.module.enau.mu *= momentum
        syn_net.module.enau.mu += syn_mu * (1 - momentum)

        seg_mu = seg_mu.mean(dim=0, keepdim=True)
        seg_net.module.enau.mu *= momentum
        seg_net.module.enau.mu += seg_mu * (1 - momentum)

def mu_updata_one_GPU(net, mu):
    with torch.no_grad():
        momentum = 0.9
        mu[0] = mu[0].mean(dim=0, keepdim=True)
        net.level1_dual_gating.enau.mu *= momentum
        net.level1_dual_gating.enau.mu += mu[0] * (1 - momentum)

        mu[1] = mu[1].mean(dim=0, keepdim=True)
        net.level2_dual_gating.enau.mu *= momentum
        net.level2_dual_gating.enau.mu += mu[1] * (1 - momentum)

        mu[2] = mu[2].mean(dim=0, keepdim=True)
        net.level3_dual_gating.enau.mu *= momentum
        net.level3_dual_gating.enau.mu += mu[2] * (1 - momentum)

        mu[3] = mu[3].mean(dim=0, keepdim=True)
        net.level4_dual_gating.enau.mu *= momentum
        net.level4_dual_gating.enau.mu += mu[3] * (1 - momentum)

def BN_mode(net, mode=0):
    attention_layer = ['level1_dual_gating', 'level2_dual_gating', 'level3_dual_gating', 'level4_dual_gating']
    if mode == 0:
        for name, module in net._modules.items():
            if name in attention_layer:
                module.eval()
            else:
                module.train()
    elif mode == 1:
        for name, module in net._modules.items():
            if name in attention_layer:
                module.train()
            else:
                module.eval()

def load_dict_without_DA(net, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = net.state_dict()
    pretrained_dict_new = {}
    for k, v in pretrained_dict.items():
        if 'level1_dual_gating' in k or 'level2_dual_gating' in k or 'level3_dual_gating' in k or 'level4_dual_gating' in k:
            pass
        else:
            if k in model_dict:
                pretrained_dict_new.update({k: v})
    model_dict.update(pretrained_dict_new)
    net.load_state_dict(model_dict)

def padding_2D(image, size):
    img = np.copy(image)
    img_shape = img.shape
    left_pad = (size[0] - img_shape[0]) // 2
    right_pad = size[0] - img_shape[0] - left_pad
    top_pad = (size[1] - img_shape[1]) // 2
    down_pad = size[1] - img_shape[1] - top_pad
    paded_img = np.pad(img, ((left_pad, right_pad), (top_pad, down_pad)), mode='constant')
    return paded_img, (left_pad, right_pad, top_pad, down_pad)