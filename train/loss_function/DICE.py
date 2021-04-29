import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        N = target.size(0)
        C = target.size(1)
        smooth = 0.2
        if self.weight:
            weight = torch.Tensor(self.weight).float().to(input.device)
        else:
            weight = np.ones(C)
            weight = torch.from_numpy(weight * (1 / C)).float().to(input.device)
        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
        weight = weight.unsqueeze(0).repeat(loss.size(0), 1)
        loss *= weight
        loss = 1 - loss.sum() / N
        return loss

class DiceLoss_3class(nn.Module):
    def __init__(self):
        super(DiceLoss_3class, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        C = target.size(1)
        smooth = 0.2

        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
        loss = loss.sum(0) / N
        weith = torch.tensor([0, 0.2, 0.8]).cuda()
        loss = loss * weith
        loss = 1 - loss.sum()

        return loss

class DiceLoss_5class(nn.Module):
    def __init__(self):
        super(DiceLoss_5class, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        C = target.size(1)
        smooth = 0.2

        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
        loss = loss.sum(0) / N
        weith = torch.tensor([0, 0.25, 0.25, 0.25, 0.25]).cuda()
        loss = loss * weith
        loss = 1 - loss.sum()

        return loss

def dice1(inputs, GT):
    N = GT.size(0)
    smooth = 0.2
    inputs[inputs > 0.5] = 1
    inputs[inputs <= 0.5] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0
    input_flat = inputs.view(N, -1)
    target_flat = GT.view(N, -1)
    intersection = input_flat * target_flat
    dice_sco = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    dice_sco = dice_sco.sum() / N
    return dice_sco

def dice3(inputs, GT):
    smooth = 0.2
    N = GT.size(0)
    C = GT.size(1)
    GT[GT < 0.5] = 0
    GT[GT >= 0.5] = 1
    input_flat = inputs.view(N, C, -1)
    target_flat = GT.view(N, C, -1)
    intersection = input_flat * target_flat
    dice_sco = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
    dice_sco = dice_sco.sum(0) / N
    #weith = torch.tensor([0, 0.2, 0.8]).cuda()
    #dice_sco = dice_sco * weith
    return dice_sco

def dice5(inputs, GT):
    #print(inputs.shape, GT.shape)
    smooth = 0.2
    N = GT.size(0)
    C = GT.size(1)
    maxindex = inputs.argmax(1)               #[bs, H, W]
    maxindex = maxindex.unsqueeze(1).cpu()          #[bs, 1, H, W]
    inputs = torch.zeros(inputs.shape).scatter_(1,maxindex,1).cuda()       #[bs, 5, H, W]   onehot
    GT[GT < 0.5] = 0
    GT[GT >= 0.5] = 1
    #print(inputs.shape, GT.shape)
    input_flat = inputs.view(N, C, -1)
    target_flat = GT.view(N, C, -1)
    #print(input_flat.shape, target_flat.shape)
    intersection = input_flat * target_flat
    dice_sco = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
    dice_sco = dice_sco.sum(0) / N
    weith = torch.tensor([0, 0.25, 0.25, 0.25, 0.25]).cuda()
    dice_sco = dice_sco * weith
    return dice_sco.sum()

