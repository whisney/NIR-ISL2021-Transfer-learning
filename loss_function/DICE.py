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