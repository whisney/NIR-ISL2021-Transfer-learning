import torch
import torch.nn as nn
from loss_function.DICE import DiceLoss

class dice_BCE_loss(nn.Module):
    def __init__(self, bce_weight, dice_weight):
        super(dice_BCE_loss, self).__init__()
        self.b_loss = nn.BCELoss()
        self.d_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, input, target):
        loss1 = self.b_loss(input, target)
        loss2 = self.d_loss(input, target)
        return self.bce_weight*loss1 + self.dice_weight*loss2

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


