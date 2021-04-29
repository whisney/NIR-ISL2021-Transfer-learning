import torch
import torch.nn as nn
from loss_function.DICE import DiceLoss, DiceLoss_3class
from torch.nn import functional as F
from torchvision import models

class recall_Loss(nn.Module):
    def __init__(self):
        super(recall_Loss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 0.2
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        recall = (intersection.sum(1) + smooth) / (target_flat.sum(1) + smooth)
        recall = recall.sum() / N
        return 1 - recall

class dice_recall_loss(nn.Module):
    def __init__(self):
        super(dice_recall_loss, self).__init__()
        self.r_loss = recall_Loss()
        self.d_loss = DiceLoss()

    def forward(self, input, target):
        loss1 = self.r_loss(input, target)
        loss2 = self.d_loss(input, target)
        return 0.75 * loss1 + 0.25 * loss2

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

def Geodesic_Distance_map_Weighted_loss(pred, foreground_distance_map, background_distance_map, labels, alpha, gamma):
    BCE = F.binary_cross_entropy(pred, labels, reduction='none')
    foreground_loss = BCE * labels * foreground_distance_map ** gamma
    background_loss = BCE * (1 - labels) * background_distance_map ** gamma
    loss_total = BCE * (1 - alpha) + alpha * (foreground_loss + background_loss)
    return torch.mean(loss_total)

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: sigmoid results,  shape=(b,1,x,y,z)
           gt: ground truth, shape=(b,1,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,1,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,1,x,y,z)
    output: boundary_loss; sclar
    """
    delta_s = (seg_soft[:, 0, ...] - gt[:, 0, ...].float()) ** 2
    s_dtm = seg_dtm[:, 0, ...] ** 2
    g_dtm = gt_dtm[:, 0, ...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

class VGG_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.modules())[1:-2])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        self.vgg = self.vgg.to(input.device)
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return torch.mean((input_vgg - target_vgg) ** 2)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

if __name__ == '__main__':
    # a = torch.sigmoid(torch.rand((3, 1, 512, 512)))
    # b = torch.sigmoid(torch.rand((3, 1, 512, 512)))
    # l = MixedLoss(alpha=10, gamma=2)
    # loss = l(a, b)
    # print(loss)
    net = models.vgg16(pretrained=False)
    net = nn.Sequential(*list(net.features.modules())[1:-2])

