import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ._get_encoder import get_encoder
from functools import partial

class Conv_Norm_Activation(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=False, dilation=1,
                 Norm=nn.BatchNorm2d, Activation=nn.ReLU):
        super(Conv_Norm_Activation, self).__init__()
        self.CNA = nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                      dilation=dilation),
            Norm(feat_out),
            Activation(inplace=True))

    def forward(self, x):
        x = self.CNA(x)
        return x

class NDDR_Layer(nn.Module):
    def __init__(self, channels, Norm=nn.BatchNorm2d, Activation=nn.ReLU):
        super(NDDR_Layer, self).__init__()
        self.CNA1 = Conv_Norm_Activation(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False,
                                         dilation=1, Norm=Norm, Activation=Activation)
        self.CNA2 = Conv_Norm_Activation(2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=False,
                                         dilation=1, Norm=Norm, Activation=Activation)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x1 = self.CNA1(x)
        x2 = self.CNA2(x)
        return x1, x2

class unet_nddr(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2_1, out_channels_2_2, encoder='resnet34',
                 decoder_channels=(256, 128, 64, 32, 16), norm='batch', activation='relu'):
        super(unet_nddr, self).__init__()
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d

        if activation == 'leakyrelu':
            activation_layer = partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            activation_layer = nn.ReLU

        self.encoder = get_encoder(encoder, in_channels=in_channels, depth=5, weights="imagenet")
        encoder_channels = self.encoder.out_channels
        encoder_channels = encoder_channels[::-1]

        self.decoder_level5_conv1_task1 = Conv_Norm_Activation(encoder_channels[0] + encoder_channels[1],
                                decoder_channels[0], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level5_conv1_task2 = Conv_Norm_Activation(encoder_channels[0] + encoder_channels[1],
                                decoder_channels[0], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level5_NDDR = NDDR_Layer(decoder_channels[0], Norm=norm_layer, Activation=activation_layer)
        self.decoder_level5_conv2_task1 = Conv_Norm_Activation(decoder_channels[0], decoder_channels[0],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level5_conv2_task2 = Conv_Norm_Activation(decoder_channels[0], decoder_channels[0],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)

        self.decoder_level4_conv1_task1 = Conv_Norm_Activation(encoder_channels[1] + encoder_channels[2],
                                decoder_channels[1], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level4_conv1_task2 = Conv_Norm_Activation(encoder_channels[1] + encoder_channels[2],
                                decoder_channels[1], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level4_NDDR = NDDR_Layer(decoder_channels[1], Norm=norm_layer, Activation=activation_layer)
        self.decoder_level4_conv2_task1 = Conv_Norm_Activation(decoder_channels[1], decoder_channels[1],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level4_conv2_task2 = Conv_Norm_Activation(decoder_channels[1], decoder_channels[1],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)

        self.decoder_level3_conv1_task1 = Conv_Norm_Activation(encoder_channels[2] + encoder_channels[3],
                                decoder_channels[2], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level3_conv1_task2 = Conv_Norm_Activation(encoder_channels[2] + encoder_channels[3],
                                decoder_channels[2], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level3_NDDR = NDDR_Layer(decoder_channels[2], Norm=norm_layer, Activation=activation_layer)
        self.decoder_level3_conv2_task1 = Conv_Norm_Activation(decoder_channels[2], decoder_channels[2],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level3_conv2_task2 = Conv_Norm_Activation(decoder_channels[2], decoder_channels[2],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)

        self.decoder_level2_conv1_task1 = Conv_Norm_Activation(encoder_channels[3] + encoder_channels[4],
                                decoder_channels[3], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level2_conv1_task2 = Conv_Norm_Activation(encoder_channels[3] + encoder_channels[4],
                                decoder_channels[3], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level2_NDDR = NDDR_Layer(decoder_channels[3], Norm=norm_layer, Activation=activation_layer)
        self.decoder_level2_conv2_task1 = Conv_Norm_Activation(decoder_channels[3], decoder_channels[3],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level2_conv2_task2 = Conv_Norm_Activation(decoder_channels[3], decoder_channels[3],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)

        self.decoder_level1_conv1_task1 = Conv_Norm_Activation(decoder_channels[3],
                                decoder_channels[4], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level1_conv1_task2 = Conv_Norm_Activation(decoder_channels[3],
                                decoder_channels[4], 3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level1_NDDR = NDDR_Layer(decoder_channels[4], Norm=norm_layer, Activation=activation_layer)
        self.decoder_level1_conv2_task1 = Conv_Norm_Activation(decoder_channels[4], decoder_channels[4],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)
        self.decoder_level1_conv2_task2 = Conv_Norm_Activation(decoder_channels[4], decoder_channels[4],
                                3, 1, 1, Norm=norm_layer, Activation=activation_layer)

        self.out_1 = nn.Conv2d(decoder_channels[4], out_channels_1, kernel_size=3, stride=1, padding=1)
        self.out_2_1 = nn.Conv2d(decoder_channels[4], out_channels_2_1, kernel_size=3, stride=1, padding=1)
        self.out_2_2 = nn.Conv2d(decoder_channels[4], out_channels_2_2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        features = features[1:]
        features = features[::-1]

        x = F.interpolate(features[0], scale_factor=2, mode='bilinear')
        x = torch.cat((x, features[1]), dim=1)
        x_task1 = self.decoder_level5_conv1_task1(x)
        x_task2 = self.decoder_level5_conv1_task2(x)
        x_task1, x_task2 = self.decoder_level5_NDDR(x_task1, x_task2)
        x_task1 = self.decoder_level5_conv2_task1(x_task1)
        x_task2 = self.decoder_level5_conv2_task2(x_task2)

        x_task1 = F.interpolate(x_task1, scale_factor=2, mode='bilinear')
        x_task1 = torch.cat((x_task1, features[2]), dim=1)
        x_task1 = self.decoder_level4_conv1_task1(x_task1)
        x_task2 = F.interpolate(x_task2, scale_factor=2, mode='bilinear')
        x_task2 = torch.cat((x_task2, features[2]), dim=1)
        x_task2 = self.decoder_level4_conv1_task1(x_task2)
        x_task1, x_task2 = self.decoder_level4_NDDR(x_task1, x_task2)
        x_task1 = self.decoder_level4_conv2_task1(x_task1)
        x_task2 = self.decoder_level4_conv2_task2(x_task2)

        x_task1 = F.interpolate(x_task1, scale_factor=2, mode='bilinear')
        x_task1 = torch.cat((x_task1, features[3]), dim=1)
        x_task1 = self.decoder_level3_conv1_task1(x_task1)
        x_task2 = F.interpolate(x_task2, scale_factor=2, mode='bilinear')
        x_task2 = torch.cat((x_task2, features[3]), dim=1)
        x_task2 = self.decoder_level3_conv1_task1(x_task2)
        x_task1, x_task2 = self.decoder_level3_NDDR(x_task1, x_task2)
        x_task1 = self.decoder_level3_conv2_task1(x_task1)
        x_task2 = self.decoder_level3_conv2_task2(x_task2)

        x_task1 = F.interpolate(x_task1, scale_factor=2, mode='bilinear')
        x_task1 = torch.cat((x_task1, features[4]), dim=1)
        x_task1 = self.decoder_level2_conv1_task1(x_task1)
        x_task2 = F.interpolate(x_task2, scale_factor=2, mode='bilinear')
        x_task2 = torch.cat((x_task2, features[4]), dim=1)
        x_task2 = self.decoder_level2_conv1_task1(x_task2)
        x_task1, x_task2 = self.decoder_level2_NDDR(x_task1, x_task2)
        x_task1 = self.decoder_level2_conv2_task1(x_task1)
        x_task2 = self.decoder_level2_conv2_task2(x_task2)

        x_task1 = F.interpolate(x_task1, scale_factor=2, mode='bilinear')
        x_task1 = self.decoder_level1_conv1_task1(x_task1)
        x_task2 = F.interpolate(x_task2, scale_factor=2, mode='bilinear')
        x_task2 = self.decoder_level1_conv1_task1(x_task2)
        x_task1, x_task2 = self.decoder_level1_NDDR(x_task1, x_task2)
        x_task1 = self.decoder_level1_conv2_task1(x_task1)
        x_task2 = self.decoder_level1_conv2_task2(x_task2)

        x_task1 = self.out_1(x_task1)
        x_task2_1 = self.out_2_1(x_task2)
        x_task2_2 = self.out_2_2(x_task2)
        return x_task1, x_task2_1, x_task2_2

if __name__ == '__main__':
    net = unet_nddr(in_channels=1, out_channels_1=1, out_channels_2=2)
    x = torch.rand((1, 1, 64, 64))
    out1, out2 = net(x)
    print(out1.size(), out2.size())