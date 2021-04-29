import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torch

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

class Local_UNet(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=1, out_channels_1=1, out_channels_2=1,
                 decoder_channels=(256, 128, 64, 32, 16)):
        super(Local_UNet, self).__init__()
        self.Unet = smp.Unet(encoder_name, in_channels=in_channels, classes=1, activation=None,
                        decoder_channels=decoder_channels)
        self.out_1 = nn.Conv2d(decoder_channels[-1], out_channels_1, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Conv2d(decoder_channels[-1], out_channels_2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = self.Unet.encoder(x)
        decoder_output = self.Unet.decoder(*features)
        return self.out_1(decoder_output), self.out_2(decoder_output)

if __name__ == '__main__':
    net = Local_UNet()
    print(net)
    a = torch.rand((1, 1, 128, 128))
    b = net(a)
    print(b[0].size())