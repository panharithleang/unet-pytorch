from os import name
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.transforms import CenterCrop, Scale

# unet consisted of 4 differents nn operation

# 1. 3*3 convolution -> relu -> 3*3 convolution -> relu


class DoubleConvolution(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConvolution, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. Down sampling with 2*2 softmax with stride 2
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.down(x)

# 3. Up sampling


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = CenterCrop((x1.size()[2], x1.size()[3]))(x2)
        x = torch.cat([x2, x1], dim=1)
        return x

# 4. Output 1*1 convoution


class OutputConv(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, num_classes, 1),
            nn.Softmax()
        )
        self.conv = nn.Conv2d(in_channel, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.conv0 = DoubleConvolution(3, 64)
        self.down = DownSample()
        self.conv1 = DoubleConvolution(64, 128)
        self.down = DownSample()
        self.conv2 = DoubleConvolution(128, 256)
        self.down = DownSample()
        self.conv3 = DoubleConvolution(256, 512)
        self.down = DownSample()
        self.conv4 = DoubleConvolution(512, 1024)
        self.up0 = UpSample(1024, 512)
        self.conv5 = DoubleConvolution(1024, 512)
        self.up1 = UpSample(512, 256)
        self.conv6 = DoubleConvolution(512, 256)
        self.up2 = UpSample(256, 128)
        self.conv7 = DoubleConvolution(256, 128)
        self.up3 = UpSample(128, 64)
        self.conv8 = DoubleConvolution(128, 64)
        self.out = OutputConv(64, num_classes)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.down(x1)
        x2 = self.conv1(x2)
        x3 = self.down(x2)
        x3 = self.conv2(x3)
        x4 = self.down(x3)
        x4 = self.conv3(x4)
        x5 = self.down(x4)
        x5 = self.conv4(x5)

        x = self.up0(x5, x4)
        x = self.conv5(x)
        x = self.up1(x, x3)
        x = self.conv6(x)
        x = self.up2(x, x2)
        x = self.conv7(x)
        x = self.up3(x, x1)
        x = self.conv8(x)
        x = self.out(x)
        return x


# if __name__ == '__main__':
#     uNet = UNet(2)
#     print(uNet)
#     print(sum(p.numel() for p in uNet.parameters()))
