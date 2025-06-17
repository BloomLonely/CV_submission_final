import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class submission_20221396(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(submission_20221396, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        #filters = [4, 8, 16, 32, 64]
        filters = [2, 4, 8, 16, 16]

        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = DepthwiseSeparableConv(filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ELU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = DepthwiseSeparableConv(filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ELU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = DepthwiseSeparableConv(filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ELU(inplace=True)

        self.h4_Cat_hd4_conv = DepthwiseSeparableConv(filters[3], self.CatChannels, kernel_size=3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ELU(inplace=True)

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = DepthwiseSeparableConv(filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ELU(inplace=True)

        self.conv4d_1 = DepthwiseSeparableConv(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ELU(inplace=True)

        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = DepthwiseSeparableConv(filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ELU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = DepthwiseSeparableConv(filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ELU(inplace=True)

        self.h3_Cat_hd3_conv = DepthwiseSeparableConv(filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ELU(inplace=True)

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ELU(inplace=True)

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = DepthwiseSeparableConv(filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ELU(inplace=True)

        self.conv3d_1 = DepthwiseSeparableConv(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ELU(inplace=True)

        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = DepthwiseSeparableConv(filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ELU(inplace=True)

        self.h2_Cat_hd2_conv = DepthwiseSeparableConv(filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ELU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ELU(inplace=True)

        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ELU(inplace=True)

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = DepthwiseSeparableConv(filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ELU(inplace=True)

        self.conv2d_1 = DepthwiseSeparableConv(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ELU(inplace=True)

        self.h1_Cat_hd1_conv = DepthwiseSeparableConv(filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ELU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ELU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd3_UT_hd1_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ELU(inplace=True)

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd4_UT_hd1_conv = DepthwiseSeparableConv(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ELU(inplace=True)

        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = DepthwiseSeparableConv(filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ELU(inplace=True)

        self.conv1d_1 = DepthwiseSeparableConv(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ELU(inplace=True)

        self.outconv1 = DepthwiseSeparableConv(self.UpChannels, num_classes, kernel_size=3, padding=1)
        self.se4d = SEBlock(self.UpChannels, reduction=4)
        self.se3d = SEBlock(self.UpChannels, reduction=4)
        self.se2d = SEBlock(self.UpChannels, reduction=4)
        self.se1d = SEBlock(self.UpChannels, reduction=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        h1 = self.conv1(inputs)

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4_fused = torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)
        hd4_fused = self.se4d(hd4_fused)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(hd4_fused)))

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3_fused = torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)
        hd3_fused = self.se3d(hd3_fused)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(hd3_fused)))
        
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2_fused = torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)
        hd2_fused = self.se2d(hd2_fused)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(hd2_fused)))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1_fused = torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)
        hd1_fused = self.se1d(hd1_fused)
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(hd1_fused)))

        d1 = self.outconv1(hd1)
        return d1

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(DepthwiseSeparableConv(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ELU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(DepthwiseSeparableConv(in_size, out_size, ks, s, p),
                                     nn.ELU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    def _init_fn(m):
        if isinstance(m, DepthwiseSeparableConv):
            return
        if init_type == 'normal':
            weights_init_normal(m)
        elif init_type == 'xavier':
            weights_init_xavier(m)
        elif init_type == 'kaiming':
            weights_init_kaiming(m)
        elif init_type == 'orthogonal':
            weights_init_orthogonal(m)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    net.apply(_init_fn)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        for m in [self.depthwise, self.pointwise]:
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
