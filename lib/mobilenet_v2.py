import torch
from torch import nn
from torch.nn import functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1, norm='bn', relu=True):
        super(ConvBlock, self).__init__()

        self.relu = relu

        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        if self.relu:
            return F.relu6(self.norm(self.conv(x)))
        else:
            return self.norm(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, norm=False):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)

        if self.use_residual:
            m = x + m

        if hasattr(self, 'norm'):
            m = self.norm(m)

        return m


class MobilenetV2(nn.Module):

    def __init__(self, num_classes=None):
        super(MobilenetV2, self).__init__()

        self.conv1 = ConvBlock(3, 32, 3, s=2, p=1)
        self.block2 = Bottleneck(32, 16, 1, 1)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2),
            Bottleneck(24, 24, 6, 1),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 32, 6, 1),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)

        if num_classes is not None:
            self.classifier = nn.Linear(1280, num_classes)

    def init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=.0001)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_logit=False):
        if not self.training and not return_logit:
            x = torch.cat((x, x[:, :, :, torch.arange(x.shape[3]-1, -1, -1)]))

        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = F.dropout(x, training=self.training)

        if self.training or return_logit:
            x = self.classifier(x)
        else:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
            x /= x.norm(dim=1, keepdim=True)

        return x


class MobilenetV2BatchNorm(nn.Module):

    def __init__(self, num_classes=None):
        super(MobilenetV2BatchNorm, self).__init__()

        self.conv1 = ConvBlock(3, 32, 3, s=2, p=1)
        self.block2 = Bottleneck(32, 16, 1, 1)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2),
            Bottleneck(24, 24, 6, 1),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 32, 6, 1),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)

        self.batch_norm = nn.BatchNorm1d(1280)

        if num_classes is not None:
            self.classifier = nn.Linear(1280, num_classes)
            nn.init.normal_(self.classifier.weight, std=.001)
            nn.init.zeros_(self.classifier.bias)

    def init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=.001)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_logit=False):
        if not self.training and not return_logit:
            x = torch.cat((x, x[:, :, :, torch.arange(x.shape[3]-1, -1, -1)]))

        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = self.batch_norm(x)

        if self.training or return_logit:
            x = self.classifier(x)
        else:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
            x /= x.norm(dim=1, keepdim=True)

        return x


class MobilenetV2IFN(nn.Module):

    def __init__(self, num_classes=None):
        super(MobilenetV2IFN, self).__init__()

        self.conv1 = ConvBlock(3, 32, 3, s=2, p=1, norm='in')
        self.block2 = Bottleneck(32, 16, 1, 1, norm=True)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2, norm=True),
            Bottleneck(24, 24, 6, 1, norm=True),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2, norm=True),
            Bottleneck(32, 32, 6, 1, norm=True),
            Bottleneck(32, 32, 6, 1, norm=True),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2, norm=True),
            Bottleneck(64, 64, 6, 1, norm=True),
            Bottleneck(64, 64, 6, 1, norm=True),
            Bottleneck(64, 64, 6, 1, norm=True),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1, norm=True),
            Bottleneck(96, 96, 6, 1, norm=True),
            Bottleneck(96, 96, 6, 1, norm=True),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.bnneck = nn.BatchNorm1d(1280)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        if num_classes is not None:
            self.classifier = nn.Linear(1280, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

    def init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, return_logit=False):
        if not self.training and not return_logit:
            x = torch.cat((x, x[:, :, :, torch.arange(x.shape[3]-1, -1, -1)]))

        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.bnneck(x)

        if self.training or return_logit:
            x = self.classifier(x)
        else:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
            x /= x.norm(dim=1, keepdim=True)

        return x
