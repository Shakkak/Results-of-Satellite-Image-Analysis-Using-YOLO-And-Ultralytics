"""CSP wrapper blocks used by the yolov7-C3C2-SPPFCSPC.yaml and
yolov7-MobileOne.yaml configs: CNeB, C3C2, MobileOne, SPPFCSPC.

Adapted from iscyy/yoloair (https://github.com/iscyy/yoloair). Depends on
the stock `Conv`, `Bottleneck`, `autopad` already present in a standard
yolov7 clone's models/common.py.
"""
import torch
import torch.nn as nn

from models.common import Conv, Bottleneck, autopad
from .convnext import ConvNextBlock
from .mobileone import MobileOneBlock


class CNeB(nn.Module):
    """CSP ConvNextBlock with 3 convolutions (by iscyy/yoloair)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(ConvNextBlock(c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3C2(nn.Module):
    """CSP Bottleneck with 3 convolutions, Mish-activated fusion branch."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv = nn.Conv2d(c1, c_, 1, 1, autopad(1, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.cv1 = Conv(2 * c_, c2, 1, act=nn.Mish())
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y = self.conv(x)
        return self.cv1(torch.cat((self.m(self.act(self.bn(y))), y), dim=1))


class MobileOne(nn.Module):
    """Stack of MobileOne re-parameterizable blocks."""

    def __init__(self, in_channels, out_channels, n, k,
                 stride=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()
        self.m = nn.Sequential(*[MobileOneBlock(in_channels, out_channels, k, stride, deploy) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class SPPFCSPC(nn.Module):
    """CSP + SPPF hybrid (https://github.com/WongKinYiu/CrossStagePartialNetworks)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
