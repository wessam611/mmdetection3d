# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn as nn

from mmdet3d.registry import MODELS


class RVConv2D(nn.Module):
    """
    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: (int, Tuple) = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding='valid')

    def forward(self, x):
        if isinstance(self.kernel_size, int):
            px = py = self.kernel_size // 2
        elif isinstance(self.kernel_size, tuple):
            kx, ky = self.kernel_size
            px = kx // 2
            py = ky // 2
        else:
            # not possible to be raised anyway
            raise TypeError()
        x = F.pad(x, [px, px, 0, 0], 'circular')
        x = F.pad(x, [0, 0, py, py], 'constant', 0)
        x = self.conv(x)
        return x


@MODELS.register_module()
class RangeViewCnnHead(nn.Module):
    """_summary_"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.en_c1 = nn.Sequential(
            RVConv2D(in_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            RVConv2D(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.en_c2 = nn.Sequential(
            RVConv2D(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RVConv2D(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.en_c3 = nn.Sequential(
            RVConv2D(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            RVConv2D(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.mid = nn.Sequential(
            RVConv2D(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            RVConv2D(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.de_c3 = nn.Sequential(
            RVConv2D(256 * 2, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            RVConv2D(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.de_c2 = nn.Sequential(
            RVConv2D(128 * 2, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            RVConv2D(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.de_c1 = nn.Sequential(
            RVConv2D(64 * 2, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            RVConv2D(64, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_e1 = self.en_c1(x)
        xm_e1, inds1 = self.max_pool(x_e1)
        x_e2 = self.en_c2(xm_e1)
        xm_e2, inds2 = self.max_pool(x_e2)
        x_e3 = self.en_c3(xm_e2)
        xm_e3, inds3 = self.max_pool(x_e3)

        x_l = self.mid(xm_e3)

        unpool_s = x_e3.shape[-2:]
        x_d3 = self.max_unpool(x_l, inds3, output_size=unpool_s)
        x_d3 = torch.concat((x_d3, x_e3), dim=-3)
        x_d3 = self.de_c3(x_d3)

        unpool_s = x_e2.shape[-2:]
        x_d2 = self.max_unpool(x_d3, inds2, output_size=unpool_s)
        x_d2 = torch.concat((x_d2, x_e2), dim=-3)
        x_d2 = self.de_c2(x_d2)

        unpool_s = x_e1.shape[-2:]
        x_d1 = self.max_unpool(x_d2, inds1, output_size=unpool_s)
        x_d1 = torch.concat((x_d1, x_e1), dim=-3)
        x_d1 = self.de_c1(x_d1)

        return x_d1
