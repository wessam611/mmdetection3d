# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch import nn as nn

from mmdet3d.registry import MODELS


class DropBlock(nn.Module):

    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """Compute gamma, eq (1) in the paper
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size**2)
        valid = (x.shape[-1]**2) / ((x.shape[-1] - self.block_size + 1)**2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


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
    """U-net style CNN used for processing the range-view image.

    Args:
        in_channels (int):
        out_channels (int):
        dropout_p (float, optional): probability of
            a feature to be dropped at each layer. Defaults to 0.2.
            also the probability of a pixel to be dropped.
        dropout_block_size (float, optional):
            size of the block to be dropped. Defaults to 11.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_p: float = 0.2,
                 dropout_block_size: float = 11):
        super().__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dropoutF = nn.Dropout2d(p=dropout_p)
        self.dropoutP = DropBlock(dropout_block_size, 1 - dropout_p)

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
        xm_e1 = self.dropoutF(xm_e1)
        xm_e1 = self.dropoutP(xm_e1)
        x_e2 = self.en_c2(xm_e1)
        xm_e2, inds2 = self.max_pool(x_e2)
        xm_e2 = self.dropoutF(xm_e2)
        x_e3 = self.en_c3(xm_e2)
        xm_e3, inds3 = self.max_pool(x_e3)
        xm_e3 = self.dropoutF(xm_e3)

        x_l = self.mid(xm_e3)

        unpool_s = x_e3.shape[-2:]
        x_d3 = self.max_unpool(x_l, inds3, output_size=unpool_s)
        x_d3 = torch.concat((x_d3, x_e3), dim=-3)
        x_d3 = self.de_c3(x_d3)
        x_d3 = self.dropoutF(x_d3)

        unpool_s = x_e2.shape[-2:]
        x_d2 = self.max_unpool(x_d3, inds2, output_size=unpool_s)
        x_d2 = torch.concat((x_d2, x_e2), dim=-3)
        x_d2 = self.de_c2(x_d2)
        x_d2 = self.dropoutF(x_d2)

        unpool_s = x_e1.shape[-2:]
        x_d1 = self.max_unpool(x_d2, inds1, output_size=unpool_s)
        x_d1 = torch.concat((x_d1, x_e1), dim=-3)
        x_d1 = self.de_c1(x_d1)
        x_d1 = self.dropoutF(x_d1)
        x_d1 = self.dropoutP(x_d1)

        return x_d1
