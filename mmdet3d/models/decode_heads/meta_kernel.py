# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class MetaKernel(nn.Module):

    def __init__(self, kernel_size, in_C, out_Cs, coord_C=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_C = in_C
        self.out_Cs = out_Cs
        self.coord_C = coord_C
        mlp_convs = []
        for i, C in enumerate(out_Cs):
            if i == 0:
                in_channels = in_C
            else:
                in_channels = out_Cs[i - 1]
            conv = nn.Conv2d(in_channels, C, 1, 1, 0, 1, bias=False)
            mlp_convs.append(conv)
        self.mlp_conv = nn.ModuleList(mlp_convs)
        self.relu = nn.ReLU()

    def sampler_im2col(self, data):
        data = nn.Unfold(self.kernel_size, dilation=1, stride=1, padding=1)
        return data

    def relative_coord(self, sample_coord, center_coord):
        B, _, H, W = sample_coord.shape
        sample_reshape = torch.reshape(
            sample_coord, [B, self.coord_C, self.kernel_size**2, H, W])
        center_coord_expand = torch.unsqueeze(center_coord, 2)
        rel_coord = sample_reshape - center_coord_expand
        rel_coord[center_coord == -1] = 0
        return rel_coord

    def mlp(self, data):
        """_summary_

        Args:
            data (torch.Tensor): relative_coord_data

        Returns:
            _type_: _description_
        """
        B, _, H, W = data.shape
        x = torch.reshape(data, [B, -1, 3 * 3 * H, W])
        for i, layer in enumerate(self.mlp_convs):
            x = layer(x)
            if i == len(self.mlp_convs) - 1:
                x = self.relu(x)
        out = torch.reshape(x, [B, self.out_Cs[-1], -1, self.H, self.W])
        return out

    def forward(self, features, coord):
        """_summary_

        Args:
            features (torch.Tensor): B, in_channels, H, W
            coord (torch.Tensor): B, 3||6, H, W
        """
        B, CF, H, W = features.shape
        coord_sample_data = self.sampler_im2col(coord)
        rel_coord = self.relative_coord(coord_sample_data, coord)
        weights = self.mlp(rel_coord)

        data_sample = self.sampler_im2col(features)
        data_sample_reshape = torch.reshape(
            data_sample, [B, CF, self.kernel_size * self.kernel_size, H, W])
        out = data_sample_reshape * weights
        out_reshape = torch.reshape(out, [B, -1, H, W])
        return out_reshape
