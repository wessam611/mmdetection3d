# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.registry import MODELS


class DLARBasicBlock(nn.Module):

    def __init__(self, name, conv, inC, outC, stride, dilate, proj):
        super().__init__()
        self.name = name
        self.proj = proj
        self.relu = nn.ReLU()
        self.inC = inC
        self.outC = outC
        if conv is not None:
            self.conv1 = MODELS.build(conv)
        else:
            self.conv1 = nn.Conv2d(inC, outC, 3, 1, dilate, dilate)
            self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, 3, stride, dilate, dilate)
        self.bn2 = nn.BatchNorm2d(outC)

        if self.proj:
            self.conv3 = nn.Conv2d(
                inC, outC, 3, stride, bias=False, padding=dilate)
            self.bn3 = nn.BatchNorm2d(outC)

    def forward(self, data, coo=None):
        if isinstance(self.conv1, nn.Conv2d):
            c1 = self.relu(self.bn1(self.conv1(data)))
        else:
            c1 = self.conv1(data, coo)
        c2 = self.bn2(self.conv2(c1))
        if self.proj:
            shortcut = self.bn3(self.conv3(data))
        else:
            shortcut = data
        eltwise = c2 + shortcut
        out = self.relu(eltwise)
        return out


class ResStage(nn.Module):

    def __init__(self, stage, conv_dict, num_block, inC, outC, stride, dilate):
        super().__init__()
        s, d = stride, dilate
        self.basic_blocks = []
        f = True
        # for c, s, d in conv_blocks:
        conv_block = conv_dict.get(f'{stage}_1', None)
        self.basic_blocks.append(
            DLARBasicBlock(f'{stage}_1', conv_block, inC, outC, s, d, True))
        for i in range(2, num_block + 1):
            conv_block = conv_dict.get(f'{stage}_{i}', None)
            self.basic_blocks.append(
                DLARBasicBlock(f'{stage}_{i}', conv_block, outC, outC, 1, d,
                               False))
        self.basic_blocks = nn.Sequential(*self.basic_blocks)

    def forward(self, data, coo=None):
        for basic_block in self.basic_blocks:
            data = basic_block(data, coo=coo)
        return data


class AggStage(nn.Module):

    def __init__(self, stage, conv_dict, num_block, inC, outC, stride, dilate,
                 deconv_k, deconv_s, deconv_p):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(inC, outC, deconv_k, deconv_s,
                                          deconv_p)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(outC)
        self.stage = stage
        self.res_stage = ResStage(stage, conv_dict, num_block, outC, outC,
                                  stride, dilate)

    def forward(self, data_const, data_upsample):
        data_upsample = self.relu(self.bn1(self.deconv1(data_upsample)))
        eltwise = data_const + data_upsample
        out = self.res_stage(eltwise)
        return out


@MODELS.register_module()
class DLABackbone(nn.Module):

    def __init__(self,
                 inC,
                 conv_dict,
                 num_block,
                 num_C,
                 add_data_sc,
                 fpn_strides,
                 init_padding=(3, 3)):
        super().__init__()
        self.add_data_sc = add_data_sc
        self.fpn_strides = fpn_strides
        self.init_padding = init_padding

        self.res1 = ResStage('res1', conv_dict, num_block['res1'], inC,
                             num_C['res1'], 1, 1)
        self.res2a = ResStage('res2a', conv_dict, num_block['res2a'],
                              num_C['res1'], num_C['res2a'], (1, 2), 1)
        self.res2 = ResStage('res2', conv_dict, num_block['res2'],
                             num_C['res2a'], num_C['res2'], (1, 2), 1)
        self.res3a = ResStage('res3a', conv_dict, num_block['res3a'],
                              num_C['res2'], num_C['res3a'], (1, 2), 1)
        self.res3 = ResStage('res3', conv_dict, num_block['res3'],
                             num_C['res3a'], num_C['res3'], (1, 2), 1)

        self.agg2 = AggStage('agg2', conv_dict, num_block['agg2'],
                             num_C['res3'], num_C['agg2'], 1, 1, (3, 8),
                             (1, 4), (1, 2))
        self.agg1 = AggStage('agg1', conv_dict, num_block['agg1'],
                             num_C['res2'], num_C['agg1'], 1, 1, (3, 8),
                             (1, 4), (1, 2))
        self.agg2a = AggStage('agg2a', conv_dict, num_block['agg2a'],
                              num_C['agg2'], num_C['agg2a'], 1, 1, (3, 4),
                              (1, 2), (1, 1))
        self.agg3 = AggStage('agg3', conv_dict, num_block['agg3'],
                             num_C['agg2a'], num_C['agg3'], 1, 1, (3, 4),
                             (1, 2), (1, 1))

    def forward(self, data, coo, *_, **__):
        """_summary_

        Args:
            data (torch.Tensor): Range image:
        """
        data = F.pad(data, self.init_padding)
        coo = F.pad(coo, self.init_padding)
        r1 = self.res1(data, coo)
        r2a = self.res2a(r1, coo)
        r2 = self.res2(r2a, coo)
        r3a = self.res3a(r2, coo)
        r3 = self.res3(r3a, coo)

        a2 = self.agg2(r2, r3)
        a1 = self.agg1(r1, r2)
        a2a = self.agg2a(r2a, a2)
        a3 = self.agg3(a1, a2a)

        if self.add_data_sc:
            a3 = torch.concat([data, a3], dim=1)

        pl, pr = self.init_padding
        a3 = a3[..., pl:-pr]
        a2a = F.interpolate(a2a, scale_factor=(1, 2))[..., pl:-pr]
        a2 = F.interpolate(a2, scale_factor=(1, 4))[..., pl:-pr]

        # if

        out_dict = {1: a3, 2: a2a, 4: a2}

        if self.fpn_strides is not None:
            return [out_dict[s] for s in self.fpn_strides]
        else:
            return [
                a3,
            ]
