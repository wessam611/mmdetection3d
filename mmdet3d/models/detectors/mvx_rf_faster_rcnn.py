# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from mmdet3d.registry import MODELS
from .mvx_faster_rcnn import MVXFasterRCNN

class DropBlock(nn.Module):
    """
    DropBlock layer. Drops patches from an image based on
    block_size and p.
    Disregarded experiment.
    """
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

@MODELS.register_module()
class MVXRFFasterRCNN(MVXFasterRCNN):
    """Multi-modality VoxelNet using Faster R-CNN.

    Implementing range view fusion
    """

    def __init__(self, rf_net=None, dla_to_dist=None, rv_dropout_p=0.2, bev_dropout_p=0.4, **kwargs):
        """_summary_

        Args:
            rf_net (_type_): _description_
            dla_to_dist (dict): maps from dla_layer stride value to
                    distance from the center range. Each pillar is fused
                    with one dla_fpn layer based on distance to sensor.
        """
        super(MVXRFFasterRCNN, self).__init__(**kwargs)
        if dla_to_dist is None:
            self.dla_to_dist = [(0, 1e5)]
        else:
            self.dla_to_dist = dla_to_dist
        if rf_net:
            self.rf_net = MODELS.build(rf_net)
        self.rv_dropout = nn.Dropout2d(rv_dropout_p)
        self.bev_dropout = nn.Dropout2d(bev_dropout_p)

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple
                inputs in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        range_image = batch_inputs_dict.get('range_image')
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        bev_vox_indices = self.get_bev_to_range_indices(
            voxel_dict['coors'], batch_size)
        pts_feats = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas)
        range_feats = self.rf_net(range_image,
                                  batch_inputs_dict.get('range_xyz_coo', None))
        if not isinstance(range_feats, list):
            range_feats = [range_feats]
        voxel_centers = voxel_dict['voxel_centers']
        pts_feats = self.stack_pts_rv_feats(
            pts_feats, range_feats, batch_inputs_dict['vox_range_indices'],
            bev_vox_indices, voxel_centers)
        return (img_feats, pts_feats)

    def stack_pts_rv_feats(self, pts_feats, range_feats, vox_range_indices,
                           bev_vox_indices, voxel_centers):
        """_summary_

        Args:
            pts_feats (list(Tensor)): [(B, Cp, W, H)]
            range_feats (list(Tensor)): [(B, Cr, W, H)], each entry is related
                to a range based on horizontal distance to sensor
            vox_range_indices (Tensor): (V_batches, max_points_per_Voxel, 2)
                last 2 dims represent range_index and batch_index(=-1 for padding)
            bev_vox_indices (list(Tensor)): [(V_batch)]

        Returns:
            list(Tensor): [(B, Cp+Cr, H, W)]
        """
        range_feat_dim = range_feats[0].shape[-3]
        canvas_feats_dims = list(pts_feats[0].shape)
        canvas_feats_dims[-3] = range_feat_dim
        s = canvas_feats_dims + [vox_range_indices.shape[-2]]
        range_canvas = torch.zeros((*s[:-3], s[-3] * s[-2], s[-1]),
                                   device=range_feats[0].device)
        l2_dist = torch.norm(voxel_centers[:, :2], 2, dim=1)

        for i, v in enumerate(range_feats):
            range_feats[i] = torch.flatten(v, -2, -1)

        for i in range(range_canvas.shape[0]):
            vox_range_mask = torch.any(vox_range_indices[..., 1] == i, dim=-1)
            range_inds = vox_range_indices[vox_range_mask][..., 0]
            range_inds[range_inds >=
                       range_feats[0].shape[-1]] -= range_feats[0].shape[-1]
            l2_dist_cur = l2_dist[vox_range_mask]
            for range_feat, range_to_center in zip(range_feats,
                                                   self.dla_to_dist):
                min_r, max_r = range_to_center
                range_inds_filtered = range_inds[(
                    l2_dist_cur >= min_r).logical_and(l2_dist_cur < max_r)]
                bev_vox_indices_filtered = bev_vox_indices[i][(
                    l2_dist_cur >= min_r).logical_and(l2_dist_cur < max_r)]
                range_canvas[i, :, bev_vox_indices_filtered, :] = range_feat[
                    i][:, range_inds_filtered]
            range_canvas[
                i, :, bev_vox_indices[i], :][:,
                                             vox_range_indices[vox_range_mask][
                                                 ..., 1] == -1] = -torch.inf

        range_canvas, _ = torch.max(range_canvas, dim=-1)
        range_canvas = range_canvas.reshape(canvas_feats_dims)
        range_canvas[range_canvas == -torch.inf] = 0
        range_canvas = self.rv_dropout(range_canvas)
        pts_feats[0] = self.bev_dropout(pts_feats[0])
        pts_feats[0] = torch.concat((pts_feats[0], range_canvas), dim=-3)

        return pts_feats

    def get_bev_to_range_indices(self, coors: Tensor, batch_size: int):
        indices_batch = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:,
                                 2] * self.pts_middle_encoder.nx + this_coors[:,
                                                                              3]
            indices = indices.type(torch.int)
            indices_batch.append(indices)
        return indices_batch
