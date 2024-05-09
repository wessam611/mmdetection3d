# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
from typing import List, Tuple, Optional
from torch import nn
from torch import Tensor

from mmengine.runner import amp
from mmengine.config import ConfigDict
from mmdet3d.utils.typing_utils import ConfigType
from mmengine.structures import InstanceData
from mmdet.models.utils import select_single_mlvl
from mmdet.models.utils import multi_apply, images_to_levels
from mmdet.utils.memory import cast_tensor_type
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import InstanceList,OptInstanceList
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import limit_period, xywhr2xyxyr

from .anchor3d_head import Anchor3DHead
from .train_mixins import get_direction_target

@MODELS.register_module()
class Anchor3DHeadIoU(Anchor3DHead):
    """
    Same as Anchor3DHead just adding an IoU prediction head.
    Had to copy a lot of the parenting classes as prediction heads
    were quite hard coded.
    """
    def __init__(self,
                 loss_iou: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True), *args, **kwargs):
        super(Anchor3DHeadIoU, self).__init__(*args, **kwargs)
        self.loss_iou = MODELS.build(loss_iou)

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        super(Anchor3DHeadIoU, self)._init_layers()
        self.conv_iou = nn.Conv2d(self.feat_channels, self.num_anchors, 1)
    
    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cls_score, bbox_pred, dir_cls_pred = super(
            Anchor3DHeadIoU, self).forward_single(x)
        iou_score = self.conv_iou(x)
        return cls_score, bbox_pred, dir_cls_pred, iou_score

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor, iou_score: Tensor, 
                             labels: Tensor, label_weights: Tensor, 
                             bbox_targets: Tensor, bbox_weights: Tensor, 
                             dir_targets: Tensor, dir_weights: Tensor, 
                             gt_ious_list, num_total_samples: int):
        label_weights[labels < 0] = 0
        loss_cls, loss_bbox, loss_dir = super(
            Anchor3DHeadIoU, self)._loss_by_feat_single(cls_score, bbox_pred,
                             dir_cls_pred, labels,
                             label_weights, bbox_targets,
                             bbox_weights, dir_targets,
                             dir_weights, num_total_samples)
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        gt_ious_list = gt_ious_list.reshape(-1)
        pos_inds = (
            ((labels >= 0))&(gt_ious_list >= 0.3)).nonzero(
                        as_tuple=False).reshape(-1)
        neg_inds = (
            ((labels >= 0))&(gt_ious_list < 0.3)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)
        neg_inds = neg_inds[torch.randint(0, len(neg_inds), (num_pos//10, ))]

        iou_score = iou_score.permute(0, 2, 3, 1).reshape(-1)
        pos_iou_score = iou_score[pos_inds]
        pos_iou_gt = gt_ious_list[pos_inds]
        neg_iou_score = iou_score[neg_inds]
        neg_iou_gt = gt_ious_list[neg_inds]
        loss_iou_score = torch.cat([pos_iou_score, neg_iou_score])
        loss_iou_gt = torch.cat([pos_iou_gt, neg_iou_gt])
        if num_pos > 0:
            loss_iou = self.loss_iou(
                loss_iou_score,
                loss_iou_gt,
                avg_factor=num_total_samples)
        else:
            loss_iou = iou_score.sigmoid().mean()
        return loss_cls, loss_bbox, loss_dir, loss_iou
        # loss calculate
    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            iou_scores: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, gt_ious_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # num_total_samples = None
        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir, losses_iou = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                cast_tensor_type(iou_scores, dst_type=torch.float32),
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                gt_ious_list,
                num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir, loss_iou=losses_iou)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        iou_scores: List[Tensor],
                        batch_input_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_input_metas (list[dict], Optional): Batch inputs meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        assert len(cls_scores) == len(iou_scores)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]

        result_list = []

        for input_id in range(len(batch_input_metas)):

            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)
            iou_scores_list = select_single_mlvl(iou_scores, input_id)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                iou_scores_list=iou_scores_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                **kwargs)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                iou_scores_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                input_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                **kwargs) -> InstanceData:
        """Transform a single points sample's features extracted from the head
        into bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single point cloud sample, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single point cloud sample, each item
                has shape (num_priors * C, H, W).
            dir_cls_pred_list (list[Tensor]): Predictions of direction class
                from all scale levels of a single point cloud sample, each
                item has shape (num_priors * 2, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            input_meta (dict): Contain point clouds and image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_iou_scores = []
        for cls_score, bbox_pred, dir_cls_pred, iou_scores, priors in zip(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                iou_scores_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            assert cls_score.size()[-2:] == iou_scores.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            iou_scores = iou_scores.permute(1, 2, 0).reshape(-1)
            iou_scores = iou_scores.sigmoid()

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                iou_scores = iou_scores[topk_inds]

            bboxes = self.bbox_coder.decode(priors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_iou_scores.append(iou_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_iou_scores = torch.cat(mlvl_iou_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores,
                                       mlvl_iou_scores=mlvl_iou_scores)
        bboxes, scores, labels, dir_scores, iou_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        results.iou_scores = iou_scores

        return results

    # TODO: Support augmentation test
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        pass

    def anchor_target_3d(self,
                         anchor_list,
                         batch_gt_instances_3d,
                         batch_input_metas,
                         batch_gt_instances_ignore=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Ground truth
                bboxes of each image.
            batch_input_metas (list[dict]): Meta info of each image.
            batch_gt_instances_ignore (list): Ignore list of gt bboxes.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of positive anchors and
                number of negative anchors.
        """
        num_inputs = len(batch_input_metas)
        assert len(anchor_list) == num_inputs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_inputs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.view(-1, self.box_code_size).size(0)
                for anchors in anchor_list[0]
            ]
            # concat all level anchors and flags to a single tensor
            for i in range(num_inputs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None for _ in range(num_inputs)]

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, all_iou_targets, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             batch_gt_instances_3d,
             batch_gt_instances_ignore,
             batch_input_metas,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        iou_targets_list = images_to_levels(all_iou_targets, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                iou_targets_list, num_total_pos, num_total_neg)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_instance_3d,
                                gt_instance_ignore,
                                input_meta,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_instance_3d (:obj:`InstanceData`): Gt bboxes.
            gt_instance_ignore (:obj:`InstanceData`): Ignored gt bboxes.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        if isinstance(self.bbox_assigner,
                      list) and (not isinstance(anchors, list)):
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
            rot_angles = anchors.size(-2)
            assert len(self.bbox_assigner) == anchors.size(-3)
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights, total_gt_ious,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(
                    -1, self.box_code_size)
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_instance_3d.labels_3d == i)
                    gt_per_cls_instance = InstanceData()
                    gt_per_cls_instance.labels_3d = gt_instance_3d.labels_3d[
                        gt_per_cls]
                    gt_per_cls_instance.bboxes_3d = gt_instance_3d.bboxes_3d[
                        gt_per_cls, :]
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_per_cls_instance,
                        gt_instance_ignore, input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_instance_3d,
                        gt_instance_ignore, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, gt_ious, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels.reshape(feat_size, 1, rot_angles))
                total_label_weights.append(
                    label_weights.reshape(feat_size, 1, rot_angles))
                total_bbox_targets.append(
                    bbox_targets.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_dir_targets.append(
                    dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(
                    dir_weights.reshape(feat_size, 1, rot_angles))
                total_gt_ious.append(
                    gt_ious.reshape(feat_size, 1, rot_angles))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
            total_label_weights = torch.cat(
                total_label_weights, dim=-2).reshape(-1)
            total_bbox_targets = torch.cat(
                total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
            total_bbox_weights = torch.cat(
                total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
            total_dir_targets = torch.cat(
                total_dir_targets, dim=-2).reshape(-1)
            total_dir_weights = torch.cat(
                total_dir_weights, dim=-2).reshape(-1)
            total_gt_ious = torch.cat(
                total_gt_ious, dim=-2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_gt_ious, total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(
                anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_gt_ious, total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_instance_3d.labels_3d == i)
                    gt_per_cls_instance = InstanceData()
                    gt_per_cls_instance.labels_3d = gt_instance_3d.labels_3d[
                        gt_per_cls]
                    gt_per_cls_instance.bboxes_3d = gt_instance_3d.bboxes_3d[
                        gt_per_cls, :]
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_per_cls_instance,
                        gt_instance_ignore, input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_instance_3d,
                        gt_instance_ignore, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, gt_ious, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(
                    bbox_targets.reshape(-1, anchors[i].size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(-1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_gt_ious.append(gt_ious)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=0)
            total_label_weights = torch.cat(total_label_weights, dim=0)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=0)
            total_bbox_weights = torch.cat(total_bbox_weights, dim=0)
            total_dir_targets = torch.cat(total_dir_targets, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_gt_ious = torch.cat(total_gt_ious, dim=0)
            total_neg_inds = torch.cat(total_neg_inds, dim=0)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_gt_ious, total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner,
                                                      anchors, gt_instance_3d,
                                                      gt_instance_ignore,
                                                      input_meta, num_classes,
                                                      sampling)

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_instance_3d,
                                      gt_instance_ignore,
                                      input_meta,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_instance_3d (:obj:`InstanceData`): Gt bboxes.
            gt_instance_ignore (torch.Tensor): Ignored gt bboxes.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long)
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        gt_ious = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_instance_3d.bboxes_3d) > 0:
            if not isinstance(gt_instance_3d.bboxes_3d, torch.Tensor):
                gt_instance_3d.bboxes_3d = gt_instance_3d.bboxes_3d.tensor.to(
                    anchors.device)
            pred_instance_3d = InstanceData(priors=anchors)
            assign_result = bbox_assigner.assign(pred_instance_3d,
                                                 gt_instance_3d,
                                                 gt_instance_ignore)
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       pred_instance_3d,
                                                       gt_instance_3d)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0,
                as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) == 0,
                as_tuple=False).squeeze(-1).unique()

        if gt_instance_3d.labels_3d is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                self.dir_offset,
                self.dir_limit_offset,
                one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_instance_3d.labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_instance_3d.labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
            gt_ious[pos_inds] = assign_result.max_overlaps[pos_inds]

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, gt_ious, pos_inds, neg_inds)
