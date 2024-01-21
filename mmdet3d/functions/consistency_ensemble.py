"""
From ST3D repository https://github.com/CVMI-Lab/ST3D/blob/master/docs/GETTING_STARTED.md
"""
import torch
import numpy as np

from mmcv.ops import boxes_iou3d
from mmdet3d.registry import FUNCTIONS

from .pseudo_label_utils import mask_dict, concatenate_array_inside_dict, check_numpy_to_torch


@FUNCTIONS.register_module()
class ConsistencyEnsemble:
    def __init__(self, 
                 iou_th=0.1,
                 memory_voting=True, 
                 mv_ignore_th=2, 
                 mv_remove_th=3):
        self.iou_th = iou_th
        self.memory_voting = memory_voting
        self.mv_ignore_th = mv_ignore_th
        self.mv_remove_th = mv_remove_th

    def __call__(self, gt_infos_a, gt_infos_b):
        classes_a = np.unique(np.abs(gt_infos_a['gt_boxes'][:, -2]))
        classes_b = np.unique(np.abs(gt_infos_b['gt_boxes'][:, -2]))

        n_classes = max(classes_a.shape[0], classes_b.shape[0])
        if n_classes == 0:
            return gt_infos_a
        
        # single category case
        if n_classes == 1:
            return self._call_elem(gt_infos_a, gt_infos_b)

        # for multi class case
        merged_infos = {}
        for i in np.union1d(classes_a, classes_b):
            mask_a = np.abs(gt_infos_a['gt_boxes'][:, -2]) == i
            gt_infos_a_i = mask_dict(gt_infos_a, mask_a)

            mask_b = np.abs(gt_infos_b['gt_boxes'][:, -2]) == i
            gt_infos_b_i = mask_dict(gt_infos_b, mask_b)

            gt_infos = self._call_elem(gt_infos_a_i, gt_infos_b_i)
            merged_infos = concatenate_array_inside_dict(merged_infos, gt_infos)
            
        return merged_infos

    def _call_elem(self, gt_infos_a, gt_infos_b):
        """
        Args:
            gt_infos_a:
                gt_boxes: (N, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for previous pseudo boxes
                cls_scores: (N)
                iou_scores: (N)
                memory_counter: (N)

            gt_infos_b:
                gt_boxes: (M, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for current pseudo boxes
                cls_scores: (M)
                iou_scores: (M)
                memory_counter: (M)

            memory_ensemble_cfg:

        Returns:
            gt_infos:
                gt_boxes: (K, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for merged pseudo boxes
                cls_scores: (K)
                iou_scores: (K)
                memory_counter: (K)
        """
        gt_box_a, _ = check_numpy_to_torch(gt_infos_a['gt_boxes'])
        gt_box_b, _ = check_numpy_to_torch(gt_infos_b['gt_boxes'])
        gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()

        new_gt_box = gt_infos_a['gt_boxes']
        new_cls_scores = gt_infos_a['cls_scores']
        new_iou_scores = gt_infos_a['iou_scores']
        new_memory_counter = gt_infos_a['memory_counter']
        
        new_cls_scores = gt_infos_a['cls_scores']
        # if gt_box_b or gt_box_a don't have any predictions
        if gt_box_b.shape[0] == 0:
            gt_infos_a['memory_counter'] += 1
            return gt_infos_a
        elif gt_box_a.shape[0] == 0:
            return gt_infos_b

        # get ious
        iou_matrix = boxes_iou3d(gt_box_a[:, :7], gt_box_b[:, :7]).cpu()

        ious, match_idx = torch.max(iou_matrix, dim=1)
        ious, match_idx = ious.numpy(), match_idx.numpy()
        gt_box_a, gt_box_b = gt_box_a.cpu().numpy(), gt_box_b.cpu().numpy()

        match_pairs_idx = np.concatenate((
            np.array(list(range(gt_box_a.shape[0]))).reshape(-1, 1),
            match_idx.reshape(-1, 1)), axis=1)

        #########################################################
        # filter matched pair boxes by IoU
        # if matching succeeded, use boxes with higher confidence
        #########################################################

        iou_mask = (ious >= self.iou_th) # TODO config

        matching_selected = match_pairs_idx[iou_mask]
        gt_box_selected_a = gt_box_a[matching_selected[:, 0]]
        gt_box_selected_b = gt_box_b[matching_selected[:, 1]]

        # assign boxes with higher confidence
        score_mask = gt_box_selected_a[:, 8] < gt_box_selected_b[:, 8]

        new_gt_box[matching_selected[score_mask, 0], :] = gt_box_selected_b[score_mask, :]

        if gt_infos_a['cls_scores'] is not None:
            new_cls_scores[matching_selected[score_mask, 0]] = gt_infos_b['cls_scores'][
                matching_selected[score_mask, 1]]
        if gt_infos_a['iou_scores'] is not None:
            new_iou_scores[matching_selected[score_mask, 0]] = gt_infos_b['iou_scores'][
                matching_selected[score_mask, 1]]

        # for matching pairs, clear the ignore counter
        new_memory_counter[matching_selected[:, 0]] = 0

        #######################################################
        # If previous bboxes disappeared: ious <= 0.1
        #######################################################
        disappear_idx = (ious < self.iou_th).nonzero()[0] # TODO config

        if self.memory_voting: # TODO config
            new_memory_counter[disappear_idx] += 1
            # ignore gt_boxes that ignore_count == IGNORE_THRESH
            ignore_mask = new_memory_counter >= self.mv_ignore_th # TODO config
            new_gt_box[ignore_mask, 7] = -1

            # remove gt_boxes that ignore_count >= RM_THRESH
            remain_mask = new_memory_counter < self.mv_remove_th # TODO config
            new_gt_box = new_gt_box[remain_mask]
            new_memory_counter = new_memory_counter[remain_mask]
            if gt_infos_a['cls_scores'] is not None:
                new_cls_scores = new_cls_scores[remain_mask]
            if gt_infos_a['iou_scores'] is not None:
                new_iou_scores = new_iou_scores[remain_mask]

        # Add new appear boxes
        ious_b2a, match_idx_b2a = torch.max(iou_matrix, dim=0)
        ious_b2a, match_idx_b2a = ious_b2a.numpy(), match_idx_b2a.numpy()

        newboxes_idx = (ious_b2a < self.iou_th).nonzero()[0]
        if newboxes_idx.shape[0] != 0:
            new_gt_box = np.concatenate((new_gt_box, gt_infos_b['gt_boxes'][newboxes_idx, :]), axis=0)
            if gt_infos_a['cls_scores'] is not None:
                new_cls_scores = np.concatenate((new_cls_scores, gt_infos_b['cls_scores'][newboxes_idx]), axis=0)
            if gt_infos_a['iou_scores'] is not None:
                new_iou_scores = np.concatenate((new_iou_scores, gt_infos_b['iou_scores'][newboxes_idx]), axis=0)
            new_memory_counter = np.concatenate((new_memory_counter, gt_infos_b['memory_counter'][newboxes_idx]), axis=0)

        new_gt_infos = {
            'gt_boxes': new_gt_box,
            'cls_scores': new_cls_scores if gt_infos_a['cls_scores'] is not None else None,
            'iou_scores': new_iou_scores if gt_infos_a['iou_scores'] is not None else None,
            'memory_counter': new_memory_counter
        }

        return new_gt_infos
