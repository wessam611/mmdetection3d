import os
import shutil
import pickle
import torch
import numpy as np
from datetime import datetime

from mmdet3d.registry import FUNCTIONS
from mmdet3d.structures import LiDARInstance3DBoxes

from .pseudo_label_utils import NAverageMeter


@FUNCTIONS.register_module()
class PseudoLabelUpdater:
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    def __init__(self,
                 psuedo_labels_dir,
                 neg_th,
                 pos_th,
                 ensemble_function,
                 pkl_prefix=None):

        self.ps_labels_dir = psuedo_labels_dir
        self.neg_th = neg_th
        self.pos_th = pos_th
        self.new_ps_labels = {}
        self.ps_labels = {}
        self.pkl_prefix = pkl_prefix
        if isinstance(ensemble_function, (dict, list)):
            self.ensemble_function = FUNCTIONS.build(ensemble_function)
        else:
            raise ValueError('ensemble function is only p as a '+
                             'dictionary describing a function under '+
                                'the FUNCTIONS registry')
        self.default_infos = {
                'gt_boxes': np.zeros((0, 9), dtype=np.float32),
                'cls_scores': None,
                'iou_scores': None,
                'memory_counter': np.zeros(0)
            }

    def frame_key(self, context, timestamp_micros):
        return f'{context}_{timestamp_micros}'

    def finish_update(self):
        self.ps_labels.clear()
        self.ps_labels.update(self.new_ps_labels)
        self.new_ps_labels.clear()
        if self.pkl_prefix is None:
            return
        i = 0
        while(True):
            if  not os.path.exists(
                os.path.join(
                    self.pkl_prefix,
                    f'pseudo_labels{i:02}.pkl'
                )):
                break 
            i += 1
        with open(os.path.join(
                    self.pkl_prefix,
                    f'pseudo_labels{i:02}.pkl'), 'wb') as f_pkl:
            pickle.dump(self.ps_labels, f_pkl)


    def get_ps_labels(self, ctx, stamp):
        """returns gt_boxes & labels & ignore

        Args:
            path (_type_): _description_
        """
        results = dict()
        
        gt_infos = self.ps_labels.get(self.frame_key(ctx, stamp), None)
        if gt_infos is not None:
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt_infos['gt_boxes'][:, :7])
            gt_labels = gt_infos['gt_boxes'][:, 7].astype(np.int64)
            gt_labels[gt_labels > 0] -= 1
            results['gt_labels_3d'] = gt_labels
            results['num_lidar_points_in_box'] = np.ones_like(gt_labels)*50
        else:
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                np.zeros((0, 7)))
            results['gt_labels_3d'] = np.zeros((0, 1))
            results['num_lidar_points_in_box'] = np.zeros((0, 1))
        return results

    def __call__(self, pred_dicts, data_batch):
        """
        adapted from ST3D 
        https://github.com/CVMI-Lab/ST3D/blob/master/docs/GETTING_STARTED.md
        Args:
            outputs (_type_): _description_
            data_batch (_type_): _description_
        """
        for out in pred_dicts:
            pred3d = out.pred_instances_3d
            bbox3d = pred3d['bboxes_3d']
            labels_3d = pred3d['labels_3d']
            scores_3d = pred3d['scores_3d']
            context = out.context
            timestamp_micros = out.timestamp_micros

        pos_ps_nmeter = NAverageMeter(len(self.CLASS_NAMES))
        ign_ps_nmeter = NAverageMeter(len(self.CLASS_NAMES))

        batch_size = len(pred_dicts)
        for b_idx in range(batch_size):
            if 'bboxes_3d' in pred_dicts[b_idx].pred_instances_3d:
                # Exist predicted boxes passing self-training score threshold
                pred3d = pred_dicts[b_idx].pred_instances_3d
                bboxes_3d = pred3d['bboxes_3d'].detach().cpu().numpy()
                labels_3d = pred3d['labels_3d'].detach().cpu().numpy() + 1
                scores_3d = pred3d['scores_3d'].detach().cpu().numpy()
                iou_scores = pred3d['iou_scores'].detach().cpu().numpy()

                # remove boxes under negative threshold
                if self.neg_th:
                    labels_remove_scores = np.array(self.neg_th)[labels_3d - 1]
                    remain_mask = scores_3d >= labels_remove_scores
                    labels_3d = labels_3d[remain_mask]
                    scores_3d = scores_3d[remain_mask]
                    bboxes_3d = bboxes_3d[remain_mask]
                    iou_scores = iou_scores[remain_mask]
                labels_ignore_scores = np.array(self.pos_th)[labels_3d - 1]
                ignore_mask = scores_3d < labels_ignore_scores
                labels_3d[ignore_mask] = -labels_3d[ignore_mask] # class = -1

                gt_box = np.concatenate((bboxes_3d,
                                        labels_3d.reshape(-1, 1),
                                        scores_3d.reshape(-1, 1)), axis=1)

            else:
                # no predicted boxes passes self-training score threshold
                gt_box = np.zeros((0, 9), dtype=np.float32)
            gt_infos = {
                'gt_boxes': gt_box,
                'cls_scores': scores_3d,
                'iou_scores': iou_scores,
                'memory_counter': np.zeros(gt_box.shape[0])
            }

            # record pseudo label to pseudo label dict
            # if need_update:
            frame_key = self.frame_key(
                pred_dicts[b_idx].context, pred_dicts[b_idx].timestamp_micros)
            gt_infos = self.ensemble_function(
                self.ps_labels.get(frame_key, self.default_infos), gt_infos,)
            # counter the number of ignore boxes for each class
            for i in range(ign_ps_nmeter.n):
                num_total_boxes = (np.abs(gt_infos['gt_boxes'][:, 7]) == (i+1)).sum()
                ign_ps_nmeter.update((gt_infos['gt_boxes'][:, 7] == -(i+1)).sum(), index=i)
                pos_ps_nmeter.update(num_total_boxes - ign_ps_nmeter.meters[i].val, index=i)

            self.new_ps_labels[frame_key] = gt_infos

        return pos_ps_nmeter, ign_ps_nmeter

    @property
    def ps_path(self):
        return self._ps_path
