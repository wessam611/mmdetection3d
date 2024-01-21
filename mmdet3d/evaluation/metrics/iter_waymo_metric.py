# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from glob import glob
from os import path as osp
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config, load
from mmengine.logging import MMLogger, print_log
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos.metrics_pb2 import Objects

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, bbox3d2result,
                                points_cam2img, xywhr2xyxyr)
from .waymo_metric import WaymoMetric

import time


@METRICS.register_module()
class IterWaymoMetric(WaymoMetric):
    """Waymo evaluation metric for iterable style dataset."""

    def __init__(self, *args, **kwargs) -> None:
        pklfile_prefix = f'submiss/{time.strftime("%Y%m%d-%H%M%S")}'
        super().__init__(prefix='Waymo metric', pklfile_prefix=pklfile_prefix, *args, **kwargs)
        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.gt_tmp_dir = tempfile.TemporaryDirectory()
        self.waymo_bin_file = f'{self.gt_tmp_dir.name}/gt.bin'

        self.gt_col_tmp_dir = tempfile.TemporaryDirectory()
        self.pred_col_tmp_dir = tempfile.TemporaryDirectory()
        self.file_idx = 0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        gt_data = data_batch['data_samples']
        for i, data_sample in enumerate(data_samples):
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d

            gt_3d = gt_data[i].gt_instances_3d
            gt_2d = gt_data[i].gt_instances
            result['gt_instances_3d'] = gt_3d.to('cpu')
            result['gt_instances'] = gt_2d.to('cpu')

            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            result['context'] = data_sample['context']
            result['timestamp_micros'] = data_sample['timestamp_micros']
            # self.results.append(result)
            self.results.append(result)

        if len(self.results) >= 100:
            self.format_results_batch(self.results, self.file_idx)
            self.file_idx += 1
            self.results = []

    def format_results_batch(self, results: List[dict], file_idx: int) -> None:
        """saves a large batch of results and gt to temp bin files.

        Args:
            results (List[dict]): Testing results of the dataset.
            file_idx (int): file name of the tmp file
        """

        waymo_results_save_file = f'{self.pred_col_tmp_dir.name}/{file_idx}.bin'
        waymo_gt_save_file = f'{self.gt_col_tmp_dir.name}/{file_idx}.bin'

        final_results = results
        for res in final_results:
            res['pred_instances_3d']['bboxes_3d'].limit_yaw(
                offset=0.5, period=np.pi * 2)
            res['gt_instances_3d']['bboxes_3d'].limit_yaw(
                offset=0.5, period=np.pi * 2)

        with open(f'{waymo_results_save_file}', 'wb') as f:
            objects = metrics_pb2.Objects()
            for res in final_results:
                self.parse_objects(res['pred_instances_3d'], res['context'],
                                   res['timestamp_micros'], objects)
            f.write(objects.SerializeToString())
        with open(waymo_gt_save_file, 'wb') as f:
            objects = metrics_pb2.Objects()
            for res in final_results:
                self.parse_objects(res['gt_instances_3d'], res['context'],
                                   res['timestamp_micros'], objects)
            f.write(objects.SerializeToString())

    def format_results(
            self,
            results: List[dict],
            pklfile_prefix: Optional[str] = None,
            *args,
            **kwargs) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Loads intermediate files and writes all predictions and GTs to final
        destination files.

        Args:
            results (List[dict]): redundant in this function
            pklfile_prefix (Optional[str], optional): final path of results bin file. Defaults to None.
        Returns:
            Tuple[dict, Union[tempfile.TemporaryDirectory, None]]
        """
        if len(self.results) > 0:
            self.format_results_batch(self.results, self.file_idx)
            self.file_idx += 1
            self.results = []

        gt_pathnames = sorted(glob(join(self.gt_col_tmp_dir.name, '*.bin')))
        gt_combined = self.combine(gt_pathnames)
        self.gt_col_tmp_dir.cleanup()
        self.gt_col_tmp_dir = tempfile.TemporaryDirectory()

        pred_pathnames = sorted(
            glob(join(self.pred_col_tmp_dir.name, '*.bin')))
        pred_combined = self.combine(pred_pathnames)
        self.pred_col_tmp_dir.cleanup()
        self.pred_col_tmp_dir = tempfile.TemporaryDirectory()

        self.file_idx = 0

        with open(self.waymo_bin_file, 'wb') as f:
            f.write(gt_combined.SerializeToString())

        with open(f'{pklfile_prefix}.bin', 'wb') as f:
            f.write(pred_combined.SerializeToString())

        return results, None

    def parse_objects(self, instances: Dict, context_name: str,
                      frame_timestamp_micros: int,
                      objects_ret: metrics_pb2.Objects) -> None:
        """Parse one prediction and corresponding GT with several instances and
        convert them to `Object` proto.

        Args:
            instances (dict): Predictions
                - labels_3d (torch.tensor): Class labels of predictions.
                - bboxes_3d (torch.tensor): LiDARInstance3DBoxes
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.

        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """
        lidar_boxes = instances['bboxes_3d'].tensor
        labels = instances['labels_3d']

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.

            Args:
                instance_idx (int): Index of the instance to be converted.

            Returns:
                :obj:`Object`: Predicted instance in waymo dataset
                    Object proto.
            """
            global countt
            cls = labels[instance_idx].item()
            box = lidar_boxes[instance_idx]
            x = box[0].item()
            y = box[1].item()
            z = box[2].item()
            length = box[3].item()
            width = box[4].item()
            height = box[5].item()
            rotation_y = box[6].item()
            if 'scores_3d' in instances:
                score = instances['scores_3d'][instance_idx].item()

            # different conventions
            heading = rotation_y
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.width = width
            box.length = length
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[self.class_names[cls]]
            if 'scores_3d' in instances:
                o.score = score
            if 'num_lidar_points_in_box' in instances:
                o.object.num_lidar_points_in_box = instances[
                    'num_lidar_points_in_box'][instance_idx].item()
            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros
            return o

        for instance_idx in range(len(instances['labels_3d'])):
            o = parse_one_object(instance_idx)
            objects_ret.objects.append(o)

    def combine(self, pathnames: List[str]) -> metrics_pb2.Objects:
        """Combine predictions in waymo format for each sample together.

        Args:
            pathnames (str): Paths to save predictions.

        Returns:
            :obj:`Objects`: Combined predictions in Objects proto.
        """
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined
