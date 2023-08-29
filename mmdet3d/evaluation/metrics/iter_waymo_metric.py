# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Tuple, Union, Sequence

import mmengine
import numpy as np
import torch


from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos.metrics_pb2 import Objects

from mmengine import Config, load
from mmengine.logging import MMLogger, print_log

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, bbox3d2result,
                                points_cam2img, xywhr2xyxyr)
from .waymo_metric import WaymoMetric


@METRICS.register_module()
class IterWaymoMetric(WaymoMetric):
    """Waymo evaluation metric.

    Args:
        ann_file (str): The path of the annotation file in kitti format.
        waymo_bin_file (str): The path of the annotation file in waymo format.
        data_root (str): Path of dataset root. Used for storing waymo
            evaluation programs.
        split (str): The split of the evaluation set. Defaults to 'training'.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'mAP'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [-85, -85, -5, 85, 85, 5].
        convert_kitti_format (bool): Whether to convert the results to kitti
            format. Now, in order to be compatible with camera-based methods,
            defaults to True.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        load_type (str): Type of loading mode during training.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default cam
              and need to convert to the FOV-based data type to support image-
              based detector.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM_FRONT'.
        use_pred_sample_idx (bool): In formating results, use the sample index
            from the prediction or from the load annotations. By default,
            KITTI: True, Waymo: False, Waymo has a conversion process, which
            needs to use the sample idx from load annotation.
            Defaults to False.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        idx2metainfo (str, optional): The file path of the metainfo in waymo.
            It stores the mapping from sample_idx to metainfo. The metainfo
            must contain the keys: 'idx2contextname' and 'idx2timestamp'.
            Defaults to None.
    """
    num_cams = 5

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            result['context'] = data_sample['context']
            result['timestamp_micros'] = data_sample['timestamp_micros']
            self.results.append(result)
    
    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to bin file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None. not used in this override
            classes (List[str], optional): A list of class name.
                Defaults to None. not used in this override

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified. 
            tmp_dir is alway None in this override
        """

        waymo_results_save_file = f'{pklfile_prefix}.bin'
        
        final_results = results
        for res in final_results:
            # Actually, `sample_idx` here is the filename without suffix.
            # It's for identitying the sample in formating.]
            res['pred_instances_3d']['bboxes_3d'].limit_yaw(
                offset=0.5, period=np.pi * 2)

        with open(f'{waymo_results_save_file}',
                    'wb') as f:
            objects = metrics_pb2.Objects()
            for res in final_results:
                self.parse_objects(res['pred_instances_3d'], res['context'], res['timestamp_micros'], objects)
            f.write(objects.SerializeToString())

        return final_results, None

    def parse_objects(self, instances, context_name,
                      frame_timestamp_micros, objects_ret):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.

        Args:
            instances (dict): Predictions
                - labels_3d (np.ndarray): Class labels of predictions.
                - bboxes_3d (np.ndarray): LiDARInstance3DBoxes
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.

        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.

            Args:
                instance_idx (int): Index of the instance to be converted.

            Returns:
                :obj:`Object`: Predicted instance in waymo dataset
                    Object proto.
            """
            cls = instances['labels_3d'][instance_idx]
            box = instances['bboxes_3d'][instance_idx]
            x = box.center[:, 0]
            y = box.center[:, 1]
            z = box.center[:, 2]
            length = box.dims[:, 0]
            width = box.dims[:, 1]
            height = box.dims[:, 2]
            rotation_y = box.yaw
            if 'scores_3d' in instances:
                score = instances['scores_3d'][instance_idx]

            z += height / 2

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
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[self.class_names[cls]]
            if 'scores_3d' in instances:
                o.score = score
            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros
            return o

        for instance_idx in range(len(instances['labels_3d'])):
            o = parse_one_object(instance_idx)
            objects_ret.objects.append(o)
