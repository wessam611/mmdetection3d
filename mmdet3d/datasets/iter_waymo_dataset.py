# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
import os
import random
from typing import Callable, Iterable, List, Optional, Union

import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import IterableWrapper
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class IterWaymoDataset(IterableDataset, Det3DDataset):
    """"""
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
        'palette': [
            (0, 120, 255),  # Waymo Blue
            (0, 232, 157),  # Waymo Green
            (255, 205, 85)  # Amber
        ]
    }

    def __init__(self,
                 cloud_bucket_version: str = 'v_1_4_1',
                 mode: str = 'train',
                 val_divs: int = 5,
                 domain_adaptation: bool = False,
                 num_parallel_reads: int = 80,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 show_ins_var: bool = False,
                 repeat: bool = False,
                 **kwargs):
        assert cloud_bucket_version in [
            'v_1_3_2', 'v_1_4_0', 'v_1_4_1', 'v_1_4_2'
        ], f'GCS bucket version {cloud_bucket_version} is not supported'
        assert not domain_adaptation, 'domain adaptation is not yet supported'
        assert mode in ['train', 'val', 'test']
        mode_folders = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing'
        }
        # some variables are still based on loading files from the cloud directly
        # however the code has been changed to read from disk directly with
        # preprocessed pointclouds. current path is pointing to disk
        cloud_bucket = tfds.core.Path('data/waymo/waymo_format/records/')
        self.mode = mode
        self._fully_initialized = True
        self.val_divs = val_divs
        self.repeat = repeat

        self.bucket_files = tf.io.gfile.glob(
            os.path.join(cloud_bucket, f'{mode_folders[self.mode]}/segment*'))
        self.used_files = self.bucket_files
        if self.mode != 'train' and self.val_divs > 1:
            partition_index = random.randint(0, self.val_divs - 1)
            files_no = math.ceil(len(self.used_files) / self.val_divs)
            self.used_files = self.used_files[files_no * partition_index:min(
                files_no * (partition_index + 1), len(self.used_files))]
        self._len_used_files = len(self.used_files)

        Det3DDataset.__init__(
            self,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            show_ins_var=show_ins_var,
            **kwargs)

        self.num_parallel_reads = num_parallel_reads
        self.filter_empty_gt = filter_empty_gt
        torch.multiprocessing.set_sharing_strategy('file_system')

    def _build_torch_dataset_iter(self, bucket_files) -> Iterable:
        random.shuffle(bucket_files)
        ds = tf.data.TFRecordDataset(
            bucket_files,
            num_parallel_reads=self.num_parallel_reads,
            compression_type='')
        torch_ds = IterableWrapper(ds)
        torch_ds = torch_ds.map(self.pipeline)
        return torch_ds

    def __iter__(self) -> dict:
        """"""
        try:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
            previous_files = round(worker_id * len(self.used_files) /
                                   worker_total_num)
            files_no = round((len(self.used_files) - previous_files) /
                             (worker_total_num - worker_id))
            files = self.used_files[
                previous_files:min(previous_files +
                                   files_no, len(self.used_files))]
        except Exception as e:
            files = self.used_files

        while (True):
            self.torch_ds = self._build_torch_dataset_iter(files)
            for elem in self.torch_ds:
                if len(elem['data_samples'].gt_instances_3d) or not (
                        self.filter_empty_gt and self.mode == 'train'):
                    yield elem
            if not self.repeat:
                break

    def __len__(self) -> int:
        if self.repeat:
            return int(8 * 120000)  # TODO: not hardcoded
        return self._len_used_files * 200  # approx
