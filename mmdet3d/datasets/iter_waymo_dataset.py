# Copyright (c) OpenMMLab. All rights reserved.
import os
import math
import random
import itertools
from typing import Callable, List, Union, Optional

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import IterableWrapper
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class IterWaymoDataset(IterableDataset, Det3DDataset):
    """
    """
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
                 shuffle: bool = True,
                 shuffle_size: int = 10,
                 buffer_size: int = 10,
                 num_parallel_reads: int = 3,
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
        self.mode = mode
        self._fully_initialized = True
        Det3DDataset.__init__(self,
                              pipeline=pipeline,
                              modality=modality,
                              default_cam_key=default_cam_key,
                              box_type_3d=box_type_3d,
                              filter_empty_gt=filter_empty_gt,
                              show_ins_var=show_ins_var,
                              **kwargs)

        mode_folders = {'train': 'training',
                        'val': 'validation',
                        'test': 'testing'}
        cloud_bucket = tfds.core.Path(
            f'gs://waymo_open_dataset_{cloud_bucket_version}/'
        )
        self.bucket_files = tf.io.gfile.glob(
            os.path.join(
                cloud_bucket, f'individual_files/{mode_folders[self.mode]}/segment*'
            )
        )[:1]

        self.buffer_size = buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size
        self.filter_empty_gt = filter_empty_gt
        self.repeat = repeat
        self.val_divs = val_divs
        
        self._len_used_files = len(self.bucket_files)
        

    def _build_torch_dataset_iter(self, bucket_files):

        ds = tf.data.TFRecordDataset(bucket_files,
                                     buffer_size=self.buffer_size,
                                     num_parallel_reads=self.num_parallel_reads,
                                     compression_type="")
        
        if self.repeat:
            ds = ds.repeat()
        torch_ds = IterableWrapper(ds)
        torch_ds = torch_ds.map(self.pipeline)
        return torch_ds

    def __iter__(self):
        """
        """
        used_files = self.bucket_files
        if self.mode != 'train' and self.val_divs > 1:
            partition_index = random.randint(0, 4)
            files_no = math.ceil(len(used_files)/self.val_divs)
            used_files = used_files[files_no*partition_index:min(files_no*(partition_index+1), len(used_files))]
        self._len_used_files = len(used_files)
        try:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
            previous_files = math.round(worker_id*len(used_files)/worker_total_num)
            files_no = math.round(len(used_files-previous_files)/(worker_total_num-worker_id))
            files = used_files[files_no*worker_id:min(files_no*(worker_id+1), len(used_files))]
        except Exception as e:
            files = self.bucket_files
        self.torch_ds = self._build_torch_dataset_iter(files)
        return iter(self.torch_ds)
    
    def __len__(self) -> int:
        if self.repeat:
            return int(1e9)
        return self._len_used_files*200

    @staticmethod
    def _filter_empty_gt_fn(elem):
        """
        TODO: 
        """
        return True