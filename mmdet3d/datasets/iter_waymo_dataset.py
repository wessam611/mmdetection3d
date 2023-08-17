# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Union, Optional

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset
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
                 domain_adaptation: bool = False,
                 batch_size: int = 2,
                 shuffle: bool = True,
                 shuffle_size: int = 2,
                 buffer_size: int = 2,
                 num_parallel_reads: int = 2,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 show_ins_var: bool = False,
                 **kwargs):
        assert cloud_bucket_version in [
            'v_1_3_2', 'v_1_4_0', 'v_1_4_1', 'v_1_4_2'
            ], f'GCS bucket version {cloud_bucket_version} is not supported'
        assert not domain_adaptation, 'domain adaptation is not yet supported'
        assert mode in ['train', 'val', 'test']
        self._fully_initialized = True
        Det3DDataset.__init__(self,
                              pipeline=pipeline,
                              modality=modality,
                              default_cam_key=default_cam_key,
                              box_type_3d=box_type_3d,
                              filter_empty_gt=filter_empty_gt,
                              show_ins_var=show_ins_var,
                              **kwargs)

        self._batch_size = batch_size

        mode_folders = {'train': 'training',
                        'val': 'validation',
                        'test': 'testing'}
        cloud_bucket = tfds.core.Path(
            f'gs://waymo_open_dataset_{cloud_bucket_version}/'
        )
        self._train_files = tf.io.gfile.glob(
            os.path.join(
                cloud_bucket, f'individual_files/{mode_folders[mode]}/segment*'
            )
        )
        self._buffer_size = buffer_size
        self._num_parallel_reads = num_parallel_reads
        self._shuffle = shuffle
        self._shuffle_size = shuffle_size
        self._filter_empty_gt = filter_empty_gt
        self._tfds = self._build_tf_dataset()
        self._tfds_iter = iter(self._tfds)

    def _build_tf_dataset(self):

        ds = tf.data.TFRecordDataset(self._train_files,
                                     buffer_size=self._buffer_size,
                                     num_parallel_reads=self._num_parallel_reads,
                                     compression_type="")
        if self._filter_empty_gt: # Should be called after frame parsing
            ds = ds.filter(IterWaymoDataset._filter_empty_gt_fn)
        if self._shuffle:
            ds = ds.shuffle(self._shuffle_size)
        ds = ds.map(lambda x: tf.py_function(self.pipeline, [x], tf.float32))
        ds = ds.prefetch(buffer_size=self._buffer_size)
        ds = ds.batch(self._batch_size)
        return ds

    def __iter__(self):
        """
        """
        return self._tfds_iter
    
    def __len__(self) -> int:
        return int(1e7)

    @staticmethod
    def _filter_empty_gt_fn(elem):
        """
        TODO: 
        """
        return True