# Copyright (c) OpenMMLab. All rights reserved.
import glob
import math
import os
import random
from typing import Callable, Iterable, List, Optional, Union

import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset
from mmdet3d.datasets.transforms import CurriculumDataAugmentation, LoadPseudoLabels

@DATASETS.register_module()
class IterWaymoDataset(Det3DDataset):
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
                 mode: str = 'train',
                 domain_adaptation: bool = False,
                 val_divs: int = 5,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 show_ins_var: bool = False,
                 repeat: bool = False,
                 skips_n: int = 5,
                 data_path: str = None,
                 files_txt: str = None,
                 **kwargs):
        assert mode in ['train', 'val', 'test']
        mode_folders = {
            'train': 'training',
            'val': 'validation',
            'test': 'validation'
        }
        self.mode = mode
        self.domain_adaptation = domain_adaptation
        self._fully_initialized = True
        self.val_divs = val_divs
        self.repeat = repeat
        if data_path is None:
            if self.domain_adaptation:
                self.pkl_files = sorted(
                    glob.glob(
                        f'data/waymo/waymo_format/records_shuffled/domain_adaptation/{mode_folders[self.mode]}/*.pkl'
                    )+
                    glob.glob(
                        f'data/waymo/waymo_format/records_shuffled/domain_adaptation/{mode_folders[self.mode]}/unlabeled/*.pkl'
                    ))
            else:
                self.pkl_files = sorted(
                    glob.glob(
                        f'data/waymo/waymo_format/records_shuffled/{mode_folders[self.mode]}/pre_data/*.pkl'
                    ))
        elif files_txt is None:
            self.pkl_files = sorted(
                glob.glob(
                    f'{data_path}/*.pkl'
                ))
        else:
            self.pkl_files = []
            with open(files_txt, 'r') as f:
                line = f.readline()
                while line:
                    self.pkl_files.append(os.path.join(data_path, line))
                    line = f.readline()
        self.skips_n = skips_n

        self.length = len(self.pkl_files) // self.skips_n

        Det3DDataset.__init__(
            self,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            show_ins_var=show_ins_var,
            **kwargs)

    def update_CDA_epoch(self, epoch):
        """updates CurriculumDataAugmentation intensity by passing epoch num

        Args:
            epoch (int): current epoch
        """
        for t in self.pipeline.transforms:
            if isinstance(t, CurriculumDataAugmentation):
                t.set_epoch(epoch)

    def set_ps_updater(self, ps_updater):
        """Sets Pseudo label updater to LoadPseudoLabels transformation

        Args:
            ps_updater (PseudoUpdater): obj. responsible for saving and
                pseudo labels
        """
        for t in self.pipeline.transforms:
            if isinstance(t, LoadPseudoLabels):
                t.set_ps_updater(ps_updater)

    def __getitem__(self, index) -> dict:
        return self.pipeline(self.pkl_files[index*self.skips_n])

    def __len__(self) -> int:
        return self.length
