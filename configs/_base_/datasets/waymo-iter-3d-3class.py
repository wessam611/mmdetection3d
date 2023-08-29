# dataset settings
dataset_type = 'IterWaymoDataset'
data_root = 'data/waymo/kitti_format/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/waymo/kitti_format/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = {}

class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    rate=1.0,
    batch_size=None,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadWaymoFrame',
    ),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadWaymoFrame',
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadWaymoFrame',
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadWaymoFrame',
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'], meta_keys=['context', 'timestamp_micros', 'box_type_3d', 'box_mode_3d', 'sample_idx']),
]

train_dataloader = dict(
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=2,
    batch_size=1,
    dataset=
        dict(
            type=dataset_type,
            pipeline=train_pipeline,
            modality=input_modality,
            mode='train',
            metainfo=metainfo,
            repeat=True,
            val_divs=1,
            box_type_3d='LiDAR',
            backend_args=backend_args
            ))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    prefetch_factor=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        repeat = False,
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        mode='val',
        val_divs=2,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=1,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        val_divs=1,
        mode='test',
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='IterWaymoMetric',
    ann_file=None,
    waymo_bin_file='./data/waymo/waymo_format/gt.bin',
    data_root=None,
    backend_args=backend_args,
    convert_kitti_format=False)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
