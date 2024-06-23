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

val_labels_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/base_gt/annotated_gt/'
train_labels_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/hv_base/hv_kalman_32222/'
input_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/pkl_data_aligned/'

class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
# db_sampler = dict(
#     rate=1.0,
#     batch_size=None,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
#     points_loader=dict(
#         type='LoadWaymoFrame',
#     ),
#     backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadWaymoFrame',
        use_dim=[0, 1, 2, -1, -1],
        filter_nlz_points=False,
        with_bbox_3d=False,
        with_label_3d=False,
        shift_height=0,
        pkl_files_path=input_pth),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.2,
        flip_ratio_bev_vertical=0.8),
    dict(
        type='RandomObjectScaling',
        scale_p=0.2,
        scale_range=[0.95, 1.05]),
    dict(
        dict(
        type='RandomObjectNoise',
        noise_p=0.5,
        noise_range=[-0.1, 0.1]),
    ),  
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='CurriculumDataAugmentation',
        epoch_intensity=dict(
           [(0, (0.1, 0.2)),
            (24+2, (0.1, 0.3)),
            (24+4, (0.2, 0.4)),
            (24+6, (0.2, 0.6))]
        )
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadWaymoFrame',
        use_dim=[0, 1, 2, -1, -1],
        filter_nlz_points=False,
        with_bbox_3d=False,
        with_label_3d=False,
        shift_height=0,
        pkl_files_path=input_pth),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'num_lidar_points_in_box', 'gt_scores_3d'
        ],
        meta_keys=[
            'context', 'timestamp_micros', 'box_type_3d', 'box_mode_3d',
            'sample_idx'
        ]),
]

train_dataloader = dict(
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    batch_size=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            pipeline=train_pipeline,
            modality=input_modality,
            mode='train',
            metainfo=metainfo,
            repeat=True,
            val_divs=1,
            box_type_3d='LiDAR',
            skips_n=1,
            data_path=input_pth,
            labels_path=train_labels_pth,
            files_txt='/home/wessam/src/ransac_preprocessing/pointclouds/train_split_cl.txt',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    # prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        repeat=False,
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        mode='val',
        val_divs=1,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        skips_n=1,
        data_path=input_pth,
        labels_path=val_labels_pth,
        files_txt='/home/wessam/src/ransac_preprocessing/pointclouds/val_split.txt',))

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    prefetch_factor=1,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        pipeline=[
            dict(
                type='LoadPointsFromDict',
                norm_intensity=True,
                norm_elongation=True,
                coord_type='LIDAR',
                use_dim=[0, 1, 2, 3, 4],),
            dict(
                type='Pack3DDetInputs',
                keys=[
                    'points', 'gt_bboxes_3d', 'gt_labels_3d',
                    'num_lidar_points_in_box'
                ],
                meta_keys=[
                    'context', 'timestamp_micros', 'box_type_3d',
                    'box_mode_3d', 'sample_idx'
                ]),
        ],
        modality=input_modality,
        test_mode=True,
        val_divs=1,
        mode='test',
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        skips_n=1))

val_evaluator = dict(
    type='IterWaymoMetric',
    ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='.data/waymo/waymo_format/gt.bin',
    data_root='gs://waymo_open_dataset_v_1_4_1/individual_files/',
    backend_args=backend_args,
    convert_kitti_format=False)
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
