_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymo-iter-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]


point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
val_labels_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/base_gt/annotated_gt/'
train_labels_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/base_gt/hv_kalman/'
input_pth = '/home/wessam/src/ransac_preprocessing/pointclouds/pkl_data_aligned/'

lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

ps_label_updater = dict(
    type='PseudoLabelUpdater',
    psuedo_labels_dir='data/waymo/da_pseudo_labels/',
    neg_th=[0.15, 0.15, 0.15], # > negative examples
    pos_th=[0.5, 0.4, 0.4], # > ignored examples > neg_th
    pkl_prefix='ps_pkl',
    ensemble_function= dict(
        type='ConsistencyEnsemble', 
        iou_th=0.1,
        memory_voting=True,
        mv_ignore_th=2,
        mv_remove_th=3
    )
)
train_cfg = dict(type='ST3DTrainLoop', max_epochs=48, val_interval=8,
                 ps_update_start=24, ps_update_interval=4, src_loss_weight=0.8)
val_cfg = dict(type='ST3DValLoop', ps_update_start=24)
update_cfg = dict(type='ST3DUpdateLoop', ps_label_updater=ps_label_updater)
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
default_hooks = dict(
    logger=dict(type='ST3DLoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1))

target_train_pipeline = [
    dict(
        type='LoadWaymoFrame',
        use_dim=[0, 1, 2, -1, -1],
        filter_nlz_points=False,
        with_bbox_3d=True,
        with_label_3d=True,
        shift_height=0,
        pkl_files_path=input_pth),
    dict(
        type='LoadPseudoLabels',
    ),
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
           [(0, (0.05, 0.1)),
            (24+2, (0.1, 0.2)),
            (24+4, (0.2, 0.2)),
            (24+6, (0.2, 0.2))]
        )
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'context', 'timestamp_micros', 'box_type_3d', 'box_mode_3d',
            'sample_idx'
        ]),
]

target_eval_pipeline = [
    dict(
        type='LoadWaymoFrame',
        use_dim=[0, 1, 2, -1, -1],
        filter_nlz_points=False,
        with_bbox_3d=False,
        with_label_3d=False,
        shift_height=0,
        pkl_files_path=input_pth),
    
    dict(
        type='LoadPseudoLabels',
    ),
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

target_train_dataloader = dict(
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    batch_size=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type={{_base_.dataset_type}},
            pipeline=target_train_pipeline,
            modality={{_base_.input_modality}},
            mode='train',
            metainfo={{_base_.metainfo}},
            repeat=True,
            val_divs=1,
            box_type_3d='LiDAR',
            skips_n=1,
            data_path=input_pth,
            labels_path=train_labels_pth,
            files_txt='/home/wessam/src/ransac_preprocessing/pointclouds/train_split_cl.txt',
            backend_args={{_base_.backend_args}})))

target_val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    # persistent_workers=True,
    # prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # persistent_workers=False,
    dataset=dict(
        type={{_base_.dataset_type}},
        repeat=False,
        pipeline=target_eval_pipeline,
        modality={{_base_.input_modality}},
        test_mode=True,
        mode='val',
        val_divs=1,
        metainfo={{_base_.metainfo}},
        box_type_3d='LiDAR',
        backend_args={{_base_.backend_args}},
        skips_n=1,
        data_path=input_pth,
        labels_path=val_labels_pth,
        files_txt='/home/wessam/src/ransac_preprocessing/pointclouds/val_split.txt',))

runner_type = 'ST3DRunner'
# load_from='work_dirs/pointpillars_st3d_hv_secfpn_sbn-all_16xb2-2x_waymo-iter-3d-3class/epoch_24.pth'
# resume=True

load_from = 'models_weights/pp_std_hv_24_fixed.pth'
resume=True
