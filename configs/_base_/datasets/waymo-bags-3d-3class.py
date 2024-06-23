# dataset settings
dataset_type = 'IterWaymoDataset'
# data_root = 'data/waymo/kitti_format/'
# 'data/waymo/waymo_format/records_shuffled/validation/pre_data/'
data_path = '/home/wessam/src/ransac_preprocessing/pointclouds/pkl_data_aligned/'
train_split_path = '/home/wessam/src/ransac_preprocessing/pointclouds/train_split_cl.txt'
val_split_path = '/home/wessam/src/ransac_preprocessing/pointclouds/val_split_cl.txt'

backend_args = {}

class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)

eval_pipeline = [
    dict(
        type='LoadWaymoFrame',
        pkl_files_path=data_path,
        use_dim=[0, 1, 2, -1, -1],
        filter_nlz_points=False,
        with_bbox_3d=True,
        with_label_3d=True,
        # range_index=True,
        # range_image=True,
        # norm_intensity=True,
        # norm_elongation=True,
        shift_height=0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d'#, 'num_lidar_points_in_box' 'range_image', 'range_index', 
        ],
        meta_keys=[
            'context', 'timestamp_micros', 'box_type_3d', 'box_mode_3d',
            'sample_idx'
        ]
    )
]

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    prefetch_factor=2,
    # persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        repeat=False,
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        skips_n=1,
        # files_txt=val_split_path,
        data_path=data_path))
