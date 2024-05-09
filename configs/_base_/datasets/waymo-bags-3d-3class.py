# dataset settings
dataset_type = 'IterWaymoDataset'
# data_root = 'data/waymo/kitti_format/'
data_path = '/home/wessam/src/ransac_preprocessing/pointclouds/pkl_data_aligned/'
train_split_path = '/home/wessam/src/ransac_preprocessing/pointclouds/train_split.txt'
val_split_path = '/home/wessam/src/ransac_preprocessing/pointclouds/val_split.txt'

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
        with_bbox_3d=False,
        with_label_3d=False,
        shift_height=0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d'#, 'num_lidar_points_in_box'
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
        data_path=data_path))
