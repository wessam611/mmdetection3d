_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymo-bags-3d-3class.py'
]


target_path = '/home/wessam/src/ransac_preprocessing/pointclouds/base_gt/hv_base/'
# load_from='models_weights/pp_std_hv_24_fixed.pth'
load_from ='work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-iter-3d-3class/epoch_24.pth'
