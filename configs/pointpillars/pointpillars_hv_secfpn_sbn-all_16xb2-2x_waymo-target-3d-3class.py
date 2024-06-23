_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymo-target-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]

lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=24,
        end=36,
        by_epoch=True,
        milestones=[28, 32],
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=2)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=2))                                                                                                                                
# load_from='models_weights/pp_TS_hv_re_30.pth'
# load_from = 'work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-target-3d-3class/epoch_32.pth'#
load_from = 'models_weights/pp_std_hv_24_fixed.pth'
resume=True
