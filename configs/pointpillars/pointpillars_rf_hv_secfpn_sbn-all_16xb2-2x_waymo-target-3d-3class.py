_base_ = [
    '../_base_/models/pointpillars_rf_dla_hv_secfpn_waymo_ped.py',
    '../_base_/datasets/waymo-rf-target-3d-3class.py',
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
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=2))                                                                                                                                
load_from='models_weights/pp_dla_fixed.pth'#pp_TS_hv_48.pth'
resume=True
