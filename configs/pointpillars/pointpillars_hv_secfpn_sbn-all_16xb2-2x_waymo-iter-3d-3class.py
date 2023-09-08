_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymo-iter-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]

lr = 0.0005
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=7000, _delete_=True)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=15000))
