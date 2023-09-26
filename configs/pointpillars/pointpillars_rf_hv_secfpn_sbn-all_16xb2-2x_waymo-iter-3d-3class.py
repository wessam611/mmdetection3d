_base_ = [
    '../_base_/models/pointpillars_rf_hv_secfpn_waymo.py',
    '../_base_/datasets/waymo-rf-iter-3d-3class.py',
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
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=120000,
    val_interval=14000,
    _delete_=True)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=15000))
