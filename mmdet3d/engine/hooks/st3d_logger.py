import time

from mmdet3d.registry import HOOKS

from mmengine.hooks import LoggerHook

@HOOKS.register_module()
class ST3DLoggerHook(LoggerHook):
    """
    Custom logger to manage UpdateLoop 
    """
    CLASSES = ['car', 'pedestrian', 'cyclist']
    def before_update_epoch(self, runner):
        self.time_sec_test_val = 0
    def before_update_iter(self, runner):
        self.t = time.time()

    def after_update_iter(self, runner, pos_ps_nmeter, ign_ps_nmeter, split, batch_idx):
        scalars_dict = {}
        for i, v in enumerate(pos_ps_nmeter):
            key = f'{split}/average_pos_boxes/{self.CLASSES[i]}'
            scalars_dict[key] = v.avg
        for i, v in enumerate(ign_ps_nmeter):
            key = f'{split}/average_ign_boxes/{self.CLASSES[i]}'
            scalars_dict[key] = v.avg

        runner.visualizer.add_scalars(
            scalars_dict, step=runner.iter + 1, file_path=self.json_log_path)
        for k, v in scalars_dict.items():
            runner.message_hub.update_scalar(k, v)
        
        message_hub = runner.message_hub
        if split == 'train':
            cur_dataloader = runner.target_train_dataloader
        else:
            cur_dataloader = runner.target_val_dataloader

        message_hub.update_scalar(f'{split}/time', time.time() - self.t)
        iter_time = message_hub.get_scalar(f'{split}/time')
        self.time_sec_test_val += iter_time.current()
        time_sec_avg = self.time_sec_test_val / (batch_idx + 1)
        eta_sec = time_sec_avg * (len(cur_dataloader) - batch_idx - 1)
        runner.message_hub.update_info('eta', eta_sec)
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, split)
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)
