# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop, BaseLoop
from mmdet3d.registry import LOOPS, FUNCTIONS

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Sequence

from mmengine.evaluator import Evaluator
from mmdet3d.datasets import IterWaymoDataset


@LOOPS.register_module()
class ST3DTrainLoop(EpochBasedTrainLoop):
    """

    Args:
        EpochBasedTrainLoop (_type_): _description_
    """
    def __init__(
            self,
            runner,
            source_dataloader: Union[DataLoader, Dict],
            target_dataloader: Union[DataLoader, Dict],
            ps_update_start,
            ps_update_interval,
            src_loss_weight=0,
            checkpoint=None,
            *args, **kwargs) -> None:
        self._runner = runner
        if isinstance(source_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.source_dataloader = self.runner.build_dataloader(
                source_dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.source_dataloader = source_dataloader
        if isinstance(target_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.target_dataloader = self.runner.build_dataloader(
                target_dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.target_dataloader = target_dataloader

        if isinstance(self.target_dataloader.dataset, IterWaymoDataset):
            self.target_dataloader.dataset.set_ps_updater(
                self.runner.update_loop.ps_label_updater
            )
        super(ST3DTrainLoop, self).__init__(runner=runner, dataloader=self.source_dataloader, *args, **kwargs)
        self.ps_update_start = ps_update_start
        self.ps_update_interval = ps_update_interval
        self.train_phase = 'train_source'
        self.src_loss_weight = src_loss_weight
        if checkpoint != None:
            self.runner.model.load_state_dict(checkpoint)

    def switch_phase(self, train_phase):
        """
        """
        assert train_phase in ['train_source', 'label_target', 'train_target']
        self.train_phase = train_phase
        self.dataloader = self.source_dataloader if train_phase ==\
            'train_source' else self.target_dataloader

    def update_dataloader_intensity(self, epoch):
        self.target_dataloader.dataset.update_CDA_epoch(epoch)

    def step_source(self):
        """
        executes one train_step from source_data while training on target 
        data
        """

        if self.src_loss_weight == 0 or self.train_phase != 'train_target':
            return
        if not hasattr(self, 'source_iter'):
            self.source_iter = iter(self.source_dataloader)
        try:
            data_batch = next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_dataloader)
            data_batch = next(self.source_iter)
        self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        # TODO pass self.src_loss_weight to optimizerw

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.step_source()
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def run_epoch(self, ) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        if self.train_phase == 'label_target':
            self.runner.model.eval()
            self.runner.update_loop.run()
            self.switch_phase('train_target')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            if self._epoch >= self.ps_update_start:
                if self._epoch%self.ps_update_interval == 0:
                    self.switch_phase('label_target')
                else:
                    self.switch_phase('train_target')
            else:
                self.switch_phase('train_source')
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run(self._epoch)

        self.runner.call_hook('after_train')
        return self.runner.model

@LOOPS.register_module()
class ST3DValLoop(ValLoop):
    """

    Args:
        EpochBasedTrainLoop (_type_): _description_
    """
    def __init__(self,
                 runner,
                 source_dataloader: Union[DataLoader, Dict],
                 target_dataloader: Union[DataLoader, Dict],
                 ps_update_start: int,
                 *args, **kwargs) -> None:
        self._runner = runner
        self.ps_update_start = ps_update_start
        if isinstance(source_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.source_dataloader = self.runner.build_dataloader(
                source_dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.source_dataloader = source_dataloader
        if isinstance(target_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.target_dataloader = self.runner.build_dataloader(
                target_dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.target_dataloader = target_dataloader

        if isinstance(self.target_dataloader.dataset, IterWaymoDataset):
            self.target_dataloader.dataset.set_ps_updater(
                self.runner.update_loop.ps_label_updater
            )
        super(ST3DValLoop, self).__init__(runner=runner, dataloader=self.source_dataloader, *args, **kwargs)

    def run(self, epoch) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        if epoch >= self.ps_update_start:
            dataloader = self.target_dataloader
        else:
            dataloader = self.source_dataloader
        for idx, data_batch in enumerate(dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

@LOOPS.register_module()
class ST3DUpdateLoop(BaseLoop):
    """
    """
    def __init__(self,
                 runner,
                 val_dataloader: Union[DataLoader, Dict],
                 train_dataloader: Union[DataLoader, Dict],
                 ps_label_updater) -> None:
        self._runner = runner
        if isinstance(train_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.train_dataloader = runner.build_dataloader(
                train_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.train_dataloader = train_dataloader
        if isinstance(val_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.val_dataloader = runner.build_dataloader(
                val_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.val_dataloader = val_dataloader

        if isinstance(ps_label_updater, (dict, list)):
            self.ps_label_updater = FUNCTIONS.build(ps_label_updater)
        else:
            raise ValueError('pseudo label updater is only p as a '+
                             'dictionary describing a function under '+
                                'the FUNCTIONS registry')
        if isinstance(self.train_dataloader.dataset, IterWaymoDataset):
            self.train_dataloader.dataset.set_ps_updater(
                self.ps_label_updater
            )
        if isinstance(self.val_dataloader.dataset, IterWaymoDataset):
            self.val_dataloader.dataset.set_ps_updater(
                self.ps_label_updater
            )

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.train_dataloader):
            self.runner.call_hook(
                f'before_update_epoch')
            self.run_iter(idx, data_batch, 'train')
        for idx, data_batch in enumerate(self.val_dataloader):
            self.runner.call_hook(
                f'before_update_epoch')
            self.run_iter(idx, data_batch, 'val')
        self.ps_label_updater.finish_update()

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], split):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook('before_update_iter')
        # outputs should be sequence of BaseDataElement
        with torch.no_grad():
            outputs = self.runner.model.val_step(data_batch)
        pos_ps_nmeter, ign_ps_nmeter = self.ps_label_updater(outputs, data_batch)

        self.runner.call_hook('after_update_iter',
                              pos_ps_nmeter=pos_ps_nmeter,
                              ign_ps_nmeter=ign_ps_nmeter,
                              split=split,
                              batch_idx=idx)
