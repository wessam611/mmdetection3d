# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.runner.runner import Runner
from mmdet3d.registry import RUNNERS
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop, LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader

from mmengine.config import Config, ConfigDict

ConfigType = Union[Dict, Config, ConfigDict]

@RUNNERS.register_module()
class ST3DRunner(Runner):
    def __init__(
        self,
        target_train_dataloader: Union[DataLoader, Dict],
        target_val_dataloader: Union[DataLoader, Dict],
        update_cfg: Union[BaseLoop, Dict],
        *args,
        **kwargs
    ):
        self._target_train_dataloader = target_train_dataloader
        self._target_val_dataloader = target_val_dataloader
        self._update_loop = update_cfg
        super().__init__(*args, **kwargs)

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if not ('type' in loop_cfg and loop_cfg['type'] == 'ST3DTrainLoop'):
            raise RuntimeError(
                'ST3DRunner only support ST3DTrainLoop as a loop type')
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self, target_dataloader=self._target_train_dataloader,
                source_dataloader=self._train_dataloader,))
        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'val_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)
        if not ('type' in loop_cfg and loop_cfg['type'] == 'ST3DValLoop'):
            raise RuntimeError(
                'ST3DRunner only support ST3DValLoop as a loop type')
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self,
                target_dataloader=self._target_val_dataloader,
                source_dataloader=self._val_dataloader,
                evaluator=self._val_evaluator))

        return loop  # type: ignore

    def build_update_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build pseudo-label update loop.

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.
        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'Update_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)
        if not ('type' in loop_cfg and loop_cfg['type'] == 'ST3DUpdateLoop'):
            raise RuntimeError(
                'ST3DRunner only support ST3DUpdateLoop as an update loop type')
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self,
                train_dataloader=self._target_train_dataloader,
                val_dataloader=self._target_val_dataloader))
        return loop  # type: ignore

    @property
    def update_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._update_loop, BaseLoop) or self._update_loop is None:
            return self._update_loop
        else:
            self._update_loop = self.build_update_loop(self._update_loop)
            return self._update_loop

    @property
    def target_train_dataloader(self):
        return self._target_train_dataloader

    @property
    def target_val_dataloader(self):
        return self._target_val_dataloader

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            target_train_dataloader=cfg.get('target_train_dataloader'),
            target_val_dataloader=cfg.get('target_val_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            update_cfg=cfg.get('update_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner
