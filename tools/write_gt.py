# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tqdm
import logging
import pickle
import os
import os.path as osp
import torch
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmdet3d.registry import RUNNERS, MODELS, DATASETS, TRANSFORMS
from mmdet3d.utils import replace_ceph_backend


device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('score_th', default=0.3, type=float, help='score threshold for written pseudo labels.')
    args = parser.parse_args()

    return args

def to_device(val, dev=device):
    if isinstance(val, dict):
        return dict([(k, to_device(d, dev)) for k, d in val.items()])
    elif isinstance(val, list):
        return [to_device(d, dev) for d in val]
    else:
        return val.to(dev)
max = 0
def write_gt(out, target_path, score_th):
    global max
    bboxes = out.pred_instances_3d['bboxes_3d'].tensor
    bboxes = bboxes.to('cpu')
    bboxes = bboxes.numpy()
    classes = out.pred_instances_3d['labels_3d']
    classes = classes.to('cpu')
    classes = classes.numpy()
    scores = out.pred_instances_3d['scores_3d']
    scores = scores[classes==1].cpu().numpy()
    bboxes = bboxes[classes==1][scores>score_th]
    scores = scores[scores>score_th]
    data_dict = {
        'frame': None,
        'points': None,
        'nlz_points': None,
        'range_index': None,
        'bboxes': bboxes,
        'scores': scores
    }
    with open(os.path.join(target_path), 'wb') as f_pkl:
        pickle.dump(data_dict, f_pkl)


def main():
    args = parse_args()
    score_th = args.score_th
    # load config
    cfg = Config.fromfile(args.config)
    cfg['model']['data_preprocessor'] = MODELS.build(cfg['model']['data_preprocessor'])
    model = MODELS.build(cfg['model'])
    model = model.to(device)
    ckpt = _load_checkpoint(cfg['load_from'])
    _load_checkpoint_to_model(model, ckpt, True)
    model.eval()
    for i, p in enumerate(cfg['val_dataloader']['dataset']['pipeline']):
        cfg['val_dataloader']['dataset']['pipeline'][i] = TRANSFORMS.build(p)
    cfg['val_dataloader']['dataset'] = DATASETS.build(cfg['val_dataloader']['dataset'])
    dataloader = Runner.build_dataloader(cfg.get('val_dataloader'))
    target_path = cfg['target_path']
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with torch.no_grad():
        for i, (data_batch, paths) in tqdm.tqdm(enumerate(dataloader)):
            paths = [path.split('/')[-1] for path in paths]
            data_batch = to_device(data_batch)
            outputs = model.test_step(data_batch)
            for path, out in zip(paths, outputs):
                write_gt(out, os.path.join(target_path, path), score_th)


if __name__ == '__main__':
    main()
