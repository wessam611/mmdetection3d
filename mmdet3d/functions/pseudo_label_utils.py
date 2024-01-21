"""
From ST3D repository https://github.com/CVMI-Lab/ST3D/blob/master/docs/GETTING_STARTED.md
"""
import copy
import torch
import numpy as np

from mmcv.ops import boxes_iou3d


###################### TRACKING AVERAGE ######################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NAverageMeter(object):
    """
    Contain N AverageMeter and update respectively or simultaneously
    """
    def __init__(self, n):
        self.n = n
        self.meters = [AverageMeter() for i in range(n)]

    def update(self, val, index=None, attribute='avg'):
        if isinstance(val, list) and index is None:
            assert len(val) == self.n
            for i in range(self.n):
                self.meters[i].update(val[i])
        elif isinstance(val, NAverageMeter) and index is None:
            assert val.n == self.n
            for i in range(self.n):
                self.meters[i].update(getattr(val.meters[i], attribute))
        elif not isinstance(val, list) and index is not None:
            self.meters[index].update(val)
        else:
            raise ValueError
    def __iter__(self):
        return iter(self.meters)
    def aggregate_result(self):
        result = "("
        for i in range(self.n):
            result += "{:.3f},".format(self.meters[i].avg)
        result += ')'
        return result
    
###################### TRACKING AVERAGE ######################

###################### ENSEMBLE HELPERS ######################

def mask_dict(result_dict, mask):
    new_dict = copy.deepcopy(result_dict)
    for key, value in new_dict.items():
        if value is None:
            new_dict[key] = None
        else:
            new_dict[key] = value[mask]
    return new_dict

def concatenate_array_inside_dict(merged_dict, result_dict):
    for key, val in result_dict.items():
        if key not in merged_dict:
            merged_dict[key] = copy.deepcopy(val)
        else:
            merged_dict[key] = np.concatenate([merged_dict[key], copy.deepcopy(val)])

    return merged_dict

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    if isinstance(x, np.float64) or isinstance(x, np.float32):
        return torch.tensor([x]).float(), True
    return x, False
###################### ENSEMBLE HELPERS ######################
