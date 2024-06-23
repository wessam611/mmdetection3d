# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def index_unflatten(range_index, range_shape):
    _, W, _ = range_shape
    out = np.empty((range_index.shape[0], 2), range_index.dtype)
    out[:, 0] = np.floor_divide(range_index, W)
    out[:, 1] = np.mod(range_index, W)
    return out


def index_flatten(range_index, range_shape):
    _, W, _ = range_shape
    out = np.empty((range_index.shape[0], ), range_index.dtype)
    out = range_index[:, 0] * W + range_index[:, 1]
    return out


def rotate_range_image(range_image,
                       range_index=None,
                       angle=90,
                       rotate_index_only=False):
    """applies horizontal or vertical yaw rotation to the pointcloud.

    Args:
        range_image (np.array): H, W, C
        range_index (np.array, optional):
            maps from pointcloud to range_image. Defaults to None.
        angle (float): degrees by which the pointcloud is rotated around yaw
        rotate_index_only: applies the rotation to the index only while ignoring
            the range-image itself (used for rot_paste_from_mask)
    """
    assert not (rotate_index_only and range_index is None)
    _, W, _ = range_image.shape
    ri_avail = range_index is not None
    flattened = len(range_index.shape) == 1
    if ri_avail and flattened:
        range_index = index_unflatten(range_index, range_image.shape)

    pix_rot = int((angle / 360) * W)
    if not rotate_index_only:
        range_image = np.roll(range_image, pix_rot, 1)
    if ri_avail:
        range_index[:, 1] = np.mod(range_index[:, 1] + pix_rot, W)

    if ri_avail and flattened:
        range_index = index_flatten(range_index, range_image.shape)

    return range_image, range_index


def flip_range_image(range_image, range_index=None, direction='horizontal'):
    """applies horizontal or vertical yaw rotation to the pointcloud.

    Args:
        range_image (np.array): H, W, C
        range_index (np.array, optional):
            maps from pointcloud to range_image. Defaults to None.
        direction (str): direction of rotation ['horizontal', 'vertical']
    """
    _, W, _ = range_image.shape
    ri_avail = range_index is not None
    flattened = len(range_index.shape) == 1
    if ri_avail and flattened:
        range_index = index_unflatten(range_index, range_image.shape)

    if direction == 'horizontal':
        range_image = np.flip(range_image, axis=1)
        if ri_avail:
            range_index[:, 1] = W - range_index[:, 1]
    else:
        range_image, range_index = rotate_range_image(range_image, range_index,
                                                      90)
        range_image = np.flip(range_image, axis=1)
        if ri_avail:
            range_index[:, 1] = W - range_index[:, 1]
        range_image, range_index = rotate_range_image(range_image, range_index,
                                                      -90)

    if ri_avail and flattened:
        range_index = index_flatten(range_index, range_image.shape)

    return range_image, range_index
