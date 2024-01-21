# common functions used for transforms
import numpy as np
import torch

def get_bbox_mask(points_cp, points_mask, box, return_scaled=False):
    """_summary_

    Args:
        points_cp (np.array): points
        points_mask (np.array): mask of obj points
        box (np.array (-1, 7)): corresponding box
        return_scaled (bool, optional): whether to return points.
            Defaults to False.
    Returns:
        points_mask
        points_mask, Scaled_points
    """
    pp = points_cp[points_mask]
    pp = center_bbox_points(pp, box)
    e = 2e-2
    p1 = pp[:, 0] >= -box[3] / 2 - e
    p2 = pp[:, 0] <= box[3] / 2 + e
    p3 = pp[:, 1] >= -box[4] / 2 - e
    p4 = pp[:, 1] <= box[4] / 2 + e
    p5 = pp[:, 2] >= -box[5] / 2 - e
    p6 = pp[:, 2] <= box[5]/ + e
    pp_mask = torch.logical_and(
        p1,
        torch.logical_and(
            p2,
            torch.logical_and(
                p3, torch.logical_and(p4, torch.logical_and(p5, p6)))))
    points_mask[points_mask.clone()] = pp_mask
    pp = pp[pp_mask]
    if return_scaled:
        return points_mask, pp
    return points_mask

def center_bbox_points(points, box, inverse=False):
    """Points in obj's coordinates

    Args:
        points (np.ndarray): obj's points
        box (np.ndarray (-1, 7)): obj box
        inverse (bool, optional): reverse transformation. Defaults to False.

    Returns:
        np.ndarray: points in obj's coordinates
    """
    R = torch.eye(3)
    R[0, 0] = np.cos(-box[-1])
    R[0, 1] = -np.sin(-box[-1])
    R[1, 0] = np.sin(-box[-1])
    R[1, 1] = np.cos(-box[-1])
    if inverse:
        points[:, :3] = points[:, :3] @ R.transpose(0, 1)
        points[:, :3] = points[:, :3] + box[:3]
    else:
        points[:, :3] = points[:, :3] - box[:3]
        points[:, :3] = points[:, :3] @ R
    return points

def get_axis_aligned_box_mask(points, box):
    """get axis_aligned obj's box mask (box isn't tight)

    Args:
        points (np.ndarray): obj's points
        box (np.ndarray (-1, 7)): obj axis-aligned box

    Returns:
        np.ndarray: points mask
    """
    x, y, z, xs, ys, zs, yaw = box
    yaw_ = (-1) * yaw if yaw < 0 else yaw
    yaw_ = yaw_ - torch.pi if yaw_ >= torch.pi else yaw_
    yaw_ = torch.pi - yaw_ if yaw_ >= torch.pi / 2 else yaw_
    l_glob = xs * torch.cos(yaw_) + ys * torch.sin(yaw_)
    w_glob = xs * torch.sin(yaw_) + ys * torch.cos(yaw_)
    p1 = points[:, 0] >= (x - l_glob / 2)
    p2 = points[:, 0] <= (x + l_glob / 2)
    p3 = points[:, 1] >= (y - w_glob / 2)
    p4 = points[:, 1] <= (y + w_glob / 2)
    points_mask = torch.logical_and(
        torch.logical_and(p1, p2), torch.logical_and(p3, p4))
    return points_mask
