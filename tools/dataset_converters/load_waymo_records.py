import os
import pickle

import matplotlib.patches as patches
import numpy as np
import open3d as o3d
import tensorflow as tf
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)

tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import metrics_pb2

from mmdet3d.datasets.transforms.waymo_utils import \
    convert_range_image_to_point_cloud


def load_frame_inputs(frame) -> tuple:
    """extract frame's range-view images and convert them to pointclouds.

    Args:
        frame: waymo frame

    Returns:
        points: pointcloud
        nlz_points: no label zone
        range_index: pointcloud to range_image indices (assumes r0&r1, flattenned range_image)
    """
    results = {}
    frame.lasers.sort(key=lambda laser: laser.name)
    (range_images, camera_projections, seg_labels, range_image_top_pose
     ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points_0, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=0,
        keep_polar_features=True)
    points_0 = [points_0[l - 1] for l in [open_dataset.LaserName.TOP]]
    results['points'] = np.concatenate(
        points_0, axis=0)  # (range, intensity, elongation, x, y, z)

    points_1, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True)
    points_1 = [points_1[l - 1] for l in [open_dataset.LaserName.TOP]]
    points_1 = np.concatenate(points_1, axis=0)
    results['points'] = np.concatenate([results['points'], points_1])

    range_images = dict([(k, range_images[k])
                         for k in [open_dataset.LaserName.TOP]])
    camera_projections = dict([(k, camera_projections[k])
                               for k in [open_dataset.LaserName.TOP]])

    mask_index = np.full_like(results['points'][:, 0], -1)
    nlz_points = np.full_like(results['points'][:, 0], -1)
    for laser_name in range_images.keys():
        offset = 0
        for ri in [0, 1]:
            range_image = range_images[laser_name][ri]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0
            of_dif = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            nlz_points[offset:offset + of_dif] = range_image_tensor.numpy()[
                range_image_mask.numpy(), -1]
            if laser_name == open_dataset.LaserName.TOP:
                cur_mask_index = tf.where(range_image_mask)
                cur_mask_index = (
                    ri * range_image_mask.shape[0] + cur_mask_index[:, 0]
                ) * range_image_mask.shape[1] + cur_mask_index[:, 1]
                mask_index[offset:offset + of_dif] = cur_mask_index
            offset += of_dif

    results['range_index'] = mask_index
    results['nlz_points'] = nlz_points
    results['points'] = results['points'][:, [3, 4, 5, 1, 2, 0]]  # x, y, z,

    return results['points'], results['nlz_points'], results['range_index']


def create_pd_file_example(out_path, cloud_path):
    """Creates a prediction objects file."""
    cloud_bucket = tfds.core.Path(cloud_path)
    for split in ['training', 'validation', 'testing']:
        split_files = tf.io.gfile.glob(
            os.path.join(cloud_bucket, f'{split}/segment*'))

        def write_file(file):
            frame = open_dataset.Frame()
            ds = tf.data.TFRecordDataset([file],
                                         num_parallel_reads=1,
                                         compression_type='')
            if split == 'training':
                ds = ds.shuffle(200)
            with tf.io.TFRecordWriter(
                    f'{out_path}/{split}/{file.split("/")[-1]}') as writer:
                for elem in tqdm(ds):
                    frame.ParseFromString(bytearray(elem.numpy()))
                    points, nlz_points, range_index = load_frame_inputs(frame)

                    data_dict = {
                        'points': points,
                        'nlz_points': nlz_points,
                        'range_index': range_index
                    }
                    with open(
                            os.path.join(
                                out_path,
                                f'{split}/pre_data/{frame.context.name}_{frame.timestamp_micros}.pkl'
                            ), 'wb') as f_pkl:
                        pickle.dump(data_dict, f_pkl)

                    writer.write(bytearray(elem.numpy()))

        Parallel(n_jobs=64)(delayed(write_file)(file) for file in split_files)


create_pd_file_example(
    '/home/source/mmdetection3d/data/waymo/waymo_format/records_shuffled',
    'gs://waymo_open_dataset_v_1_4_1//individual_files/')
