import torch
import numpy as np

from waymo_open_dataset import dataset_pb2


"""
From Waymo Open Dataset python library, transformed code to pytorch
"""


def compute_inclination(inclination_range, height):
    """Computes uniform inclination range based on the given range and height.

    Args:
        inclination_range: [..., 2] tensor. Inner dims are [min inclination, max inclination].
        height: an integer indicating height of the range image.

    Returns:
        inclination: [..., height] tensor. Inclinations computed.
    """
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (0.5 + torch.arange(0, height, dtype=inclination_range.dtype)) /
        height * diff + inclination_range[..., 0:1]
    )
    return inclination

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This can be
    useful for maintaining information about the shapes in PyTorch, although it
    doesn't have direct static/dynamic dimensions like TensorFlow.

    Args:
        tensor: A PyTorch tensor of any type.

    Returns:
        A list containing integers (for static dimensions) or tensors (for dynamic dimensions).
    """
    static_tensor_shape = tensor.shape
    dynamic_tensor_shape = tensor.size()
    combined_shape = []

    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(int(dim))
        else:
            combined_shape.append(dynamic_tensor_shape[index])

    return combined_shape

def get_rotation_matrix(roll, pitch, yaw, name=None):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    Args:
        roll : x-rotation in radians.
        pitch: y-rotation in radians. The shape must be the same as roll.
        yaw: z-rotation in radians. The shape must be the same as roll.
        name: the op name (not needed in PyTorch).

    Returns:
        A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3, 3].
    """
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)

    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    r_roll = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_roll, -1.0 * sin_roll], dim=-1),
        torch.stack([zeros, sin_roll, cos_roll], dim=-1),
    ],
                        dim=-2)
    r_pitch = torch.stack([
        torch.stack([cos_pitch, zeros, sin_pitch], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-1.0 * sin_pitch, zeros, cos_pitch], dim=-1),
    ],
                         dim=-2)
    r_yaw = torch.stack([
        torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], dim=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ],
                       dim=-2)

    return torch.matmul(r_yaw, torch.matmul(r_pitch, r_roll))

def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Args:
        rotation: [..., N, N] rotation tensor.
        translation: [..., N] translation tensor. This must have the same type as
          rotation.

    Returns:
        transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
          rotation.
    """
    transform = torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)
    last_row = torch.zeros_like(translation)
    last_row = torch.cat([last_row, torch.ones_like(last_row[..., 0:1])], dim=-1)
    transform = torch.cat([transform, last_row.unsqueeze(-2)], dim=-2)
    return transform

def compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose=None, frame_pose=None, dtype=torch.float32):
    """Computes range image cartesian coordinates from polar ones.

    Args:
        range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
          coordinate in sensor frame.
        extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
          range image pixel.
        frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
          It decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
          inclination sometimes have higher resolution than range_image.

    Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = range_image_polar.to(dtype)
    extrinsic = extrinsic.to(dtype)
    if pixel_pose is not None:
        pixel_pose = pixel_pose.to(dtype)
    if frame_pose is not None:
        frame_pose = frame_pose.to(dtype)

    azimuth, inclination, range_image_range = torch.unbind(range_image_polar, dim=-1)

    cos_azimuth = torch.cos(azimuth)
    sin_azimuth = torch.sin(azimuth)
    cos_incl = torch.cos(inclination)
    sin_incl = torch.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = torch.stack([x, y, z], dim=-1)
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = extrinsic[..., 0:3, 3].unsqueeze(1).unsqueeze(1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = torch.einsum('bkr,bijr->bijk', (rotation, range_image_points)) + translation
    if pixel_pose is not None:
        # To global frame.
        # [B, H, W, 3, 3]
        pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
        # [B, H, W, 3]
        pixel_pose_translation = pixel_pose[..., 0:3, 3]
        # [B, H, W, 3]
        range_image_points = torch.einsum(
            'bhwij,bhwj->bhwi', (pixel_pose_rotation, range_image_points)) + pixel_pose_translation
        if frame_pose is None:
            raise ValueError('frame_pose must be set when pixel_pose is set.')
        # To vehicle frame corresponding to the given frame_pose
        # [B, 4, 4]
        world_to_vehicle = torch.inverse(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
        world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
        # [B, H, W, 3]
        range_image_points = torch.einsum(
            'bij,bhwj->bhwi', (world_to_vehicle_rotation, range_image_points)) + world_to_vehicle_translation.unsqueeze(1).unsqueeze(1)

    range_image_points = range_image_points.to(dtype=range_image_polar_dtype)
    return range_image_points

def compute_range_image_polar(range_image, extrinsic, inclination, dtype=torch.float32):
    """Computes range image polar coordinates.

    Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
        inclination: [B, H] tensor. Inclination for each row of the range image.
          0-th entry corresponds to the 0-th row of the range image.
        dtype: float type to use internally. This is needed as extrinsic and
          inclination sometimes have higher resolution than range_image.

    Returns:
        range_image_polar: [B, H, W, 3] polar coordinates.
    """
    _, height, width = range_image.shape
    range_image_dtype = range_image.dtype
    range_image = range_image.to(dtype)
    extrinsic = extrinsic.to(dtype)
    inclination = inclination.to(dtype)

    with torch.no_grad():
        az_correction = torch.atan2(extrinsic[:, 1, 0], extrinsic[:, 0, 0])
        ratios = (torch.arange(width, 0, -1).to(dtype) - 0.5) / width
        azimuth = ((ratios * 2.0 - 1.0) * torch.tensor([3.141592653589793], dtype=dtype) - az_correction.view(-1, 1))

        azimuth_tile = azimuth.unsqueeze(1).expand(-1, height, -1)
        inclination_tile = inclination.unsqueeze(2).expand(-1, -1, width)

        range_image_polar = torch.stack([azimuth_tile, inclination_tile, range_image], dim=-1)
    
    return range_image_polar.to(dtype=range_image_dtype)

def extract_point_cloud_from_range_image(range_image, extrinsic, inclination, pixel_pose=None, frame_pose=None, dtype=torch.float32):
    """Extracts point cloud from range image.

    Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
        inclination: [B, H] tensor. Inclination for each row of the range image.
          0-th entry corresponds to the 0-th row of the range image.
        pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
          image pixel.
        frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
          decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
          inclination sometimes have higher resolution than range_image.

    Returns:
        range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
    """
    with torch.no_grad():
        range_image_polar = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian = compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose=pixel_pose, frame_pose=frame_pose, dtype=dtype)
    
    return range_image_cartesian

def convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, ri_index=0, keep_polar_features=False):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
        keep_polar_features: If true, keep the features from the polar range image
          (i.e., range, intensity, and elongation) as the first features in the
          output range image.

    Returns:
        dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
          will be 3 if keep_polar_features is False (x, y, z) and 6 if
          keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = torch.tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = torch.reshape(
        torch.tensor(range_image_top_pose.data),
        tuple(range_image_top_pose.shape.dims))
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = compute_inclination(
                torch.tensor([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = torch.tensor(c.beam_inclinations)

        beam_inclinations = torch.flip(beam_inclinations, [0])
        extrinsic = torch.reshape(torch.tensor(c.extrinsic.transform), [4, 4])

        range_image_tensor = torch.reshape(
            torch.tensor(range_image.data), tuple(range_image.shape.dims))
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor.unsqueeze(0)
            frame_pose_local = frame_pose.unsqueeze(0)
        range_image_cartesian = extract_point_cloud_from_range_image(
            range_image_tensor[..., 0].unsqueeze(0),
            extrinsic.unsqueeze(0),
            beam_inclinations.unsqueeze(0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = range_image_cartesian.squeeze(0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = torch.cat(
                [range_image_tensor[..., 0:3], range_image_cartesian], dim=-1)

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0, keep_polar_features=False):
    """Convert range images to point cloud.

    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name, [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
        keep_polar_features: If true, keep the features from the polar range image
          (i.e. range, intensity, and elongation) as the first features in the
          output range image.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          (NOTE: Will be {[N, 6]} if keep_polar_features is true.
        cp_points: {[N, 6]} list of camera projections of length 5
          (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []

    cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = torch.reshape(
            torch.tensor(range_image.data), tuple(range_image.shape.dims))
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = range_image_cartesian[range_image_mask]

        cp = camera_projections[c.name][ri_index]
        cp_tensor = torch.reshape(torch.tensor(cp.data), tuple(cp.shape.dims))
        cp_points_tensor = cp_tensor[range_image_mask]
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points, cp_points
