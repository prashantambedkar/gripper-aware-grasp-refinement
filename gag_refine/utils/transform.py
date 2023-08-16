import torch
import numpy as np


def transform_points(tf, points):
    """
    Transforms the points with the given 4x4 transformation matrix.
    Can also handle numpy arrays.

    Args:
        tf (torch.tensor): Bx4x4 or 4x4 transformation matrix
        points (torch.tensor): BxNx3 or Nx3 points

    Returns:

    """
    if not torch.is_tensor(tf) or not torch.is_tensor(points):
        # check if we can delegate to numpy function, else raise Error
        if isinstance(tf, np.ndarray) and isinstance(points, np.ndarray):
            return _transform_points_np(tf, points)
        else:
            raise TypeError(f'Not a torch.Tensor. tf: {type(tf)}. points: {type(points)}')
    if not tf.device == points.device:
        raise TypeError(f'Not the same device: tf: {tf.device}. points: {points.device}')
    if len(tf.shape) != len(points.shape) or len(tf.shape) == 3 and not tf.shape[0] == points.shape[0]:
        raise ValueError(f'Shapes do not match: tf: {tf.shape}. points: {points.shape}')

    remember_dtype = None
    if points.dtype != torch.double:
        remember_dtype = points.dtype
        points = points.to(torch.double)
    tf = tf.to(points.dtype)

    # ensure batch size 1
    squeeze = False
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
        tf = tf.unsqueeze(0)
        squeeze = True

    points_h = torch.nn.functional.pad(points, [0, 1], 'constant', 1.0)  # to homogenous points
    tf_points_h = torch.matmul(tf.unsqueeze(1), points_h.unsqueeze(-1)).squeeze(-1)  # transform
    tf_points = tf_points_h[..., :-1]  # back from homogenous points

    if remember_dtype is not None:
        tf_points = tf_points.to(remember_dtype)

    if squeeze:
        tf_points = tf_points.squeeze(0)

    return tf_points


def _transform_points_np(tf, points):
    """
    Args:
        tf (numpy.ndarray): 4x4 transformation matrix
        points (numpy.ndarray): BxNx3 or Nx3 points

    Returns:
        numpy.ndarray: transformed points, BxNx3 or Nx3
    """
    if len(tf.shape) > 2:
        raise NotImplementedError(f'transformation of numpy points with batched tfs not implemented. tf: {tf.shape}')

    # make points homogenuous
    shape = points.shape
    points_h = np.ones((*shape[:-1], shape[-1] + 1))
    points_h[..., :-1] = points

    original_shape = points_h.shape  # remember shape for shaping back
    points_h = points_h.reshape(-1, 4)
    points_h = (tf @ points_h.T).T.reshape(original_shape)

    # back to cartesian
    points = points_h[..., :-1]
    return points
