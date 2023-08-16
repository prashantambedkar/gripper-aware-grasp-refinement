import os

import torch
import numpy as np
import burg_toolkit as burg

from ..utils.transform import transform_points

SCENE_EDGE_LENGTH = 0.297


test_objects = [
    '004_sugar_box',
    '009_gelatin_box',
    '061_foam_brick',
    '077_rubiks_cube',
    '010_potted_meat_can',
    '025_mug',
    '024_bowl',
    '065-a_cups',
    '065-h_cups',
    '005_tomato_soup_can',
    '011_banana',
    '056_tennis_ball',
    '055_baseball',
    '006_mustard_bottle',
]

# objects that are in the library but should not be in the training dataset
ignore_objects = [
    '028_skillet_lid',
    '073-g_lego_duplo',
]


def load_object_library(dataset_dir, split=None):
    """ Loads the object library, optionally specify the split.

    Args:
        dataset_dir: base directory of dataset, usually data/gag/ (directory should contain an object_library subdir)
        split (str): optional, train | test to get corresponding lib. will by default return library of all objects

    Returns:
        burg.ObjectLibrary
    """
    assert split in ['train', 'test', None]
    print(dataset_dir)
    object_library_fn = os.path.join(dataset_dir, 'object_library/object_library.yaml')
    if not os.path.exists(object_library_fn):
        raise FileNotFoundError(f'requested to load object library from {object_library_fn} but the file could not '
                                f'be found. You might need to run `python data/download_data.py`, or check your paths')

    # load object library
    object_library = burg.ObjectLibrary.from_yaml(object_library_fn)

    # remove ignored objects
    for obj in ignore_objects:
        object_library.pop(obj, None)  # none so it does not fail if obj is not in library

    # full object library with train and test objects
    if split is None:
        return object_library

    # remove all objects that do not belong to the chosen split
    if split == 'train':
        for obj in test_objects:
            object_library.pop(obj, None)  # none so it does not fail if obj is not in library
    else:
        keys_to_remove = set(object_library.keys()) - set(test_objects)
        for key in keys_to_remove:
            object_library.pop(key)
        assert len(object_library) == len(test_objects), 'not all test objects present in object library'

    return object_library


def get_scenes_from_split(base_dir, split):
    """ Gives all the items from the train.lst/test.lst files.
    Args:
        base_dir (str): the directory that contains the split.lst
        split (str): train, val, test

    Returns:
        list(str), with all elements of the split
    """
    split_lst_fn = os.path.join(base_dir, f'{split}.lst')
    with open(split_lst_fn, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    return lines


def pre_normalisation_tf():
    """
    In our dataset, all points are in the range of [0, SCENE_EDGE_LENGTH]. However, ConvONet expects input in the range
    of [-0.5, 0.5] (plus padding). In order to comply, we need to apply this pre_normalisation to both point cloud
    points and occupancy points before feeding our data to ConvONet; we set the padding to 0.
    """
    tf = torch.eye(4)
    tf[:3, :3] /= SCENE_EDGE_LENGTH  # scaling to [0, 1]
    tf[:3, 3] -= 0.5  # shifting to [-0.5, 0.5]
    return tf


def pre_normalise_points(points):
    """
    Apply pre_normalisation_tf to the points.
    Args:
        points (torch.Tensor): BxNx3 or Nx3

    Returns:
        torch.Tensor, transformed points BxNx3 or Nx3
    """
    as_numpy = False
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
        as_numpy = True

    tf = pre_normalisation_tf().to(points.device)
    if len(points.shape) == 3:
        tf = tf.unsqueeze(0).repeat(points.shape[0], 1, 1)
    else:
        assert len(points.shape) == 2, f'unexpected points.shape: {points.shape}'

    tf_points = transform_points(tf, points)

    if as_numpy:
        tf_points = tf_points.numpy()
    return tf_points


def balanced_sample_sizes(n_sample_total, n_splits):
    """
    Divides n_sample_total into n_splits, such that the splits are as equal as possible and sum up to n_sample_total.
    Will randomly distribute which splits get additional items.
    Useful for e.g. balancing class frequencies while subsampling.

    Args:
        n_sample_total (int): number of samples in total
        n_splits (int): number of splits to divide the samples into

    Returns:
        np.ndarray, (n_splits,) with n_sample for each split
    """
    n, remainder = divmod(n_sample_total, n_splits)
    n_samples = np.full(n_splits, fill_value=n, dtype=int)

    # choose randomly how to distribute the remainder
    idcs = np.random.choice(n_splits, remainder, replace=False)
    n_samples[idcs] += 1

    return n_samples
