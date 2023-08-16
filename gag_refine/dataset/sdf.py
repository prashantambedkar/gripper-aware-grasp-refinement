import numpy as np
from scipy.spatial import cKDTree


def compute_sdf_values(points, occupancy, full_pc, check_z=True):
    """
    Computes the signed distance function for the points. I.e. it is the signed distance to the closest object surface
    in the scene. If the sign is negative, the point is within an object, if positive, outside.
    It approximates the SDF by computing the minimum distance to a point in the full scene point cloud.

    :param points: (n, 3) ndarray of points to check
    :param occupancy: (n,) ndarray of bool for occupied / not occupied
    :param full_pc: (m, 3) ndarray of full point cloud sampled from object surface
    :param check_z: correct values for ground plane in computation
    :return: (n,) sdf values np.float16
    """
    # use kd tree of the full point cloud to determine the distance to the nearest neighbour
    kd_tree = cKDTree(full_pc)
    min_distances, _ = kd_tree.query(points, k=1, eps=0, p=2, workers=4)
    signs = 1 * (~occupancy) - 1 * occupancy  # if occupied==True, sign will be negative
    signed_distances = min_distances * signs

    if check_z:
        # the ground plane is at z=0, i.e. we can directly take points' z-values rather than their distance to
        # sampled points from the ground plane (less accurate)
        z_values = points[:, 2]

        # where both signed distance and z_values are negative, we want to use the one closer to zero (the larger)
        # for all remaining values, we always want to use the smaller value as it takes precedence
        signed_distances = np.where((z_values < 0) & (signed_distances < 0),
                                    np.maximum(z_values, signed_distances),
                                    np.minimum(z_values, signed_distances))

    return signed_distances
