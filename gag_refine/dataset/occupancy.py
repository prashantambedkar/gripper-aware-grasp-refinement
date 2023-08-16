import numpy as np
from convonets.src.utils.libmesh.inside_mesh import check_mesh_contains as mesh_contains


def get_occupancy_map(points, scene, check_z=True):
    """
    Computes the occupancy for each point. Occupancy maps points to objects:
    -1 means not occupied
    0 point is in ground plane
    1...n_objects correspond to certain object in scene

    :param points: (n, 3) ndarray of points to check
    :param scene: burg.Scene, scene to check occupancy for
    :param check_z: whether to also check occupancy by ground plane
    :return: (n,) occupancy map np.int8
    """
    collision_map = np.zeros(len(points), dtype=int) - 1

    if check_z:
        # points below z=0 are occupied by the ground plane
        collision_map[points[:, 2] < 0] = 0

    # todo: potential speed up by not checking occupied points again (it is already quite fast though)
    for i, obj_instance in enumerate(scene.objects):
        mesh = obj_instance.get_trimesh()
        collision_map[mesh_contains(mesh, points)] = i+1

    return collision_map


def sample_points(n_points, lower_bounds, upper_bounds):
    """
    sample n_points random points within lower_bounds/upper_bounds. bounds determine the dimension of the points.
    :param n_points: int
    :param lower_bounds: tuple/array with lower bounds per dimension
    :param upper_bounds: tuple/array with upper bounds per dimension
    :return: (n_points, d) ndarray
    """
    assert len(lower_bounds) == len(upper_bounds), 'bounds must be same size'
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)

    points = np.random.rand(n_points, len(lower_bounds)) - [0.5]*len(lower_bounds)
    points *= (upper_bounds - lower_bounds)[None, :]
    points += (lower_bounds + (upper_bounds - lower_bounds) / 2)[None, :]
    return points
