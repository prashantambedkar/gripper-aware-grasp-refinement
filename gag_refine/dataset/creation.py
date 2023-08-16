import os

import numpy as np
import trimesh
import burg_toolkit as burg

from . import get_scenes_from_split
from .occupancy import sample_points, get_occupancy_map
from .sdf import compute_sdf_values


def create_full_point_cloud_data(base_dir, object_library, splits=None, ground_factor=0.2, n_points=100000,
                                 with_semantics=False, with_ply=True):
    """
    This method will open the existing scenes in the dataset and will sample point clouds from the object surfaces.
    It assumes a directory that contains a train.lst/test.lst (or other specified splits) and will read the subfolders
    from those lists. In each subfolder, it assumes a scene.yaml file that can be opened with the given object library.
    In the same folder, it will generate the point cloud and save it as full_point_cloud.npz. The point cloud will
    contain points as well as normals (and possibly semantics). We can also create an additional ply file in the same
    folder for convenience.

    Args:
        base_dir: Base directory that contains the split.lst files
        object_library: Object library that can be used to load the scenes
        splits: name of the .lst files, if None given then it defaults to using both train and test
        ground_factor: Determines the proportion of the points we sample that should be from the ground plane
        n_points: The number of points to sample in total
        with_semantics: Whether to store semantics in the npz file as well (will store separate id for every object)
        with_ply: Whether to produce a .ply file

    Returns:
        Nothing
    """
    # i.e. point cloud data will be sampled from objects' surfaces as in ConvONets
    print(f'creating full point cloud data for dataset in {base_dir}')
    if splits is None:
        splits = ['train', 'test']
        print(f'given split was None, assuming the following splits: {splits}')

    points_dtype = np.float16
    semantics_dtype = np.int8

    for split in splits:
        all_scenes = get_scenes_from_split(base_dir, split)
        for scene_dir_id in all_scenes:
            scene_dir = os.path.join(base_dir, scene_dir_id)
            print('scene_dir:',scene_dir)
            scene, _, _ = burg.Scene.from_yaml(os.path.join(scene_dir, 'scene.yaml'), object_library)

            meshes = scene.get_mesh_list(with_plane=False, with_bg_objects=False, as_trimesh=True)
            # determine how many points to sample by surface area of meshes
            ratios = np.zeros(len(meshes)+1)
            ratios[0] = scene.ground_area[0] * scene.ground_area[1] * ground_factor
            ratios[1:] = [mesh.area for mesh in meshes]
            ratios /= np.sum(ratios)
            n_points_per_obj = [int(ratio * n_points) for ratio in ratios]

            # perform the sampling
            all_points, all_normals, all_semantics = [], [], []
            for i in range(len(ratios)):
                if i == 0:
                    # ground plane
                    points = np.zeros((n_points_per_obj[0], 3))
                    points[:, :2] = sample_points(len(points), [0, 0], [scene.ground_area[0], scene.ground_area[1]])
                    normals = np.zeros(points.shape)
                    normals[:, 2] = 1  # all are facing up
                else:
                    # actual mesh
                    mesh = meshes[i-1]  # because 0 was ground plane
                    points, face_idx = mesh.sample(n_points_per_obj[i], return_index=True)
                    normals = mesh.face_normals[face_idx]

                all_points.append(points)
                all_normals.append(normals)

                if with_semantics:
                    semantics = np.full(len(points), i)  # ConvONet uses actual class correspondence instead
                    all_semantics.append(semantics)

            out_dict = {
                'points': np.concatenate(all_points, axis=0).astype(points_dtype),
                'normals': np.concatenate(all_normals, axis=0).astype(points_dtype),
            }
            if with_semantics:
                out_dict['semantics'] = np.concatenate(all_semantics, axis=0).astype(semantics_dtype)

            fn = os.path.join(scene_dir, 'full_point_cloud.npz')
            np.savez(fn, **out_dict)

            if with_ply:
                # also save the point cloud in ply format, so we can easily view it with meshlab
                trimesh.Trimesh(vertices=out_dict['points'], process=False).export(
                    os.path.join(scene_dir, 'pointcloud.ply'))

            print(f'saved {split}-{scene_dir_id}')


def create_annotated_query_points(base_dir, object_library, splits=None, with_sdf=True, n_points_whole_scene=100000,
                                  n_points_per_object_bb=0, with_semantics=False, ground_plane=False):
    """
    Creates the query points and their occupancy/sdf annotations. Assumes that base_dir contains the split.lst files
    and each subfolder contains a scene.yaml that can be opened with the given object_library. The subfolder also
    must contain a full_point_cloud.npz file, which can be created using `create_full_point_cloud_data()` method.
    This will create a file points_iou.npz in the subdirectories, which contains points, occupancies, and optionally
    semantics and sdf.

    Args:
        base_dir: Base directory that contains the split.lst files
        object_library: Object library that can be used to load the scenes
        splits: name of the .lst files, if None given then it defaults to using both train and test
        with_sdf: If True, will also annotate SDF and not just binary occupancy
        n_points_whole_scene: number of points to sample for the whole scene
        n_points_per_object_bb: number of additional points to sample in the bounding box of each object, e.g. 30k
        with_semantics: whether to save semantic information as well, i.e. an object id
        ground_plane: whether the scene has a ground plane at z=0 (important for correct calculation of SDF)

    Returns:
        Nothing
    """
    scene_padding = 0.012  # sample points with some padding, set padding in ConvONet to 0.08
    object_padding = 0.03  # some more padding around the objects, but we limit it to the scene padding

    points_dtype = np.float16
    semantics_dtype = np.int8
    sdf_dtype = np.float16

    print(f'creating annotations for dataset in {base_dir}')
    if splits is None:
        splits = ['train', 'test']
        print(f'given split was None, assuming the following splits: {splits}')

    for split in splits:
        all_scenes = get_scenes_from_split(base_dir, split)
        for scene_dir_id in all_scenes:
            scene_dir = os.path.join(base_dir, scene_dir_id)
            scene, _, _ = burg.Scene.from_yaml(os.path.join(scene_dir, 'scene.yaml'), object_library)
            assert scene.ground_area[0] == scene.ground_area[1], f'{split}-{scene_dir_id} ground area is not square'
            scene_edge_len = scene.ground_area[0]

            all_points = []
            all_occupancy_maps = []

            # 1st step: sample the whole scene randomly
            lower_bounds = [-scene_padding]*3
            upper_bounds = [scene_edge_len + scene_padding]*3
            points = sample_points(n_points_whole_scene, lower_bounds, upper_bounds)
            occupancy_map = get_occupancy_map(points, scene)
            all_points.append(points)
            all_occupancy_maps.append(occupancy_map)

            # 2nd step: sample more specifically in bounding box of each object
            if n_points_per_object_bb is not None and n_points_per_object_bb > 0:
                for obj_instance in scene.objects:
                    # get bounding box and sample points within
                    mesh = obj_instance.get_mesh()
                    lower_bounds = mesh.get_min_bound() - object_padding
                    upper_bounds = mesh.get_max_bound() + object_padding
                    points = sample_points(n_points_per_object_bb, lower_bounds, upper_bounds)
                    points = points[points[:, 2] > -scene_padding]  # filter out those that are too low in z
                    occupancy_map = get_occupancy_map(points, scene)
                    all_points.append(points)
                    all_occupancy_maps.append(occupancy_map)

            # 3rd step: save results
            points = np.concatenate(all_points, axis=0)
            occupancy_map = np.concatenate(all_occupancy_maps, axis=0)

            # make sure the number of points is dividable by 8, so numpy's packbits/unpackbits works nicely
            # simply discard the surplus elements
            max_idx = points.shape[0] - (points.shape[0] % 8)
            points = points[:max_idx]
            occupancy_map = occupancy_map[:max_idx]
            occ_labels = (occupancy_map >= 0).astype(bool)

            out_dict = {
                'points': points.astype(points_dtype),
                'occupancies': np.packbits(occ_labels),
                'z_scale': 0,  # not sure if required, but it's present in ConvONets
            }

            if with_semantics:
                out_dict['semantics'] = occupancy_map.astype(semantics_dtype)

            if with_sdf:
                full_point_cloud = dict(np.load(os.path.join(scene_dir, 'full_point_cloud.npz')))['points']
                sdf_values = compute_sdf_values(points, occ_labels, full_point_cloud, check_z=ground_plane)
                out_dict['sdf'] = sdf_values.astype(sdf_dtype)

            np.savez(os.path.join(scene_dir, 'points_iou.npz'), **out_dict)
            print(f'saved annotations for {split}-{scene_dir_id}')
