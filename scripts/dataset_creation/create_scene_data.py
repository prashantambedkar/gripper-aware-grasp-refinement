import os

import numpy as np
import burg_toolkit as burg
import open3d as o3d

from gag_refine.dataset.occupancy import sample_points, get_occupancy_map
from gag_refine.dataset.sdf import compute_sdf_values
from gag_refine.dataset import load_object_library, get_scenes_from_split
from gag_refine.dataset.creation import create_full_point_cloud_data, create_annotated_query_points


def create_scenes(n_scenes, n_objects, area, base_dir, split, start_idx=0, idx_offset=0):
    """
    Creates randomly sampled scenes using the BURG toolkit.
    Elements in n_scenes and n_objects are pairs, i.e. will produce n_s scenes with n_o objects for each element in
    those lists/arrays.

    Unfortunately there is some memory leak or so, I get a SIGKILL occasionally. So that's why we have a start_idx
    that allows to continue in the middle after a kill.
    Setting start_idx will skip the corresponding first entries in n_objects and n_scenes. In contrast, setting
    idx_offset will simply start counting from that offset, but create all the scenes.

    :param n_scenes: list/array of numbers of scenes to sample (corresponds to n_objects list)
    :param n_objects: list/array of numbers of how many objects should be in those scenes
    :param area: 2-tuple with size of ground area
    :param base_dir: base directory of the dataset, will put scenes in ./<split>/scenes/<scene_idx>/scene.yaml
    :param split: 'train' or 'test'
    :param start_idx: if you want to continue at a certain point
    :param idx_offset: if you want to add an arbitrary number to offset the output scene indices
    :return: nothing, saves the scenes to files
    """
    n_scenes = np.array(n_scenes)
    n_objects = np.array(n_objects)
    assert n_scenes.ndim == n_objects.ndim
    if n_scenes.ndim == 0:
        n_scenes = np.array([n_scenes])
        n_objects = np.array([n_objects])
    else:
        assert n_scenes.ndim == 1

    # check start_idx
    skipped = 0
    i = 0
    # set all n_scenes[i] that we can skip entirely to 0
    while start_idx >= n_scenes[i] + skipped:
        skipped += n_scenes[i]
        n_scenes[i] = 0
        i += 1
    # now only correct the amount of the remaining one
    n_scenes[i] -= (start_idx - skipped)
    scene_fn_idx = start_idx + idx_offset

    object_library = load_object_library(os.path.join(base_dir, '..'), split)
    for n_s, n_o in zip(n_scenes, n_objects):
        for i in range(n_s):
            while True:
                # sample_scene might return scene with fewer objects if not all of them could be placed, just repeat
                scene = burg.sampling.sample_scene(object_library, ground_area=area, instances_per_scene=n_o,
                                                   instances_per_object=1)
                if len(scene.objects) == n_o:
                    scene_dir = os.path.join(base_dir, f'{scene_fn_idx:04d}')
                    burg.io.make_sure_directory_exists(scene_dir)
                    fn = os.path.join(scene_dir, f'scene.yaml')
                    scene.to_yaml(fn, object_library)
                    print(f'saved {fn}')
                    scene_fn_idx += 1
                    # burg.visualization.show_geometries([scene])
                    break
                del scene


def create_split_files(base_dir, splits, n_scenes_per_split):
    idx = 0
    for split, n_scenes in zip(splits, n_scenes_per_split):
        split_lst_fn = os.path.join(base_dir, f'{split}.lst')
        with open(split_lst_fn, 'w') as f:
            for i in range(n_scenes):
                f.write(f'{idx:04d}\n')
                idx += 1


def create_scene_data(base_dir, n_objects, n_scenes_train, n_scenes_test):
    # ***** scene sampling *****
    # make scenes fit on A3 paper
    # make them square, so that all the feature planes of ConvONet are be same size
    scene_edge_len = 0.297

    create_scenes(n_scenes=n_scenes_train, n_objects=n_objects, area=(scene_edge_len, scene_edge_len),
                  base_dir=base_dir, split='train', start_idx=0)

    create_scenes(n_scenes=n_scenes_test, n_objects=n_objects, area=(scene_edge_len, scene_edge_len),
                  base_dir=base_dir, split='test', idx_offset=np.sum(n_scenes_train))

    # create train.lst and test.lst files
    create_split_files(base_dir, ['train', 'test'], [np.sum(n_scenes_train), np.sum(n_scenes_test)])


def crop_point_cloud(pc, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
    delete_mask = np.zeros(len(pc), dtype=bool)
    if x_min is not None:
        delete_mask |= pc[:, 0] < x_min
    if x_max is not None:
        delete_mask |= pc[:, 0] > x_max
    if y_min is not None:
        delete_mask |= pc[:, 1] < y_min
    if y_max is not None:
        delete_mask |= pc[:, 1] > y_max
    if z_min is not None:
        delete_mask |= pc[:, 2] < z_min
    if z_max is not None:
        delete_mask |= pc[:, 2] > z_max

    return pc[~delete_mask]


def create_partial_point_cloud_data(base_dir):
    # i.e. point cloud will be generated from a single_view depth images
    splits = ['train', 'test']
    points_dtype = np.float16
    combined_pc_cam_indices = [9, 10, 13, 14]  # these poses will be used for the combined pc
    visualise = False

    render_engine = burg.render.PyBulletRenderEngine()
    camera = burg.render.Camera.create_kinect_like()

    for split in splits:
        lib = load_object_library(os.path.join(base_dir, '..'), split)
        all_scenes = get_scenes_from_split(base_dir, split)
        for scene_dir_id in all_scenes:
            scene_dir = os.path.join(base_dir, scene_dir_id)
            scene, _, _ = burg.Scene.from_yaml(os.path.join(scene_dir, 'scene.yaml'), lib)
            render_engine.setup_scene(scene, camera, with_plane=True)

            # there is no randomness in the view point - the focus point is always the same. perhaps augment?
            w_x, w_y = scene.ground_area
            pose_generator = burg.render.CameraPoseGenerator(
                cam_distance_min=1.0, cam_distance_max=1.0, upper_hemisphere=True, lower_hemisphere=False,
                center_point=[w_x/2, w_y/2, 0.1],
            )
            cam_poses = pose_generator.icosphere(subdivisions=1, in_plane_rotations=1, scales=1, random_distances=False)
            # visualise camera poses
            if visualise:
                print('num poses', cam_poses.shape)
                frames = []
                for pose in cam_poses:
                    frames.append(burg.visualization.create_frame(size=0.1, pose=pose))
                burg.visualization.show_geometries([*frames, scene])

            fn_name = 'partial_point_cloud'
            out_dir = os.path.join(scene_dir, fn_name)
            burg.io.make_sure_directory_exists(out_dir)
            combine_pcs = []
            for i, pose in enumerate(cam_poses):
                _, depth_img = render_engine.render(pose)
                pc = camera.point_cloud_from_depth(depth_img, pose)
                pc = crop_point_cloud(pc, x_min=0, x_max=w_x, y_min=0, y_max=w_y)
                out_dict = {
                    'points': pc.astype(points_dtype),
                    'normals': np.zeros(pc.shape).astype(points_dtype)  # fake stuff... not really needed
                }
                np.savez(os.path.join(out_dir, f'{fn_name}_{i:02}.npz'), **out_dict)
                if i in combined_pc_cam_indices:
                    combine_pcs.append(pc)

                if visualise:
                    print(pc.shape)
                    pcs = [pc]
                    # also visualise projection to the feature planes
                    for d in range(3):
                        projected_pc = np.copy(pc)
                        projected_pc[:, d] = 0
                        pcs.append(projected_pc)
                    burg.visualization.show_geometries([scene, *pcs])

            # save a merged point cloud in one npz file
            combined_pc = np.concatenate(combine_pcs, axis=0)
            out_dict = {
                'points': combined_pc.astype(points_dtype),
                'normals': np.zeros(combined_pc.shape).astype(points_dtype)  # fake stuff... not really needed
            }
            np.savez(os.path.join(scene_dir, f'pointcloud_4views.npz'), **out_dict)
            print(f'saved all for {split}-{scene_dir_id}')
            if visualise:
                print(combined_pc.shape)
                burg.visualization.show_geometries([scene, combined_pc])


def show_scene(base_dir, scene_id=0, split='train'):
    # show the scene from BURG toolkit
    # show the point clouds
    # show the occupancy points

    lib = load_object_library(os.path.join(base_dir, '..'), split)
    scene_dir = os.path.join(base_dir, f'{scene_id:04d}')
    scene, _, _ = burg.Scene.from_yaml(os.path.join(scene_dir, 'scene.yaml'), lib)

    occupancy_data = dict(np.load(os.path.join(scene_dir, 'points_iou.npz')))
    occ_labels = np.unpackbits(occupancy_data['occupancies']).astype(bool)
    occ_points = [occupancy_data['points'][occ_labels], occupancy_data['points'][~occ_labels]]

    # pc_4views = dict(np.load(os.path.join(scene_dir, 'pointcloud_4views.npz')))['points']

    burg.visualization.show_geometries([scene, *occ_points])
    print('all stuff drawn')


def main():
    base_dir = '/home/rudorfem/datasets/gag/scenes/'
    base_dir = '/home/martin/dev/gripper-aware-grasp-refinement/data/gag/scenes/'
    base_dir = 'drive/MyDrive/dev/gripper-aware-grasp-refinement/data/gag-refine/scenes/'

    n_objects = [3, 4, 5]
    n_scenes_train = [200, 200, 200]
    n_scenes_train = [20, 20, 10]
    n_scenes_test = [20, 20, 20]
    n_scenes_test = [2, 2, 1]
    # create_scene_data(base_dir, n_objects, n_scenes_train, n_scenes_test)

    lib = load_object_library(os.path.join(base_dir, '..'))
    create_full_point_cloud_data(base_dir, lib)
    create_annotated_query_points(base_dir, lib, with_sdf=True, n_points_whole_scene=100000,
                                  n_points_per_object_bb=30000, ground_plane=True)
    # create_partial_point_cloud_data(base_dir)

    # burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.IPYNB_VIEWER)
    show_scene(base_dir)


if __name__ == '__main__':
    main()

