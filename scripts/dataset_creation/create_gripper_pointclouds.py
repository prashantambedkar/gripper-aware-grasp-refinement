import os
import argparse

import burg_toolkit as burg
import numpy as np

from gag_refine.dataset.gripper_contacts import supported_grippers, get_gripper_contact_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/gag/')
    parser.add_argument('--dir_name', type=str, default='gripper_point_clouds')
    parser.add_argument('--n_points', type=int, default=2048)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=20)
    return parser.parse_args()


class PoseIndicator:
    def __init__(self, size=0.1):
        self.size = size
        self.mesh = burg.visualization.create_frame(size=size)


def get_camera_poses(show_poses=False):
    pose_generator = burg.render.CameraPoseGenerator(
        cam_distance_min=0.7,       # max distance is irrelevant with only one scale and no randomness
        upper_hemisphere=True,      # we want views from all sides
        lower_hemisphere=True,
        center_point=[0, 0, 0.1]
    )
    poses = pose_generator.icosphere(
        subdivisions=0,         # subdivisions of the icosahedron, setting to 3 or 4 will give loads of poses
        in_plane_rotations=1,   # in-plane rotations and scales don't matter so much for point clouds
        scales=1,
        random_distances=False   # within the cam_distance limits defined above
    )

    if show_poses:
        gs = burg.grasp.GraspSet.from_poses(poses)
        burg.visualization.show_grasp_set([], gs, gripper=PoseIndicator(0.1))
    return poses


def create_gripper_pcs(gripper_type, output_dir, n_samples, camera_poses, n_points=2048, open_scale_range=(0.1, 1.0)):
    # prepare rendering
    render_engine = burg.render.PyBulletRenderEngine()
    camera = burg.render.Camera.create_kinect_like()
    render_engine.setup_scene(
        burg.Scene(),  # create empty scene
        camera,
        with_plane=False
    )
    # add gripper to the scene
    gripper = gripper_type(simulator=render_engine, gripper_size=1.0)
    gripper.load(np.eye(4))
    open_scales = np.random.uniform(open_scale_range[0], open_scale_range[1], n_samples)

    # prepare directories
    save_dir = os.path.join(output_dir, f'{gripper_type.__name__}')
    burg.io.make_sure_directory_exists(save_dir)

    for pc_idx in range(n_samples):
        gripper.set_open_scale(open_scales[pc_idx])

        # create point clouds from multiple view points, then do farthest point sampling
        pcs = []
        for pose in camera_poses:
            _, depth = render_engine.render(pose)
            pc = camera.point_cloud_from_depth(depth, pose)
            pcs.append(pc)

        full_pc = np.concatenate(pcs)  # roughly 100k points
        idcs = burg.sampling.farthest_point_sampling(full_pc, n_points)
        out_dict = {
            'points': full_pc[idcs].astype(np.float16),
            'contact_points': get_gripper_contact_points(gripper).astype(np.float16),
            'config': open_scales[pc_idx],
        }
        fn = os.path.join(save_dir, f'{pc_idx:04d}.npz')
        np.savez(fn, **out_dict)
        print('saved ', fn)


def create_split_lst_files(dataset_dir, gripper_type, n_train, n_test):
    save_dir = os.path.join(dataset_dir, f'{gripper_type.__name__}')
    splits = {
        'train': n_train,
        'test': n_test
    }

    pc_idx = 0
    for split in ['train', 'test']:
        filename = os.path.join(save_dir, f'{split}.lst')
        with open(filename, 'w') as file:
            for i in range(splits[split]):
                file.write(f'{pc_idx:04d}\n')
                pc_idx += 1
        print(f'created {filename}')


def main(args):
    # potentially make a similar data structure like for the scenes?
    # gripper_point_clouds
    #   - <gripper_name>
    #       - 0000 ... 9999.npz: [points, open_scale]
    #       - test.lst, train.lst
    poses = get_camera_poses(show_poses=False)
    gripper_types = supported_grippers
    open_scale_range = (0.1, 1.0)
    n_samples = args.n_train + args.n_test
    for gripper_type in gripper_types:
        print('gripper type:', gripper_type.__name__)
        create_gripper_pcs(gripper_type, os.path.join(args.base_dir, args.dir_name), n_samples=n_samples,
                           n_points=args.n_points, open_scale_range=open_scale_range,  camera_poses=poses)
        create_split_lst_files(os.path.join(args.base_dir, args.dir_name), gripper_type, args.n_train, args.n_test)


if __name__ == '__main__':
    main(parse_args())
