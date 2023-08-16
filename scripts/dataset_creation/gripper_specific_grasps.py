import os

import numpy as np
import burg_toolkit as burg

from gag_refine.dataset import load_object_library, get_scenes_from_split
from gag_refine.utils.transform import transform_points
from gag_refine.utils.vis import show_frame_axes, spheres


def create_grasp_poses_2f(dataset_dir):
    """
    Goes through all the test scenes and creates some exemplary grasp poses based on the contact points.
    Only works for two-fingered grasps at the moment.
    Note that we just assume some approach direction (which is roughly from the top).
    Grasp pose convention aligns with BURG toolkit: origin is in the middle between contact points, x-axis points
    towards a contact, z-axis points towards gripper base (i.e. from where the gripper approaches).
    """
    scenes_dir = os.path.join(dataset_dir, 'scenes')
    grasps_dir = os.path.join(dataset_dir, 'grasps')
    poses_dir = os.path.join(dataset_dir, 'initial_grasp_poses')

    # go through all the test scenes
    lib = load_object_library(dataset_dir, split='test')
    scene_idcs = get_scenes_from_split(scenes_dir, split='test')
    for scene_idx in scene_idcs:
        # load scene
        scene_fn = os.path.join(scenes_dir, scene_idx, 'scene.yaml')
        scene, _, _ = burg.Scene.from_yaml(scene_fn, object_library=lib)

        for obj_instance in scene.objects:
            identifier = obj_instance.object_type.identifier
            n_fingers = 2
            obj_grasp_dir = os.path.join(grasps_dir, identifier, f'{n_fingers}')
            pose = obj_instance.pose

            # we have multiple grasp files per obj in which different types of grasps (qualities) are stored
            # gather grasps from each file
            contact_points_list = []
            scores_list = []
            files = os.listdir(obj_grasp_dir)
            for file in files:
                f = os.path.join(obj_grasp_dir, file)
                data = np.load(f)

                contact_points = data['contact_points']
                contact_points = transform_points(pose, contact_points)  # object's pose in the scene
                contact_points_list.append(contact_points)

                scores = data['scores']
                scores_list.append(scores)

            contact_points = np.concatenate(contact_points_list, axis=0)
            scores = np.concatenate(scores_list, axis=0)

            # from the contact points, we can derive the origin as well as the x-axes
            c01 = contact_points[:, 1, :] - contact_points[:, 0, :]
            origins = contact_points[:, 0, :] + c01/2

            x_axes = c01
            # check whether some x_axes are 0. to avoid division by zero, set them to some arbitrary x-axis
            x_axes[np.all(x_axes == 0, axis=-1)] = [1, 0, 0]
            x_axes = x_axes/np.linalg.norm(x_axes, axis=-1)[:, None]

            # now we want to find an approach vector that is orthogonal to the x_axes but also to the xy plane
            xy_plane = np.array([0, 1, 0])
            z_axes = -np.cross(x_axes, xy_plane)
            z_axes = z_axes/np.linalg.norm(z_axes, axis=-1)[:, None]

            # might need to flip z_axes if it is pointing downwards
            downwards_idcs = z_axes[:, 2] < 0
            z_axes[downwards_idcs] *= -1

            # y_axis follows
            y_axes = -np.cross(x_axes, z_axes)
            y_axes = y_axes/np.linalg.norm(y_axes, axis=-1)[:, None]

            grasp_poses = np.broadcast_to(np.eye(4), (contact_points.shape[0], 4, 4)).copy()  # identities
            grasp_poses[:, :3, 0] = x_axes
            grasp_poses[:, :3, 1] = y_axes
            grasp_poses[:, :3, 2] = z_axes
            grasp_poses[:, :3, 3] = origins

            poses_scene_dir = os.path.join(poses_dir, scene_idx, obj_instance.object_type.identifier)
            burg.io.make_sure_directory_exists(poses_scene_dir)
            npz_fn = os.path.join(poses_scene_dir, f'{n_fingers}.npz')
            data_dict = {
                'poses': grasp_poses,
                'contact_points': contact_points,
                'scores': scores
            }
            np.savez(npz_fn, **data_dict)
            print('saved file:', npz_fn)

            # show individual grasps to see they make sense
            # for i in range(grasp_poses.shape[0]):
            #     axes_vis = show_frame_axes(grasp_poses[i], origin=origins[i], show=False)
            #     burg.visualization.show_geometries([scene, *axes_vis, *spheres(contact_points[i])])

            # show a bunch of grasps to see they make sense
            # gs = burg.GraspSet.from_poses(grasp_poses)
            # gs.scores = np.clip(scores, -1, 1)
            # burg.visualization.show_grasp_set([scene], gs, n=100, score_color_func=lambda s: [1-(s+1)/2, (s+1)/2, 0],
            #                                   gripper=burg.gripper.TwoFingerGripperVisualisation())


def check_collisions_for_gripper(dataset_dir, gripper_type):
    """
    loads the initial grasp poses with a specific gripper model
    checks whether there are collisions, saves in corresponding npz file
    """
    scenes_dir = os.path.join(dataset_dir, 'scenes')
    collisions_dir = os.path.join(dataset_dir, 'gripper_specific_collisions')
    poses_dir = os.path.join(dataset_dir, 'initial_grasp_poses')

    # go through all the test scenes
    lib = load_object_library(dataset_dir, split='test')
    scene_idcs = get_scenes_from_split(scenes_dir, split='test')
    for scene_idx in scene_idcs:
        # load scene
        scene_fn = os.path.join(scenes_dir, scene_idx, 'scene.yaml')
        scene, _, _ = burg.Scene.from_yaml(scene_fn, object_library=lib)

        # prepare sim
        sim = burg.sim.GraspSimulator(scene, plane_and_gravity=True)
        gripper = gripper_type(sim)
        gripper.load(np.eye(4))
        gripper.set_open_scale(1.0)

        for obj_instance in scene.objects:
            identifier = obj_instance.object_type.identifier
            n_fingers = 2
            grasp_poses_fn = os.path.join(poses_dir, scene_idx, identifier, f'{n_fingers}.npz')
            data = np.load(grasp_poses_fn)
            grasp_poses = data['poses']
            collisions = np.zeros(grasp_poses.shape[0], dtype=bool)

            # this is quite slow but i was too lazy to parallelize it
            # todo: use parallelised implementation from gag_refine.utils.gripper_collisions
            print('todo: use parallelised implementation from gag_refine.utils.gripper_collision')
            for i, grasp_pose in enumerate(grasp_poses):
                gripper.reset_pose(grasp_pose)
                result = sim.check_collisions(gripper, obj_instance)
                collisions[i] = result != burg.sim.GraspScores.SUCCESS

            # show a bunch of grasps to see they make sense
            # gs = burg.GraspSet.from_poses(grasp_poses)
            # gs.scores = collisions.astype(float)
            # burg.visualization.show_grasp_set([scene], gs, n=100, score_color_func=lambda s: [s, (1-s), 0],
            #                                   gripper=burg.gripper.TwoFingerGripperVisualisation())

            collisions_obj_dir = os.path.join(collisions_dir, gripper_type.__name__, scene_idx, identifier)
            burg.io.make_sure_directory_exists(collisions_obj_dir)
            collisions_fn = os.path.join(collisions_obj_dir, f'{n_fingers}.npz')
            np.savez(collisions_fn, **{'collisions': collisions})

            print(f'{np.mean(collisions):.4f}. saved', collisions_fn)


if __name__ == '__main__':
    path = 'data/gag/'
    # create_grasp_poses_2f(path)
    check_collisions_for_gripper(path, burg.gripper.Robotiq2F85)
    check_collisions_for_gripper(path, burg.gripper.Franka)
    check_collisions_for_gripper(path, burg.gripper.Robotiq2F140)
