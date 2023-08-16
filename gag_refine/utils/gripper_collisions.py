from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import burg_toolkit as burg


def check_collisions_helper(zipped_poses_and_configs, scene, gripper_type):
    """ helper function that performs checking sequentially (slow) """
    grasp_poses, gripper_configs = zipped_poses_and_configs
    n_grasps = grasp_poses.shape[0]

    # prepare sim
    sim = burg.sim.GraspSimulator(scene, plane_and_gravity=True)
    gripper = gripper_type(sim)
    gripper.load(np.eye(4))
    gripper.set_open_scale(1.0)

    collisions = np.zeros(grasp_poses.shape[0], dtype=bool)

    # we need to declare one object as target object, as the simulator will give detailed information about which
    # object we collide with. however, as we are only interested in binary value, we can just declare any object to
    # be the target object
    target_obj = scene.objects[0]

    for i in range(n_grasps):
        gripper.reset_pose(grasp_poses[i])
        gripper.set_open_scale(gripper_configs[i])
        result = sim.check_collisions(gripper, target_obj)
        collisions[i] = result != burg.sim.GraspScores.SUCCESS

    sim.dismiss()
    return collisions


def chunkify_and_zip_grasps(grasp_poses, gripper_configs, chunk_size=10):
    n_grasps = grasp_poses.shape[0]
    zipped_grasps = []
    n_chunks = int(np.ceil(n_grasps/chunk_size))
    start_idx = 0
    for i in range(n_chunks):
        end_idx = start_idx + chunk_size
        chunk = (grasp_poses[start_idx:end_idx], gripper_configs[start_idx:end_idx])
        zipped_grasps.append(chunk)
        start_idx = end_idx

    return zipped_grasps


def check_collisions(scene, gripper_type, grasp_poses, gripper_configs):
    """
    will load the given gripper model in the given grasp poses & configurations to check for collisions

    Args:
        scene (burg.Scene): a scene
        gripper_type (class): burg.gripper Class (not an instance)
        grasp_poses (ndarray): (N, 4, 4) grasp poses
        gripper_configs (ndarray): (N, C) configs with C being the DOF of the gripper
    """
    assert len(grasp_poses.shape) == 3, 'shape of grasp poses should be (N, 4, 4)'
    n_grasps = grasp_poses.shape[0]
    assert gripper_configs.shape[0] == n_grasps

    grasps = chunkify_and_zip_grasps(grasp_poses, gripper_configs)
    pool = ProcessPoolExecutor()
    collisions = list(pool.map(
        partial(check_collisions_helper, scene=scene, gripper_type=gripper_type), grasps)
    )

    collisions = np.concatenate(collisions, axis=0)
    return collisions
