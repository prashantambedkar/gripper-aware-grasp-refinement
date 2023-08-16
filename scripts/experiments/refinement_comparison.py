"""
idea of this script is to find out if the iterative refinement actually improves a given grasp set.
as initial grasp set, we choose:
- some sampled grasp set G from test set, using the GT scores
compare to:
- [baseline-Franka] grasps from G, using GT collisions with Franka gripper
- [baseline-Robotiq2F85] grasps from G, using GT collisions with Robotiq-2F85 gripper
- [refined-Franka] using G as initial set, refinement with Franka gripper decoder, then compute FC and collisions
"""

import os
import argparse
import pprint

import torch
import numpy as np
import burg_toolkit as burg
import trimesh

import gag_refine.dataset
from convonets.src import config
from gag_refine.gripper_decoder import load_gripper_decoder
from gag_refine.refinement import PoseRefinement, ConfigRefinement, GraspRefinement
from gag_refine.utils.gripper_collisions import check_collisions
from gag_refine.utils.vis import show_frame_axes, sphere
from gag_refine.dataset.ferrari_canny import FerrariCanny


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine the gripper pose and configuration in a given scene'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')
    parser.add_argument('gripper_decoder_config', type=str, help='path to gripper decoder config file')

    return parser.parse_args()


def load_grasp_candidates(scene_data, random_seed=42, n_per_obj=200, n_fingers=2):
    scene_idx = scene_data['meta.scene_idx'][0]
    dataset_dir = scene_data['meta.dataset_dir'][0]
    scene_grasps_dir = os.path.join(dataset_dir, 'initial_grasp_poses', scene_idx)

    rng = np.random.default_rng(seed=random_seed)
    indices = None

    grasp_candidate_data = {}

    obj_dirs = os.listdir(scene_grasps_dir)
    for obj_dir in obj_dirs:
        grasp_data = np.load(os.path.join(scene_grasps_dir, obj_dir, f'{n_fingers}.npz'))
        poses = grasp_data['poses']
        scores = grasp_data['scores']

        if indices is None:
            # random sample only the first time, then use the same indices
            indices = rng.choice(len(poses), n_per_obj, replace=False)

        grasp_candidate_data[obj_dir] = {
            'poses': poses[indices],
            'scores': scores[indices],
            'indices': indices,
        }

    return grasp_candidate_data


def append_gripper_collisions(gripper_type, scene_data, grasp_candidate_data, n_fingers=2):
    scene_idx = scene_data['meta.scene_idx'][0]
    dataset_dir = scene_data['meta.dataset_dir'][0]
    collisions_dir = os.path.join(dataset_dir, 'gripper_specific_collisions', gripper_type.__name__, scene_idx)

    for obj, grasp_dict in grasp_candidate_data.items():
        collisions = np.load(os.path.join(collisions_dir, obj, f'{n_fingers}.npz'))['collisions']
        collisions = collisions[grasp_dict['indices']]

        if 'collisions' not in grasp_dict.keys():
            grasp_dict['collisions'] = {}
        grasp_dict['collisions'][gripper_type.__name__] = collisions


def summarise_scene_and_update_stats(grasp_candidates_data, summary_dict=None, show=False):
    """ prints a summary of the data, collisions and refinements """
    if show:
        print(f'objects: {list(grasp_candidates_data.keys())}')
    for key, grasp_dict in grasp_candidates_data.items():
        scores = grasp_dict['scores']
        stable_grasps = scores > 0
        if summary_dict is not None:
            summary_dict['stable candidates'].append(np.sum(stable_grasps))
            summary_dict['total candidates'].append(len(scores))

        if show:
            print(key)
            print(f'\t{np.sum(stable_grasps)}/{len(scores)} grasp candidates are stable grasps')
        if 'collisions' in grasp_dict.keys():
            for gripper_name, collisions in grasp_dict['collisions'].items():
                success = np.sum(stable_grasps & ~collisions)
                stable_but_colliding = np.sum(stable_grasps & collisions)
                unstable_non_colliding = np.sum(~stable_grasps & ~collisions)
                unstable_colliding = np.sum(~stable_grasps & collisions)
                if show:
                    print(f'\tgripper: {gripper_name}')
                    print(f'\t\tsuccessful: {success}')
                    print(f'\t\tstable but colliding: {stable_but_colliding}')
                    print(f'\t\tunstable non colliding: {unstable_non_colliding}')
                    print(f'\t\tunstable and colliding: {unstable_colliding}')
                if summary_dict is not None:
                    summary_dict[gripper_name]['successful'].append(success)
                    summary_dict[gripper_name]['stable but colliding'].append(stable_but_colliding)
                    summary_dict[gripper_name]['unstable non colliding'].append(unstable_non_colliding)
                    summary_dict[gripper_name]['unstable and colliding'].append(unstable_colliding)

    return summary_dict


def update_stats_with_refined_grasps(name, summary_dict, collisions, scores):
    stable_grasps = scores > 0
    success = np.sum(stable_grasps & ~collisions)
    stable_but_colliding = np.sum(stable_grasps & collisions)
    unstable_non_colliding = np.sum(~stable_grasps & ~collisions)
    unstable_colliding = np.sum(~stable_grasps & collisions)
    summary_dict[name]['successful'].append(success)
    summary_dict[name]['stable but colliding'].append(stable_but_colliding)
    summary_dict[name]['unstable non colliding'].append(unstable_non_colliding)
    summary_dict[name]['unstable and colliding'].append(unstable_colliding)

    return summary_dict


def sum_up_lists(eval_summary):
    for key, item in eval_summary.items():
        if isinstance(item, list):
            eval_summary[key] = np.sum(item)
        elif isinstance(item, dict):
            eval_summary[key] = sum_up_lists(item)  # apply recursively

    return eval_summary


def get_object_instance_by_name(scene, obj_name):
    # assumes each object type can only occur once in a scene (which is true for our dataset but not generally)
    for instance in scene.objects:
        if instance.object_type.identifier == obj_name:
            return instance

    raise ValueError(f'could not find {obj_name} in scene with objects'
                     f' {[i.object_type.identifier for i in scene.objects]}')


def find_contacts_for_2finger_grasps(mesh, grasp_poses):
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    has_contact_points = np.ones(len(grasp_poses), dtype=bool)
    contact_points = np.zeros((len(grasp_poses), 2, 3))
    contact_normals = np.zeros((len(grasp_poses), 2, 3))

    for i in range(len(grasp_poses)):
        grasp_center = grasp_poses[i, :3, 3].reshape(1, 3)  # grasp center
        x_axis = grasp_poses[i, :3, 0].reshape(1, 3)  # grasp x-axis

        # find first contact point
        first_contact_candidates, _, index_tri1 = intersector.intersects_location(grasp_center, x_axis)
        if len(first_contact_candidates) == 0:
            has_contact_points[i] = False
            continue

        # find second contact point
        second_contact_candidates, _, index_tri2 = intersector.intersects_location(grasp_center, -x_axis)
        if len(second_contact_candidates) == 0:
            has_contact_points[i] = False
            continue

        # have found candidates for both contacts
        # heuristics: if multiple hits, we choose the contacts that are farthest away from the grasp center
        # this may be an issue with hollow structures (cups) that are grasped along the rim, but should be ok
        # for most of the objects
        contact1_idx = np.argmax(np.linalg.norm(first_contact_candidates - grasp_center, axis=-1))
        contact2_idx = np.argmax(np.linalg.norm(second_contact_candidates - grasp_center, axis=-1))

        contact_points[i, 0] = first_contact_candidates[contact1_idx]
        contact_points[i, 1] = second_contact_candidates[contact2_idx]

        # find normals as well now
        triangle_indices = [index_tri1[contact1_idx], index_tri2[contact2_idx]]
        contact_normals[i] = burg.mesh_processing.compute_interpolated_vertex_normals(
            mesh, contact_points[i], triangle_indices)

    return has_contact_points, contact_points, -contact_normals  # inward-facing normals


def main(args):
    # load config files
    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    gripper_decoder_cfg = config.load_config(args.gripper_decoder_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    model = config.get_model_interface(cfg, device, dataset)
    assert hasattr(model, 'eval_grasps'), f'model does not have eval_grasps. backbone: {type(model)}'
    assert hasattr(model, 'eval_sdf'), f'model does not have eval_sdf. backbone: {type(model)}'

    # load gripper decoder
    gripper_decoder = load_gripper_decoder(gripper_decoder_cfg).to(device)
    gripper_decoder.eval()
    # todo: figure out gripper type based on config

    # initialise refinement
    refinement = GraspRefinement(model, gripper_decoder,
                                 pose_refiner=PoseRefinement(device, alpha=1.0),
                                 config_refiner=ConfigRefinement(device, alpha=5.0))

    # define GT grippers
    gt_grippers = [burg.gripper.Franka, burg.gripper.Robotiq2F85]

    # load object library for evaluation
    lib = gag_refine.dataset.load_object_library(cfg['data']['path'], split='test')

    # statistics
    eval_summary = {
        'stable candidates': [],
        'total candidates': [],
    }
    eval_items = [*[g.__name__ for g in gt_grippers], 'Franka_refined']
    for method in eval_items:
        eval_summary[method] = {
            'successful': [],
            'stable but colliding': [],
            'unstable non colliding': [],
            'unstable and colliding': [],
        }

    # loop through the data
    for data in test_loader:
        # data: dict_keys(['points', 'points.occ', 'points.sdf', 'points.volume', 'grasps.contact_points',
        # 'grasps.scores', 'points_iou', 'points_iou.occ', 'points_iou.sdf', 'inputs', 'inputs.normals',
        # 'meta.scene_idx', 'meta.scene_dir', 'meta.scene_fn', 'meta.dataset_dir'])
        scene_idx = data['meta.scene_idx'][0]
        print(f'* scene {scene_idx}')

        # load the initial grasp poses we want to refine
        print(f'loading grasp candidate data...')
        grasp_candidates_data = load_grasp_candidates(data, n_per_obj=100)

        # evaluate for the grippers
        skip_grippers = True
        if not skip_grippers:
            for gripper in gt_grippers:
                print(f'retrieving data for gripper {gripper.__name__}')
                append_gripper_collisions(gripper, data, grasp_candidates_data)

        for obj_name, grasp_dict in grasp_candidates_data.items():
            fn_path = os.path.join(data['meta.dataset_dir'][0], 'experiment_Wed', data['meta.scene_idx'][0])
            fn = os.path.join(fn_path, f'{obj_name}.npz')

            read_from_file_instead_of_apply = True
            if read_from_file_instead_of_apply:
                grasp_data = np.load(fn)
                refined_poses = grasp_data['refined_poses']
                refined_configs = grasp_data['refined_configs']
                had_contact = grasp_data['had_contact']
                was_successful = grasp_data['was_successful']
                print(f'loaded {refined_poses.shape[0]} refined grasps from {fn}')
            else:
                print(f'applying GAG Refine...')
                candidate_poses = grasp_dict['poses']
                candidate_configs = np.ones(len(candidate_poses))  # todo: might want to initialise randomly?
                refined_poses, refined_configs, had_contact, was_successful = refinement.refine_grasps(
                    data, grasp_candidates=candidate_poses, initial_configs=candidate_configs)

                # save the refined poses and refined configs to a file, so we can jumpstart the evaluation tomorrow
                burg.io.make_sure_directory_exists(fn_path)
                save_data = {
                    'refined_poses': refined_poses,
                    'refined_configs': refined_configs,
                    'had_contact': had_contact,
                    'was_successful': was_successful,
                }
                np.savez(fn, **save_data)
                print(f'saved refined grasps to {fn}')

            # compute collisions using ground truth gripper model in refined pose and configuration
            # scene, _, _ = burg.Scene.from_yaml(data['meta.scene_fn'][0], object_library=lib)
            # collisions = check_collisions(scene, burg.gripper.Franka, refined_poses, refined_configs)  # todo temp
            # print(f'have {np.mean(collisions)} collisions')

            # determine the contact points, normals etc. and compute FC scores
            # obj_instance = get_object_instance_by_name(scene, obj_name)
            # mesh = obj_instance.get_trimesh()
            # com = mesh.center_mass
            # is_grasp, contact_points, contact_normals = find_contacts_for_2finger_grasps(mesh, refined_poses)
            # if 'has_contacts' not in eval_summary['Franka_refined'].keys():
            #     eval_summary['Franka_refined']['has_contacts'] = []
            # eval_summary['Franka_refined']['has_contacts'].append(np.sum(is_grasp))
            # scores = np.full(len(is_grasp), fill_value=-np.inf)  # todo temp
            # scores[is_grasp] = FerrariCanny.compute_L1_scores_parallel(
            #     contact_points[is_grasp], contact_normals[is_grasp], com
            # )
            # update_stats_with_refined_grasps('Franka_refined', eval_summary, collisions, scores)
            if 'stopped_early' not in eval_summary.keys():
                eval_summary['stopped_early'] = []
            eval_summary['stopped_early'].append(np.sum(was_successful))
            if 'probably_had_contact' not in eval_summary.keys():
                eval_summary['probably_had_contact'] = []
            eval_summary['probably_had_contact'].append(np.sum(had_contact))

            # todo: for n-fingered grasps we will need to determine the contact points in a clever way
            # we need the gripper-specific contact points after refinement --> get using GT gripper model
            # check how many / which contact points are close to a surface --> subset
            # if subset >= 2, determine which object they belong to --> subset(s)
            # if any object has subset >= 2, find object-specific contact points and normals
            # then compute FC score
            # problem: this is actually more restrictive than the GT scores, since for the GT scores we do not
            # consider the gripper configuration at all for determining the contact points / grasp scores
            # comparison might therefore be unfair and favouring the GT scores
            # therefore, we should apply the same thing to the GT grasps somehow

            # another problem: the pose and configs change independently for some grippers
            # e.g. Robotiq2F85 will change the height of the contact points, while Franka does not.
            # the pre-computed scores might not apply to Robotiq2F85, as the actual contact points will be different
            # but if we use the same strategy as above, the scores will be plummeting as we require the gripper
            # contact points to actually make contact, which by chance is unlikely to happen

            # so because of this stuff, we do sth quick and dirty to get some first quantification... and then we'll
            # think more thoroughly of this.

            visualise = False
            if visualise:
                gs_original = burg.grasp.GraspSet.from_poses(candidate_poses)
                gs_refined = burg.grasp.GraspSet.from_poses(refined_poses)
                scene, _, _ = burg.Scene.from_yaml(data['meta.scene_fn'][0], object_library=lib)
                burg.visualization.show_grasp_set([scene], gs_original,
                                                  gripper=burg.gripper.TwoFingerGripperVisualisation())
                burg.visualization.show_grasp_set([scene], gs_refined,
                                                  gripper=burg.gripper.TwoFingerGripperVisualisation())

        eval_summary = summarise_scene_and_update_stats(grasp_candidates_data, eval_summary)

    eval_summary = sum_up_lists(eval_summary)
    pprint.pprint(eval_summary)


if __name__ == '__main__':
    main(parse_args())
