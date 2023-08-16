import argparse
import os
from timeit import default_timer as timer
import pprint

import torch
import numpy as np
import burg_toolkit as burg

from gag_refine.dataset import pre_normalisation_tf
from gag_refine.utils.transform import transform_points
from gag_refine.refinement import PoseRefinement, ConfigRefinement
from gag_refine.gripper_decoder import load_gripper_decoder
from gag_refine.grasp_quality_estimator import load_grasp_quality_net
from convonets.src import config
from convonets.src.common import sdf_to_occ


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine the gripper pose and configuration in a given scene'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')
    parser.add_argument('gripper_decoder_config', type=str, help='path to gripper decoder config file')

    return parser.parse_args()


def load_gripper_data_mockup(idx=0):
    data_path = f'/home/martin/dev/gripper-aware-grasp-refinement/data/gag/gripper_point_clouds/Franka/{idx:04d}.npz'
    print(data_path)
    gripper_data = dict(np.load(data_path))
    points = gripper_data['points']
    open_scale = gripper_data['config']
    print(f'loaded gripper with {points.shape} points, open_scale={open_scale}')

    return torch.from_numpy(points)


def main(args):
    visualise_each_iter = False

    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    gripper_decoder_cfg = config.load_config(args.gripper_decoder_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    model = config.get_model_interface(cfg, device, dataset)
    assert hasattr(model, 'eval_grasps'), f'model does not have eval_grasps. backbone: {type(model)}'
    assert hasattr(model, 'eval_sdf'), f'model does not have eval_sdf. backbone: {type(model)}'

    # load gripper decoder
    gripper_decoder = load_gripper_decoder(gripper_decoder_cfg).to(device)
    gripper_decoder.eval()

    # statistics
    stats = {}
    for it, data in enumerate(test_loader):
        # if it != 0:
        #     continue

        scene_idx = data['idx'].item()
        print('data:', data.keys())
        print(f'{"*"*5} scene {scene_idx}')

        # process scene point cloud to create latent representation
        start_time = timer()
        model.eval_scene_pointcloud(data)
        stats['time to encode scene [s]'] = timer() - start_time

        # okay so what we do here is load the gripper pc in original space (un-normalised, un-transformed)
        # we apply the initial grasp pose
        initial_pose = np.eye(4)
        initial_pose[0, 3] += 0.24
        initial_pose[1, 3] += 0.11
        initial_config = 0.2

        inputs = data['inputs'].squeeze(0).cpu().numpy()  # for vis purposes

        # initialise the tensors we want to optimise!
        pose_refiner = PoseRefinement(device, refine_translation=True, refine_orientation=True, alpha=1.0)
        pose_refiner.set_initial_pose(initial_pose)

        joint_refiner = ConfigRefinement(device, alpha=5.0, config_refinement=True)
        joint_refiner.set_initial_config(initial_config)

        vis_data = {
            'gripper_point_clouds': [],
            'transforms': [],
        }
        for i in range(20):
            print(f'iter {i} --------------')
            iter_stats = {}

            # generate gripper point cloud and contact points using gripper decoder
            joint_config = joint_refiner.get_config()
            gd_prediction = gripper_decoder(joint_config)
            gripper_pc = gd_prediction['gripper_points'].squeeze(0)
            gripper_contact_points = gd_prediction['contact_points'].squeeze(0)

            # apply current pose to gripper pc
            current_pose = pose_refiner.get_pose()
            gripper_pc = transform_points(current_pose, gripper_pc)
            gripper_contact_points = transform_points(current_pose, gripper_contact_points)

            # prepare for backbone ConvONet (normalise to [-0.5, 0.5])
            pre_norm_tf = pre_normalisation_tf().to(device)
            gripper_pc = transform_points(pre_norm_tf, gripper_pc)
            gripper_contact_points = transform_points(pre_norm_tf, gripper_contact_points)

            # evaluate occupancy or SDF and compute losses
            start_time = timer()
            sdf_gripper, sdf_contacts = model.eval_sdf([gripper_pc, gripper_contact_points])

            # we can define a distance at which we consider the gripper points to not be in collision
            sdf_range = cfg['data']['clamp_sdf']
            collision_avoidance_distance = 0.02  # distance that is accepted as collision-free in [m]
            collision_eps = collision_avoidance_distance / sdf_range  # scale according to backbone config

            # basically make sure all gripper points are at least that far away from a surface
            clamped_sdf = torch.clamp_max(sdf_gripper, collision_eps)
            loss_collision = torch.nn.functional.l1_loss(
                clamped_sdf, torch.full_like(sdf_gripper, fill_value=collision_eps), reduction='mean')

            # make sure gripper contact points are at sdf=0, i.e. at the surface
            loss_contact = torch.nn.functional.mse_loss(
                sdf_contacts, torch.zeros_like(sdf_contacts), reduction='mean')

            # also check for grasp quality
            contact_distance = 0.005  # distance from surface that is accepted as being in contact in [m]
            contact_eps = contact_distance / sdf_range

            n_contacts = torch.sum(torch.abs(sdf_contacts) <= contact_eps, dim=-1)
            points_in_contact = gripper_contact_points[n_contacts >= 2]
            print(f'points in contact: {torch.unique(n_contacts, sorted=True, return_counts=True)}')
            print(f'points_in_contact.shape: {points_in_contact.shape}')
            if len(points_in_contact) > 0:
                qualities = model.eval_grasps(points_in_contact)
                print('qualities:', qualities)
                # loss_quality = torch.nn.functional.mse_loss(qualities, torch.ones_like(qualities), reduction='mean')
                loss_quality = torch.nn.functional.l1_loss(qualities, torch.ones_like(qualities), reduction='mean')
                # todo: maybe try l1 loss here?
            else:
                loss_quality = torch.tensor(2)

            iter_stats['time to eval sdf and grasps [s]'] = timer() - start_time

            # make the update
            lambda_1 = 0.1
            lambda_2 = 0.01
            total_refinement_loss = 5 * loss_collision + lambda_1 * loss_contact + lambda_2 * loss_quality
            total_refinement_loss.backward(torch.ones_like(total_refinement_loss))
            pose_refiner.step()
            pose_refiner.zero_grad()

            iter_stats['L_collision'] = f'{loss_collision.cpu().item():.4f}'
            iter_stats[f'L_contact ({lambda_1:.2f})'] = f'{loss_contact.cpu().item():.4f}'
            iter_stats[f'L_quality ({lambda_2:.2f})'] = f'{loss_quality.cpu().item():.4f}'
            iter_stats['L_total'] = f'{total_refinement_loss.cpu().item():.4f}'

            joint_refiner.step()
            joint_refiner.zero_grad()

            iter_stats['joint_config'] = joint_refiner.get_config().item()

            # visualise and record statistics
            sdf_gripper = sdf_gripper.detach().cpu().numpy()
            occupied_mask = sdf_gripper <= -contact_eps
            unoccupied_mask = sdf_gripper >= contact_eps
            iter_stats['n_occupied'] = np.count_nonzero(occupied_mask)
            iter_stats['n_unoccupied'] = np.count_nonzero(unoccupied_mask)
            iter_stats['n_surface'] = np.count_nonzero(~occupied_mask & ~unoccupied_mask)
            iter_stats['min_sdf'] = np.min(sdf_gripper)

            gripper_pc = gripper_pc.detach().cpu()
            vis_data['gripper_point_clouds'].append(gripper_pc)
            vis_data['transforms'].append(pose_refiner.get_pose().cpu().detach().numpy())
            if visualise_each_iter:
                surface_pts = gripper_pc[~occupied_mask & ~unoccupied_mask]
                occupied_pts = gripper_pc[occupied_mask]
                unoccupied_pts = gripper_pc[unoccupied_mask]

                pprint.pprint(iter_stats)
                point_clouds = [pts.numpy() for pts in [occupied_pts, unoccupied_pts, surface_pts] if len(pts) > 1]
                burg.visualization.show_geometries([inputs, *point_clouds])

            stats[f'iteration {i:02d}'] = iter_stats

            if iter_stats['n_occupied'] == 0 and iter_stats['n_surface'] == 0 and loss_quality.cpu().item() < 1.0:
                print(f'SUCCESSFUL AND COLLISION-FREE GRASP')
                break

        pprint.pprint(stats)
        n = len(vis_data['gripper_point_clouds'])
        vis_objs = [inputs]
        for i, gripper_pc in enumerate(vis_data['gripper_point_clouds']):
            # if i not in [0, 1, 2, 10]:
            #     continue
            gripper_pc = burg.util.numpy_pc_to_o3d(gripper_pc)
            gripper_pc.paint_uniform_color([(n-i-1)/(n-1), i/(n-1), 0])
            vis_objs.append(gripper_pc)
            # vis_objs.append(burg.visualization.create_frame(size=0.05, pose=vis_data['transforms'][i]))

        burg.visualization.show_geometries(vis_objs)


if __name__ == '__main__':
    main(parse_args())
