import argparse
from timeit import default_timer as timer
import pprint

import torch
import numpy as np
import burg_toolkit as burg
import trimesh

from convonets.src import config
from convonets.src.common import sdf_to_occ


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine the gripper pose and configuration in a given scene'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')

    return parser.parse_args()


def get_pcs_coloured_by_occupancy(points, occupancy, threshold=0.5, epsilon=0.02):
    occupancy = occupancy.detach().cpu().numpy()
    points = points.detach().cpu().numpy()

    occupied_mask = occupancy >= threshold + epsilon
    unoccupied_mask = occupancy <= threshold - epsilon

    surface_points = points[~occupied_mask & ~unoccupied_mask]
    occupied_pts = points[occupied_mask]
    unoccupied_pts = points[unoccupied_mask]

    print(f'occupied points: {occupied_pts.shape}')
    print(f'surface points: {surface_points.shape}')
    print(f'free space points: {unoccupied_pts.shape}')
    return_mask = [len(pc) > 1 for pc in [occupied_pts, surface_points, unoccupied_pts]]

    occupied_pts = burg.util.numpy_pc_to_o3d(occupied_pts)
    surface_points = burg.util.numpy_pc_to_o3d(surface_points)
    unoccupied_pts = burg.util.numpy_pc_to_o3d(unoccupied_pts)

    occupied_pts.paint_uniform_color([1, 0, 0])
    surface_points.paint_uniform_color([0.78, 0.78, 0])
    unoccupied_pts.paint_uniform_color([0, 1, 0])

    return_pcs = []
    for pc, mask in zip([occupied_pts, surface_points, unoccupied_pts], return_mask):
        if mask:
            return_pcs.append(pc)
    return return_pcs


def main(args):
    visualise_each_iter = False

    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    backbone = config.get_model_interface(cfg, device, dataset)

    # statistics
    stats = {}
    for it, data in enumerate(test_loader):
        scene_idx = data['idx'].item()
        print(f'{"*"*5} scene {scene_idx}')

        # process scene point cloud to create latent representation
        start_time = timer()
        backbone.eval_scene_pointcloud(data)
        stats['time to encode scene [s]'] = timer() - start_time
        inputs = data['inputs'].squeeze(0).cpu().numpy()  # for vis purposes

        # sample initial points
        points = torch.rand((1000, 3), device=device, dtype=torch.float)
        points[:, 2] /= 2  # make them closer to the plane initially
        points -= 0.5

        icosphere_pts = torch.from_numpy(trimesh.creation.icosphere(subdivisions=1, radius=1).vertices) * 0.05
        icosphere_pts = icosphere_pts.float()
        icosphere_pts = torch.cat([torch.zeros(1, 3), icosphere_pts], dim=0)  # add original point to it as well
        icosphere_pts = icosphere_pts.to(device)

        vis_data = {
            'points': [points.cpu().numpy()],
            'probe_points': []
        }
        for i in range(100):
            print(f'iter {i} --------------')
            iter_stats = {}

            # create samples around the points
            probe_points = icosphere_pts.repeat(points.shape[0], 1, 1)
            probe_points /= torch.tensor([i+1], device=device).float()
            probe_points = probe_points + points.unsqueeze(1)

            # evaluate occupancy
            start_time = timer()
            occ = backbone.eval_occupancy(probe_points.reshape(1, -1, 3), compute_grad=False)  # N x 17(?)
            occ = occ.reshape(points.shape[0], -1)
            iter_stats['time to eval occupancy [s]'] = timer() - start_time

            # check out which of the occupancy values is closest to the target, choose corresponding probe point as
            # new point (pretty stupid optimisation, not computing any sort of gradient)
            threshold = 0.5
            error_val = torch.abs(occ - threshold)  # n_points X n_probe_points
            min_error, min_indices = torch.min(error_val, dim=1)  # n_points
            # todo: i assume there are no gradients where the points escape... i.e. min_error is 0.5 ??
            # for those, we should randomise it
            points = probe_points[torch.arange(len(min_indices)), min_indices]
            print(f'point errors avg: {min_error.mean()}, min: {min_error.min()}, max: {min_error.max()}')

            # visualise and record statistics
            vis_data['points'].append(points.cpu().numpy())
            threshold = 0.5
            epsilon = 0.4
            occ = occ.detach().cpu().numpy()
            occupied_mask = occ >= threshold + epsilon
            unoccupied_mask = occ <= threshold - epsilon
            iter_stats['n_occupied'] = np.count_nonzero(occupied_mask)
            iter_stats['n_unoccupied'] = np.count_nonzero(unoccupied_mask)
            iter_stats['n_surface'] = np.count_nonzero(~occupied_mask & ~unoccupied_mask)

            if visualise_each_iter:
                pprint.pprint(iter_stats)
                # visualise current points, previous points, and inputs
                burg.visualization.show_geometries([inputs, vis_data['points'][-2], vis_data['points'][-1]])

            stats[f'iteration {i:02d}'] = iter_stats

        pprint.pprint(stats)
        n = len(vis_data['points'])
        vis_objs = [inputs]
        for i, gripper_pc in enumerate(vis_data['points']):
            # if i not in [0, 1, 2, 10]:
            #     continue
            gripper_pc = burg.util.numpy_pc_to_o3d(gripper_pc)
            gripper_pc.paint_uniform_color([(n-i-1)/(n-1), i/(n-1), 0])
            vis_objs.append(gripper_pc)
            # vis_objs.append(burg.visualization.create_frame(size=0.05, pose=vis_data['transforms'][i]))

        burg.visualization.show_geometries(vis_objs)

        # evaluate final points
        occ = backbone.eval_occupancy(points, compute_grad=False)
        pcs = get_pcs_coloured_by_occupancy(points, occ)
        burg.visualization.show_geometries([inputs, *pcs])


def gradient_based(args):
    visualise_each_iter = True
    use_probe_points = True

    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    backbone = config.get_model_interface(cfg, device, dataset)

    icosphere_pts = torch.from_numpy(trimesh.creation.icosphere(subdivisions=1, radius=1).vertices) * 0.05
    icosphere_pts = icosphere_pts.float()
    icosphere_pts = torch.cat([icosphere_pts, torch.zeros(1, 3)], dim=0)  # add original point to it as well
    icosphere_pts = icosphere_pts.to(device)

    # statistics
    stats = {}
    for it, data in enumerate(test_loader):
        scene_idx = data['idx'].item()
        print(f'{"*" * 5} scene {scene_idx}')

        # process scene point cloud to create latent representation
        start_time = timer()
        backbone.eval_scene_pointcloud(data)
        stats['time to encode scene [s]'] = timer() - start_time
        inputs = data['inputs'].squeeze(0).cpu().numpy()  # for vis purposes

        # sample initial points
        points = torch.rand((1000, 3), device=device, dtype=torch.float)
        points[:, 2] /= 2  # make them closer to the plane initially
        points -= 0.5

        # init params and optimizer
        points = torch.nn.Parameter(points, requires_grad=True)
        opt = torch.optim.Adam(params=[points], lr=1e-3)

        vis_data = {
            'points': [points.detach().cpu().numpy()],
        }
        for i in range(100):
            print(f'iter {i} --------------')
            iter_stats = {}

            # evaluate occupancy
            opt.zero_grad()

            if use_probe_points:
                # create samples around the points to gather more gradients
                probe_points = icosphere_pts.repeat(points.shape[0], 1, 1)
                # probe_points = probe_points / torch.tensor([i+1], device=device).float()  # get smaller over time
                probe_points = probe_points + points.unsqueeze(1)
            else:
                probe_points = points.unsqueeze(1)

            start_time = timer()
            occ = backbone.eval_occupancy(probe_points.reshape(1, -1, 3), compute_grad=True)
            occ = occ.reshape(points.shape[0], -1)
            iter_stats['time to eval occupancy [s]'] = timer() - start_time

            get_pcs_coloured_by_occupancy(points, torch.mean(occ, dim=-1))  # prints the frequencies

            loss_contact = torch.nn.functional.mse_loss(occ, torch.full_like(occ, 0.5), reduction='mean')
            loss_contact.backward()
            opt.step()

            # visualise and record statistics
            vis_data['points'].append(points.detach().cpu().numpy())
            # stats[f'iteration {i:02d}'] = iter_stats

        pprint.pprint(stats)
        n = len(vis_data['points'])
        vis_objs = [inputs]
        for i, gripper_pc in enumerate(vis_data['points']):
            # if i not in [0, 1, 2, 10]:
            #     continue
            gripper_pc = burg.util.numpy_pc_to_o3d(gripper_pc)
            gripper_pc.paint_uniform_color([(n - i - 1) / (n - 1), i / (n - 1), 0])
            vis_objs.append(gripper_pc)
            # vis_objs.append(burg.visualization.create_frame(size=0.05, pose=vis_data['transforms'][i]))

        burg.visualization.show_geometries(vis_objs)

        # evaluate final points
        occ = backbone.eval_occupancy(points, compute_grad=False)
        pcs = get_pcs_coloured_by_occupancy(points, occ)
        burg.visualization.show_geometries([inputs, *pcs])


if __name__ == '__main__':
    # main(parse_args())
    gradient_based(parse_args())
