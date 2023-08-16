import argparse
from timeit import default_timer as timer
import pprint

import torch
import numpy as np
import burg_toolkit as burg

from gag_refine.dataset import pre_normalise_points
from convonets.src import config


def parse_args():
    parser = argparse.ArgumentParser(
        description='refine the gripper pose and configuration in a given scene'
    )
    parser.add_argument('convonet_config', type=str, help='path to ConvONet config file')

    return parser.parse_args()


def main(args):
    visualise_each_iter = True
    mode = 'random'  # 'full_pc', 'input_pc', 'random'

    cfg = config.load_config(args.convonet_config, config.default_config_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # load ConvONet model and checkpoint
    convonet = config.get_model_interface(cfg, device, dataset)

    for it, data in enumerate(test_loader):
        scene_idx = data['idx'].item()
        print(f'{"*"*5} scene {scene_idx}')

        # statistics
        stats = {}

        # process scene point cloud to create latent representation
        start_time = timer()
        convonet.eval_scene_pointcloud(data)
        stats['time to encode scene [s]'] = timer() - start_time

        vis_data = {
            'inputs': data['inputs'].squeeze(0).cpu().numpy(),
            'surface_points': [],
        }

        # load the full point cloud, because those points are in fact sampled from the GT object surface and hence
        # should all have surface property
        if mode == 'full_pc':
            fn = f'data/gag/scenes/{600+scene_idx:04d}/full_point_cloud.npz'
            full_pc = dict(np.load(fn))['points']
            points = torch.from_numpy(full_pc).to(device)
            points = pre_normalise_points(points).float()
        elif mode == 'input_pc':
            points = data['inputs']
        else:
            points = torch.rand((convonet.points_batch_size, 3), device=device).float()
            points -= 0.5  # shift to [-0.5, 0.5]

        # evaluate occupancy
        start_time = timer()
        occ = convonet.eval_occupancy(points, compute_grad=False).detach().cpu().numpy()
        print('occ shape', occ.shape)
        stats['time to eval occupancy [s]'] = timer() - start_time

        # visualise and record statistics
        threshold = 0.5
        epsilon = 0.05
        occupied_mask = occ >= threshold + epsilon
        unoccupied_mask = occ <= threshold - epsilon
        stats['n_occupied (red)'] = np.count_nonzero(occupied_mask)
        stats['n_unoccupied (green)'] = np.count_nonzero(unoccupied_mask)
        stats['n_surface (orange)'] = np.count_nonzero(~occupied_mask & ~unoccupied_mask)

        points = points.detach().cpu().numpy()
        surface_points = points[~occupied_mask & ~unoccupied_mask]
        vis_data['surface_points'].append(surface_points)

        if visualise_each_iter:
            occupied_pts = points[occupied_mask]
            unoccupied_pts = points[unoccupied_mask]

            pprint.pprint(stats)
            point_clouds = []
            for j, pts in enumerate([occupied_pts, surface_points, unoccupied_pts]):
                if len(pts) <= 1:
                    continue
                o3dpc = burg.util.numpy_pc_to_o3d(pts)
                o3dpc.paint_uniform_color([(2-j)/2, j/2, 0])
                point_clouds.append(o3dpc)
            burg.visualization.show_geometries([vis_data['inputs'], *point_clouds])


if __name__ == '__main__':
    main(parse_args())
