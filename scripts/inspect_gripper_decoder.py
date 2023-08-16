import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
import burg_toolkit as burg

from convonets.src.config import load_config
from gag_refine.gripper_decoder import load_gripper_decoder
from gag_refine.dataset.gripper_dataset import GripperPointCloudData


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect a trained gripper decoder.'
    )
    parser.add_argument('config', type=str, help='Path to config file (in out directory).')
    return parser.parse_args()


def inspect(config, device):
    # get some data
    test_data = GripperPointCloudData(
        dataset_dir=config['data']['path'],
        split='test'
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    gripper_decoder = load_gripper_decoder(config).to(device)
    gripper_decoder.eval()
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            joint_conf = data['config']
            gt_points = data['points']
            print(f'predicting gripper point cloud for joint conf {joint_conf}')
            gt_points, joint_conf = gt_points.float().to(device), joint_conf.float().to(device)

            prediction = gripper_decoder(joint_conf)
            pred_points = prediction['gripper_points']
            contact_points = prediction['contact_points']

            # visualise
            gt_points = gt_points.squeeze(0).cpu().detach().numpy()
            pred_points = pred_points.squeeze(0).cpu().detach().numpy()
            print(f'gt_points: {gt_points.shape}; pred_points: {pred_points.shape}')

            o3d_gt_points = burg.util.numpy_pc_to_o3d(gt_points)
            o3d_gt_points.paint_uniform_color([1.0, 0., 0.])
            burg.visualization.show_geometries([o3d_gt_points, pred_points])


def show_joint_space(config, device):
    # show generated gripper point clouds covering the whole joint space
    gripper_decoder = load_gripper_decoder(config).to(device)
    gripper_decoder.eval()
    generated_point_clouds = []
    steps = np.linspace(0.1, 1., num=5)
    print('steps:', steps)
    with torch.no_grad():
        for joint_config in steps:
            joint_config = torch.tensor(joint_config, device=device)
            generated_point_clouds.append(gripper_decoder(joint_config)['gripper_points'].squeeze(0).cpu().numpy())

    burg.visualization.show_geometries(generated_point_clouds)


if __name__ == '__main__':
    args = parse_args()
    config_fn = args.config
    cfg = load_config(config_fn)

    computing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {computing_device}')

    # inspect(cfg, computing_device)
    show_joint_space(cfg, computing_device)
    # create_figure_of_net(cfg, computing_device)
