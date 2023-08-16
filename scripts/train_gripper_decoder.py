import os
import argparse
import shutil

import torch
from torch.utils.data import DataLoader
import numpy as np
import burg_toolkit as burg

from convonets.src.config import load_config
from convonets.src.checkpoints import CheckpointIO
from gag_refine.gripper_decoder import create_gripper_decoder
from gag_refine.utils.chamfer_distance import ChamferDistance
from gag_refine.utils.earth_mover_distance import EarthMoverDistance
from gag_refine.dataset.gripper_dataset import GripperPointCloudData

# todo:
# inspect some qualitative results in training using the inspect_gripper_decoder.py
# decoding fewer points as in the gt_data might lead to bad approximation of the finger tips
# options are:
#   - match the numbers by decoding 2048 points
#   - down sample annotations to 1024 points, also introduces some randomness
# to increase robustness we might:
#   - introduce some tiny noise to the annotated points (as in ConvONet)
#   - introduce dropout?
# further play with the architecture
#   - varying number of layers, nodes, etc.
#   - it seems the point cloud basically scales with q, although the fingers do not open enough
#       - might be reasonable to have some static latent vector as input, that is not subject to q
#       - much like autodecoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a gripper decoder.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    return parser.parse_args()


def validate(config, model, device):
    print(f'{"*"*5} validating {"*"*5}')
    test_data = GripperPointCloudData(
        dataset_dir=config['data']['path'],
        split='test'
    )
    test_dataloader = DataLoader(test_data, batch_size=1)

    # compute both distances
    cd = ChamferDistance()
    emd = EarthMoverDistance()
    mse = torch.nn.MSELoss()

    cd_losses = []
    emd_losses = []
    mse_losses = []

    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            joint_conf = data['config'].float().to(device)
            gt_points = data['points'].float().to(device)
            gt_contacts = data['contact_points'].float().to(device)

            prediction = model(joint_conf)
            pred_points = prediction['gripper_points']
            contacts = prediction['contact_points']

            cd_losses.append(cd(gt_points, pred_points).mean().item())  # average loss over samples in a batch
            emd_losses.append(emd(gt_points, pred_points).mean().item())
            mse_losses.append(mse(contacts, gt_contacts).mean().item())

    cd_losses = np.asarray(cd_losses)
    emd_losses = np.asarray(emd_losses)
    mse_losses = np.asarray(mse_losses)
    print(f'evaluated {len(test_data)} samples')
    print(f'\t CD: avg {np.mean(cd_losses):>7f}; min: {np.min(cd_losses):>7f}; max: {np.max(cd_losses):>7f}')
    print(f'\tEMD: avg {np.mean(emd_losses):>7f}; min: {np.min(emd_losses):>7f}; max: {np.max(emd_losses):>7f}')
    print(f'\tMSE: avg {np.mean(mse_losses):>7f}; min: {np.min(mse_losses):>7f}; max: {np.max(mse_losses):>7f}')
    print('*'*10)
    result_dict = {
        'chamfer_distance': cd_losses.mean(),
        'earth_mover_distance': emd_losses.mean(),
        'mse_loss_contacts': mse_losses.mean()
    }
    return result_dict


def train(config):
    # prepare output directory, save the config
    out_dir = config['training']['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

    # get some data
    train_data = GripperPointCloudData(
        dataset_dir=config['data']['path'],
        noise_std=config['training']['point_cloud_noise'],
        split='train',
    )
    train_dataloader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    # initialise model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    gripper_decoder = create_gripper_decoder(config).to(device)

    # setup distance measure for loss computation
    assert config['training']['loss_fn'] in ['chamfer_distance', 'earth_mover_distance']
    if config['training']['loss_fn'] == 'chamfer_distance':
        loss_fn = ChamferDistance()
    else:
        loss_fn = EarthMoverDistance()

    contact_loss_fn = torch.nn.MSELoss()

    # optimizer
    assert config['training']['optimizer']['method'] in ['adam', 'sgd']
    if config['training']['optimizer']['method'] == 'adam':
        optimizer = torch.optim.Adam(gripper_decoder.parameters(), **config['training']['optimizer']['adam_kwargs'])
    else:
        optimizer = torch.optim.SGD(gripper_decoder.parameters(), **config['training']['optimizer']['sgd_kwargs'])

    # learning rate scheduler
    if config['training']['use_lr_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config['training']['lr_scheduler_kwargs'])
    else:
        scheduler = None

    for epoch in range(config['training']['epochs']):
        gripper_decoder.train()
        epoch_losses = {
            'points_loss': [],
            'contacts_loss': [],
            'total_loss': []
        }

        def register_losses(ploss, closs, tloss):
            epoch_losses['points_loss'] = ploss.item()
            epoch_losses['contacts_loss'] = closs.item()
            epoch_losses['total_loss'] = tloss.item()

        for batch, data in enumerate(train_dataloader):
            joint_conf = data['config'].float().to(device)
            gt_points = data['points'].float().to(device)
            gt_contacts = data['contact_points'].float().to(device)

            optimizer.zero_grad()
            prediction = gripper_decoder(joint_conf)
            pred_points = prediction['gripper_points']
            pred_contacts = prediction['contact_points']

            loss = loss_fn(gt_points, pred_points).mean()  # average loss over samples in a batch
            contact_loss = contact_loss_fn(pred_contacts, gt_contacts).mean()
            total_loss = loss + contact_loss

            total_loss.backward()
            optimizer.step()
            register_losses(loss, contact_loss, total_loss)

        if scheduler is not None:
            scheduler.step()

        print(f'** ep{epoch}:')
        for loss_name, loss_values in epoch_losses.items():
            losses = np.asarray(loss_values)
            print(f'\t{loss_name}: avg {np.mean(losses):>7f}; min: {np.min(losses):>7f}; max: {np.max(losses):>7f}')

        if (epoch+1) % config['training']['validate_every'] == 0:
            validate(config, gripper_decoder, device)

    # save the model
    if scheduler is not None:
        checkpoint_io = CheckpointIO(out_dir, model=gripper_decoder, optimizer=optimizer, scheduler=scheduler)
    else:
        checkpoint_io = CheckpointIO(out_dir, model=gripper_decoder, optimizer=optimizer)
    checkpoint_io.save('model.pt')
    print(f'saved gripper decoder to {os.path.join(out_dir, "model.pt")}')

    # *****************************************
    # training finished, now do some evaluation
    # only visual inspection for now
    gripper_decoder.eval()
    generated_point_clouds = []
    with torch.no_grad():
        for open_width in np.linspace(0.1, 1.0, num=5):
            print(f'opening width: {open_width}')
            open_width = torch.Tensor([open_width]).to(device)
            prediction = gripper_decoder(open_width)
            generated_point_clouds.append(prediction['gripper_points'].squeeze(0).cpu().numpy())
            generated_point_clouds.append(prediction['contact_points'].squeeze(0).cpu().numpy())

    burg.visualization.show_geometries(generated_point_clouds)


if __name__ == '__main__':
    args = parse_args()
    config_fn = args.config
    train(load_config(config_fn))
