import os
import argparse
import shutil

import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter

from convonets.src import config, data
from convonets.src.checkpoints import CheckpointIO
from gag_refine.grasp_quality_estimator import create_grasp_quality_net


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a grasp quality network.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    return parser.parse_args()


def log_losses(loss_dict, logger, epoch, prefix=None):
    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '/'

    for loss_name, loss_values in loss_dict.items():
        losses = np.asarray(loss_values)
        print(f'  {loss_name:15s}  avg {np.mean(losses):>7f}; min: {np.min(losses):>7f}; max: {np.max(losses):>7f}')
        logger.add_scalar(f'{prefix}{loss_name}', np.mean(losses), epoch)


def append_stats_to_dict(loss_dict, ground_truth, predictions):
    tp = ((ground_truth > 0) & (predictions > 0)).sum().item()
    fp = ((ground_truth <= 0) & (predictions > 0)).sum().item()
    fn = ((ground_truth > 0) & (predictions <= 0)).sum().item()
    tn = ((ground_truth <= 0) & (predictions <= 0)).sum().item()
    n_total = torch.numel(ground_truth)
    ccr = (tp + tn) / n_total
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    for key, score in zip(['ccr', 'precision', 'recall'], [ccr, precision, recall]):
        if key not in loss_dict.keys():
            loss_dict[key] = []
        loss_dict[key].append(score)

    for key, score in zip(['tp', 'fp', 'fn', 'tn'], [tp, fp, fn, tn]):
        key = f'detail/{key}'
        if key not in loss_dict.keys():
            loss_dict[key] = []
        loss_dict[key].append(score/n_total)


def train(cfg):
    # config id
    epochs = 2000
    learning_rate = 1e-3
    hidden_dim = 64
    loss_name = 'l1_loss'  # l1_loss | mse_loss
    lr_sched_every = None  # None or epoch
    train_conf = f'ep{epochs}_lr{learning_rate}_n{cfg["data"]["sample_grasps"]}_' \
                 f'sched{lr_sched_every}_hd{hidden_dim}_{loss_name}'
    print('training config:', train_conf)

    # prepare output directory, save the config
    out_dir = os.path.join(cfg['training']['out_dir'], 'grasp_quality_estimator')
    log_dir = os.path.join(out_dir, 'logs', train_conf)
    out_dir = os.path.join(out_dir, train_conf)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg, return_idx=True)

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    # initialise model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

    # setup convonet backbone
    backbone = config.get_model_interface(cfg, device, train_dataset)

    # grasp quality estimator
    grasp_quality_estimator = create_grasp_quality_net(cfg).to(device)

    # setup distance measure for loss computation
    mse_loss = torch.nn.MSELoss()  # todo: see if there's anything better
    l1_loss = torch.nn.L1Loss()
    assert loss_name in ['l1_loss', 'mse_loss']
    if loss_name == 'l1_loss':
        loss_fn = l1_loss
    else:
        loss_fn = mse_loss

    # optimizer
    # assert config['training']['optimizer']['method'] in ['adam', 'sgd']
    # if config['training']['optimizer']['method'] == 'adam':
    #     optimizer = torch.optim.Adam(gripper_decoder.parameters(), **config['training']['optimizer']['adam_kwargs'])
    # else:
    #     optimizer = torch.optim.SGD(gripper_decoder.parameters(), **config['training']['optimizer']['sgd_kwargs'])
    optimizer = torch.optim.Adam(grasp_quality_estimator.parameters(), lr=learning_rate, weight_decay=1e-4)

    # learning rate scheduler, initialising checkpoint
    if lr_sched_every is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_sched_every)
        checkpoint_io = CheckpointIO(out_dir, model=grasp_quality_estimator, optimizer=optimizer, scheduler=scheduler)
    else:
        scheduler = None
        checkpoint_io = CheckpointIO(out_dir, model=grasp_quality_estimator, optimizer=optimizer)

    logger = SummaryWriter(log_dir)
    iteration = 0
    best_val_score = np.inf
    best_epoch = None
    for epoch in range(1, epochs+1):
        grasp_quality_estimator.train()
        epoch_losses = {
            loss_name: [],
        }

        for batch in train_loader:
            iteration += 1
            # todo: we somehow need to ensure that all grasps in the same batch have the same number of contacts
            backbone.eval_scene_pointcloud(batch)
            scene_encoding = backbone.latent_code
            contact_points = batch['grasps.contact_points'].float().to(device)
            gt_scores = batch['grasps.scores'].float().to(device)

            optimizer.zero_grad()
            pred_scores = grasp_quality_estimator(contact_points, scene_encoding)

            loss = loss_fn(gt_scores, pred_scores).mean()  # average loss over samples in a batch
            loss.backward()
            optimizer.step()
            epoch_losses[loss_name].append(loss.item())
            append_stats_to_dict(epoch_losses, gt_scores, pred_scores)

        if scheduler is not None:
            scheduler.step()

        print(f'** ep{epoch} - it{iteration}:')
        log_losses(epoch_losses, logger, iteration, 'train')

        if (epoch+1) % 10 == 0:
            print(f'{"*" * 5} validating {"*" * 5}')
            grasp_quality_estimator.eval()
            val_losses = {
                'l1_loss': [],
                'mse_loss': [],
            }
            with torch.no_grad():
                for val_batch in val_loader:
                    backbone.eval_scene_pointcloud(val_batch)
                    scene_encoding = backbone.latent_code
                    contact_points = val_batch['grasps.contact_points'].float().to(device)
                    gt_scores = val_batch['grasps.scores'].float().to(device)

                    pred_scores = grasp_quality_estimator(contact_points, scene_encoding)
                    val_losses['mse_loss'].append(mse_loss(gt_scores, pred_scores).mean().item())
                    val_losses['l1_loss'].append(l1_loss(gt_scores, pred_scores).mean().item())
                    append_stats_to_dict(val_losses, gt_scores, pred_scores)

            # todo: should somehow be able to evaluate all the grasps, i.e. 2-finger, 3-finger, etc.
            # not just the randomly sampled ones
            print(f'evaluated {len(val_loader)} scenes')
            log_losses(val_losses, logger, iteration, 'val')
            val_score = np.mean(val_losses[loss_name])
            print(f'best val {loss_name} score was {best_val_score:.5f} from epoch {best_epoch}')
            if val_score < best_val_score:
                print('='*10)
                print(f'saving new best model at ep{epoch}, it{iteration}, '
                      f'with score {val_score:.5f}')
                checkpoint_io.save('model_best.pt')
                best_val_score = val_score
                best_epoch = epoch
                print('='*10)
            print('*'*10)

    checkpoint_io.save('model_final.pt')
    print(f'saved grasp_quality_estimator to {os.path.join(out_dir, "model_final.pt")}')


if __name__ == '__main__':
    args = parse_args()
    train(config.load_config(args.config, config.default_config_fn))
