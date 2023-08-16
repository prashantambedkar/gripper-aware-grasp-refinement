import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from convonets.src import config, data
from convonets.src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from pprint import pprint


# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
print('Args initialized')
cfg = config.load_config(args.config, 'configs/default.yaml')
print('cfg initialized')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
print('is cuda available')
device = torch.device("cuda" if is_cuda else "cpu")
print('device checked')
# Set t0
t0 = time.time()
print('to time set')
# Shorthands
out_dir = cfg['training']['out_dir']
print('out dir ok')
batch_size = cfg['training']['batch_size']
print('batch size set')
backup_every = cfg['training']['backup_every']
print('backup_every')
vis_n_outputs = cfg['generation']['vis_n_outputs']
print('vis_n_outputs')
exit_after = args.exit_after
print('exit_after')
model_selection_metric = cfg['training']['model_selection_metric']
print('model_selection_metric')
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
    print('model_selection_sign:1')
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
    print('model_selection_sign:-1')
else:
    print('model_selection_mode:error')
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print('makedirs',out_dir)
print('exists',out_dir)
shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
print('copyfile')
# Dataset
train_dataset = config.get_dataset('train', cfg)
print('train_dataset')
val_dataset = config.get_dataset('val', cfg, return_idx=True)
print('val_dataset')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
print('train_loader')

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
print('val_loader')
# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
print('vis_loader')
model_counter = defaultdict(int)
print('model_counter')
data_vis_list = []

# Build a data dictionary for visualization
iterator = iter(vis_loader)
print('iterator',iterator)
print('len(vis_loader)',len(vis_loader))
for i in range(len(vis_loader)):
    data_vis = next(iterator)
    print('data_vis')
    idx = data_vis['idx'].item()
    print('idx')
    model_dict = val_dataset.get_model_dict(idx)
    print('model_dict')
    category_id = model_dict.get('category', 'n/a')
    print('category_id')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    print('category_name')
    category_name = category_name.split(',')[0]
    print('category_name_split')
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    print('c_it')
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1
    print('model_counter')

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
print('model_config')
# Generator
generator = config.get_generator(model, cfg, device=device)
print('generator')
# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print('optimizer')
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)
print('trainer')
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
print('checkpoint_io')
try:
    load_dict = checkpoint_io.load('model.pt')
    print('load_dict_try')
except FileExistsError:
    load_dict = dict()
    print('load_dict_except')
epoch_it = load_dict.get('epoch_it', 0)
print('epoch_it')
it = load_dict.get('it', 0)
print('it')
it_best = load_dict.get('it_best', 0)
print('it_best')
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)
print('metric_val_best')
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))
print('logger')
# Shorthands
print_every = cfg['training']['print_every']
print('print_every')
checkpoint_every = cfg['training']['checkpoint_every']
print('checkpoint_every')
validate_every = cfg['training']['validate_every']
print('validate_every')
visualize_every = cfg['training']['visualize_every']
print('visualize_every')
# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)
        print('loss')
        if not isinstance(loss, dict):
            loss = {
                'loss': loss
            }
        for key, val in loss.items():
            logger.add_scalar(f'train/{key}', val, it)
            print('logger_add_scalar')

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            elapsed_time_str = str(datetime.timedelta(seconds=time.time() - t0)).split('.')[0]  # remove milliseconds
            loss_str = ''
            for key, val in loss.items():
                loss_str += f'{key}: {val:.4f}, '
            loss_str = loss_str[:-2]
            print(f'[{t.hour:02d}:{t.minute:02d}]-[{elapsed_time_str}] ep: {epoch_it:03d}, it: {it:06d}, {loss_str}')

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            for data_vis in data_vis_list:
                if cfg['generation']['sliding_window']:
                    out = generator.generate_mesh_sliding(data_vis['data'])    
                else:
                    out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}

                mesh.export(os.path.join(out_dir, 'vis', '{}_{}_{}.off'.format(it, data_vis['category'], data_vis['it'])))


        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best, it_best=it_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best, it_best=it_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)

            metric_val = eval_dict[model_selection_metric]
            print(f'validation metrics ({model_selection_metric} used to determine model_best.pt)')
            for key, val in eval_dict.items():
                print(f'\t{key:17s} {val:.5f}')
            print(f'last best model was at it {it_best} with val score {metric_val_best}')

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                it_best = it
                print(f'found new best model at it {it} with val score {metric_val_best:.4f}')
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best, it_best=it_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best, it_best=it_best)
            exit(3)
