from torch import nn
import os
from ..encoder import encoder_dict
from ..conv_onet.models import decoder_dict
from .. import data
from .model import ConvolutionalSDFNetwork
from .training import ConvSDFNetTrainer
from .generation import Generator3DSDF
from .interface import ConvSDFNetInterface
from gag_refine.grasp_quality_estimator import create_grasp_quality_net


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    decoder = decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if 'grasp_quality_net' in cfg['model'].keys() and cfg['model']['grasp_quality_net'] is not None:
        grasp_quality_net = create_grasp_quality_net(cfg)
    else:
        grasp_quality_net = None

    model = ConvolutionalSDFNetwork(
        decoder, encoder, grasp_quality_net, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = ConvSDFNetTrainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    generator = Generator3DSDF(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type=cfg['data']['input_type'],
        padding=cfg['data']['padding']
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])

    input_type = cfg['data']['input_type']
    if input_type != 'pointcloud':
        raise NotImplementedError('conv_sdfnet only supports pointcloud as input type')

    fields = {}
    if cfg['data']['points_file'] is not None:
        print(f'loading {mode} sdf data..')
        fields['points'] = data.PointsSDFField(
            cfg['data']['points_file'], points_transform,
            unpackbits=cfg['data']['points_unpackbits'],
            multi_files=cfg['data']['multi_files'],
            clamp_sdf=cfg['data']['clamp_sdf'],
            clamp_margin=cfg['data']['clamp_margin_sdf']
        )

    if 'grasp_quality_net' in cfg['model'].keys() and cfg['model']['grasp_quality_net'] is not None:
        print(f'loading {mode} grasp data..')
        subsample = None if mode in ('val', 'test') else cfg['data']['sample_grasps']
        fields['grasps'] = data.GraspsField(
            dataset_dir=cfg['data']['path'],
            n_fingers=cfg['data']['n_fingers'],
            n_sample=subsample,
            clamp_fc=cfg['data']['clamp_fc'],
            clamp_margin=cfg['data']['clamp_margin_fc'],
            contact_noise=cfg['data']['contact_noise']
        )
    else:
        print('not loading grasp data. please specify data/sample_grasps in config yaml.')

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsSDFField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files'],
                clamp_sdf=cfg['data']['clamp_sdf'],
                clamp_margin=cfg['data']['clamp_margin_sdf']
            )

    return fields


def get_model_interface(model, device, cfg):
    """
    Get interface for usage in GAG Refine
    Args:
        model (ConvolutionalOccupancyNetwork): the model
        device: device
        cfg: configuration

    Returns:
        ConvSDFNetInterface
    """
    return ConvSDFNetInterface(model, cfg, device=device)

