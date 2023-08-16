import torch
import torch.nn as nn
import torch.nn.functional as F

from convonets.src.checkpoints import CheckpointIO
from convonets.src.layers import ResnetBlockFC
from convonets.src.common import normalize_coordinate, normalize_3d_coordinate

"""
so what do i want.
basically a mix of the LocalDecoder and the LocalPoolPointnet used for encoding.
- retrieve the features for the contact points from the scene encoding (N x c_dim)
- pass then through the first ResNet block  (N x c_dim) -> (N x hidden_dim)
- then for each block:
    - perform global max pooling to get pooled features  (N x hidden_dim)
    - concatenate contact point features (N x hidden_dim) with pooled features (N x hidden_dim) -> (N x 2*hidden_dim)
    - [ALTERNATIVE]: also concatenate the latent features from the scene encoding, processed by some linear (as in decoder)
    - feed through next ResNet block w/ (N x 2*hidden_dim) -> (N x hidden_dim)
"""


def create_grasp_quality_net(config):
    """ creates a model based on the given config """
    print('loading grasp quality estimator with hard-coded configuration.... todo')
    grasp_quality_estimator = GraspQualityEstimator(
        dim=config['data']['dim'],
        padding=config['data']['padding'],
        c_dim=config['model']['c_dim'],
        leaky=False,
        **config['model']['grasp_quality_net_kwargs']
    )

    print(grasp_quality_estimator)
    print(f'total number of parameters: {sum(p.numel() for p in grasp_quality_estimator.parameters())}')
    return grasp_quality_estimator


def load_grasp_quality_net(config, model_fn='model.pt'):
    grasp_quality_net = create_grasp_quality_net(config)
    checkpoint_io = CheckpointIO(config['training']['out_dir'], model=grasp_quality_net)
    checkpoint_io.load(model_fn)
    return grasp_quality_net


class GraspQualityEstimator(nn.Module):
    """ Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        pooling (str): pooling used, max|avg
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1,
                 pooling='max'):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.dim = dim

        self.fc_p = nn.Linear(dim, hidden_size)  # position encoder

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)
        ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size),
            *[ResnetBlockFC(2*hidden_size, hidden_size) for _ in range(n_blocks-1)]
        ])

        assert pooling in ['max', 'avg'], f'pooling is {pooling}'
        if pooling == 'max':  # applied over the features of the different contact points
            self.pool = lambda x: torch.max(x, dim=-2, keepdim=True)[0]
        else:
            self.pool = lambda x: torch.mean(x, dim=-2, keepdim=True)

        self.actvn = F.relu if not leaky else lambda x: F.leaky_relu(x, 0.2)
        self.fc_out = nn.Linear(hidden_size, 1)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze \
            (-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        # gathering the features only works with a 3-dim tensor, therefore we need to reshape
        # p is in the form of: [b1, b2, n, 3], if not, we have to adjust it to b1 of c
        # c is in the form of: [b1, c_dim, plane_res, plane_res]
        #   b1 = number of scenes
        #   b2 = number of grasps
        #   n = number of contact points
        squeeze = False
        if len(p.shape) == 3:
            squeeze = True
            p = p.unsqueeze(0)

        b1, b2, n = p.shape[0], p.shape[1], p.shape[2]
        p = p.reshape(b1, n*b2, self.dim)

        # gather latent code features
        plane_type = list(c_plane.keys())
        c = 0
        if 'grid' in plane_type:
            c += self.sample_grid_feature(p, c_plane['grid'])
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
        c = c.transpose(1, 2)

        # reshape back
        p = p.reshape(b1, b2, n, self.dim)
        c = c.reshape(b1, b2, n, self.c_dim)

        p = p.float()
        net = self.fc_p(p)  # first layer uses only positional encoding, no latent code
        for i, block in enumerate(self.blocks):
            net = net + self.fc_c[i](c)  # add processed features from scene encoding
            # concatenate pooled features only after first block
            if i > 0:
                pooled = self.pool(net)  # (b1, b2, 1, hidden_dim)
                pooled = pooled.repeat(1, 1, n, 1)  # propagate to the number of contact points
                net = torch.cat([net, pooled], dim=-1)  # (b1, b2, n, 2*hidden_dim)
            net = block(net)  # (b1, b2, n, hidden_dim)

        net = self.pool(net).squeeze(-2)  # final pool, then reduce contact point dimension -> (b1, b2, hidden_dim)
        out = self.fc_out(self.actvn(net))  # final layer
        out = out.squeeze(-1)  # squeeze to (b1, b2), we have one predicted value per grasp
        if squeeze:
            out = out.squeeze(0)
        out = torch.tanh(out)
        return out
