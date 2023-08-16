import torch
import torch.nn as nn

from convonets.src.checkpoints import CheckpointIO


def create_gripper_decoder(config):
    """ creates a model based on the given config """
    requested_decoder = config['decoder']['model']

    decoders = {
        'simple_decoder': {
            'class': GripperDecoder,
            'kwargs': 'simple_decoder_kwargs'
        },
        'latent_decoder': {
            'class': GripperDecoderWithLatentVector,
            'kwargs': 'latent_decoder_kwargs'
        },
        'latent_skip': {
            'class': GripperDecoderFromLatentWithSkip,
            'kwargs': 'latent_skip_kwargs'
        }
    }

    if requested_decoder != 'simple_decoder':
        raise NotImplementedError('need to update the other decoders to predict contact points as well')

    if requested_decoder not in decoders.keys():
        raise NotImplementedError(f'no support for {requested_decoder}, only have {decoders}.')

    kwargs = config['decoder'][decoders[requested_decoder]['kwargs']]
    if 'n_joints' not in kwargs.keys():
        kwargs['n_joints'] = config['data']['n_joints']
    if 'n_contacts' not in kwargs.keys():
        kwargs['n_contacts'] = config['data']['n_contacts']

    gripper_decoder = decoders[requested_decoder]['class'](**kwargs)
    print('loaded following gripper decoder:')
    print(gripper_decoder)
    print(f'total number of parameters: {sum(p.numel() for p in gripper_decoder.parameters())}')
    return gripper_decoder


def load_gripper_decoder(config, model_fn='model.pt'):
    gripper_decoder = create_gripper_decoder(config)
    checkpoint_io = CheckpointIO(config['training']['out_dir'], model=gripper_decoder)
    checkpoint_io.load(model_fn)
    return gripper_decoder


class GripperDecoder(nn.Module):
    """
    This is a simple Gripper Decoder which, starting from the joint configuration q, uses a stack of fc layers with relu
    to produce n_points points.
    """
    def __init__(self, n_joints=1, n_contacts=2, n_points=2048, layers=(256, 512)):
        super().__init__()
        self.n_joints = n_joints
        self.n_points = n_points
        self.n_contacts = n_contacts

        assert len(layers) == 2, 'not implemented'
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_joints, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], (n_points+n_contacts)*3)  # xyz per point
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.n_joints))  # batch_size, n_joints
        x = self.linear_relu_stack(x)
        x = torch.reshape(x, (-1, self.n_points+self.n_contacts, 3))  # batch_size, n_points, xyz
        output = {
            'gripper_points': x[:, :self.n_points],
            'contact_points': x[:, self.n_points:]
        }
        return output


class GripperDecoderWithLatentVector(nn.Module):
    """
    This gripper decoder is similar to the simple GripperDecoder, but it concatenates q with a latent parameter
    vector and uses this combined feature as input, to produce the point cloud points.
    """
    def __init__(self, n_joints=1, n_points=2048, layers=(256, 512), n_latent=16):
        super().__init__()
        self.n_joints = n_joints
        self.n_points = n_points

        if len(layers) != 2:
            raise NotImplementedError('currently only exactly two layers supported')

        self.latent_vector = nn.Parameter(torch.empty((1, n_latent)), requires_grad=True)
        nn.init.kaiming_uniform_(self.latent_vector)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_joints+n_latent, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], n_points*3)  # xyz per point
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.n_joints))  # batch_size, n_joints
        latent_vector = self.latent_vector.repeat(x.shape[0], 1)
        x = torch.cat((x, latent_vector), dim=1)
        x = self.linear_relu_stack(x)
        x = torch.reshape(x, (-1, self.n_points, 3))  # batch_size, n_points, xyz
        return x


class GripperDecoderFromLatentWithSkip(nn.Module):
    """
    Starting from only a latent parameter vector, we feed q into every subsequent layer by concatenation.
    """
    def __init__(self, n_joints=1, n_points=2048, layers=(256, 512), n_latent=16):
        super().__init__()
        self.n_joints = n_joints
        self.n_points = n_points
        self.n_latent = n_latent

        if len(layers) != 2:
            raise NotImplementedError('currently only exactly two layers supported')

        self.latent_vector = nn.Parameter(torch.empty((1, n_latent)), requires_grad=True)
        nn.init.kaiming_uniform_(self.latent_vector)

        self.fc1 = nn.Sequential(nn.Linear(n_latent, layers[0]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(n_joints + layers[0], layers[1]), nn.ReLU())
        self.fc3 = nn.Linear(n_joints + layers[1], n_points*3)

    def forward(self, q):
        q = torch.reshape(q, (-1, self.n_joints))  # batch_size, n_joints
        x = self.latent_vector.repeat(q.shape[0], 1)
        x = self.fc1(x)
        x = torch.cat((q, x), dim=1)
        x = self.fc2(x)
        x = torch.cat((q, x), dim=1)
        x = self.fc3(x)
        x = torch.reshape(x, (-1, self.n_points, 3))  # batch_size, n_points, xyz
        return x

