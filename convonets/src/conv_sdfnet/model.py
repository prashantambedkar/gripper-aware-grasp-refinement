import torch

from ..conv_onet.models import ConvolutionalOccupancyNetwork


class ConvolutionalSDFNetwork(ConvolutionalOccupancyNetwork):
    """
    This class modifies the ConvONet in a way to predict SDF instead of binary occupancy.
    In particular, the last decoder layer is changed to tanh; other than that it is pretty much the same.
    It also potentially adds a Grasp Quality Net as second decoder.
    """
    def __init__(self, decoder, encoder=None, grasp_quality_net=None, device=None):
        super().__init__(decoder, encoder, device)

        if grasp_quality_net is not None:
            self.grasp_quality_net = grasp_quality_net.to(device)
        else:
            self.grasp_quality_net = None

    def decode(self, p, c, **kwargs):
        """ Returns predicted SDF values for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        return torch.tanh(self.decoder(p, c, **kwargs))

    def predict_grasp_quality(self, p, c):
        """ Predicts the grasp quality for the given contact point tuples.

        Args:
            p (tensor): contact points
            c (tensor): latent conditioned code c
        """
        return self.grasp_quality_net(p, c)
