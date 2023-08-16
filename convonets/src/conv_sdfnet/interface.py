from timeit import default_timer as timer

import torch

from convonets.src.common import add_key, sdf_to_occ


class ConvSDFNetInterface(object):
    """  Interface class for using Convolutional Occupancy Networks in GAG Refine.

    Args:
        model (nn.Module): trained Occupancy Network model
        cfg (dict): config (loaded configuration file)
        points_batch_size (int): batch size for points evaluation
        device (device): pytorch device
    """
    def __init__(self, model, cfg, points_batch_size=100000, device=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.device = device
        self.cfg = cfg

        # save latent scene code
        self._latent_code = None

    @property
    def latent_code(self):
        if self._latent_code is None:
            raise ValueError('need to eval point cloud first to generate latent code')
        return self._latent_code

    def eval_scene_pointcloud(self, data):
        """ Processes pointcloud data to create a latent code for a scene.

        Args:
            data (dict): data to be processed

        Returns:
            dict, statistics with time
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        t0 = timer()
        with torch.no_grad():
            self._latent_code = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = timer() - t0
        return stats_dict

    def eval_sdf(self, query_points, compute_grad=True):
        """ Evaluates the SDF of the given point sets. You can provide either a list of tensors or one single
        tensor; the returned types adjust accordingly.
        Will also compute the gradients per default.
        Todo: Currently it seems that the decoder can only handle batch size of 1

        Args:
            query_points (torch.Tensor or list): Nx3 or BxNx3 gripper points; or list of Nx3 or BxNx3 tensors
            compute_grad (bool): If set to false, will not compute the gradients.
        Returns:
            torch.Tensor or list of torch.Tensors, Nx1 or BxNx1 sdf values [0, 1]
        """
        input_was_list = True
        if not isinstance(query_points, list):
            input_was_list = False
            query_points = [query_points]

        points = torch.cat(query_points, dim=-2)  # either (M+N)x3 or Bx(M+N)x3

        if points.ndim == 2:
            assert points.shape[0] <= self.points_batch_size
        elif points.ndim == 3:
            assert points.shape[0] * points.shape[1] <= self.points_batch_size
            if points.shape[0] > 1:
                print(f'using batch size of more than 1, might give errors now...?')

        squeeze_back = False
        if points.ndim == 2:
            points = points.unsqueeze(0)
            squeeze_back = True
        points = points.to(self.device)

        self.model.eval()
        if compute_grad:
            sdf_hat = self.model.decode(points, self.latent_code)
        else:
            with torch.no_grad():
                sdf_hat = self.model.decode(points, self.latent_code)

        if squeeze_back:
            sdf_hat = sdf_hat.squeeze(0)

        if input_was_list:
            start_idx = 0
            return_values = []
            for t in query_points:
                end_idx = start_idx + t.shape[-2]
                return_values.append(sdf_hat[..., start_idx:end_idx])
                start_idx = end_idx
            return return_values
        else:
            return sdf_hat

    def eval_occupancy(self, query_points, compute_grad=True):
        """
        function that internally computes sdf, but then converts to occupancy-like values.
        not probabilities though!!

        Args:
            query_points (torch.Tensor or list): Nx3 or BxNx3 gripper points; or list of Nx3 or BxNx3 tensors
            compute_grad (bool): If set to false, will not compute the gradients.
        Returns:
            torch.Tensor or list of torch.Tensors, Nx1 or BxNx1 occupancy values in [0, 1] (but actually scaled sdf...)
        """
        sdf_hat = self.eval_sdf(query_points, compute_grad=compute_grad)
        if isinstance(sdf_hat, list):
            occ_hat = [sdf_to_occ(sdf) for sdf in sdf_hat]
        else:
            occ_hat = sdf_to_occ(sdf_hat)
        return occ_hat

    def eval_grasps(self, contact_points, compute_grad=True):
        """
        evaluates the grasp stability of the given contact points.
        Args:
            contact_points (tensor): BxNxMx3 or NxMx3 contact points
            compute_grad (bool): Whether to compute gradients

        Returns:
            torch.Tensor, BxNx1 or Nx1
        """
        if not hasattr(self.model, 'grasp_quality_net') or self.model.grasp_quality_net is None:
            raise ValueError(f'current model has no grasp quality net. model type: {type(self.model)}')

        self.model.eval()
        if compute_grad:
            fc_hat = self.model.predict_grasp_quality(contact_points, self.latent_code)
        else:
            with torch.no_grad():
                fc_hat = self.model.predict_grasp_quality(contact_points, self.latent_code)

        return fc_hat
