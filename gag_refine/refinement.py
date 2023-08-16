import torch
import numpy as np

from .utils.gram_schmidt import GramSchmidtRotationMapping
from .utils.transform import transform_points
from .dataset import pre_normalisation_tf


class PoseRefinement:
    def __init__(self, device, refine_translation=True, refine_orientation=True, alpha=0.1, max_translation_step=0.005):
        self.device = device
        self.refine_translation = refine_translation
        self.refine_orientation = refine_orientation
        self._translation = None
        self._orientation = None
        self.alpha = alpha  # learning rate
        self.max_translation_step = max_translation_step  # limits the updates to this step (default: 5mm)
        self.orientation_mapping = GramSchmidtRotationMapping

    def set_initial_pose(self, pose):
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)

        # create tensors for translation and rotation with grad etc.
        translation = pose[:3, 3].clone().detach()
        self._translation = translation.to(self.device).float()
        self._translation.requires_grad = self.refine_translation

        orientation = self.orientation_mapping.representation_for_optimisation(pose[:3, :3]).clone().detach()
        self._orientation = orientation.to(self.device).float()
        self._orientation.requires_grad = self.refine_orientation

    def get_pose(self):
        # retrieve (3x3) rotation matrix from internal orientation representation
        rot_mat = self.orientation_mapping.rotation_matrix(self._orientation)
        # append translation (3x4)
        pose = torch.cat([rot_mat, self._translation.unsqueeze(1)], dim=1)
        # append last line
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=pose.dtype)], dim=0)
        return pose

    def step(self):
        # make one optimisation step
        if self.refine_translation:
            delta_t = self._translation.grad

            # limit magnitude of gradient
            norm_t = torch.norm(delta_t, p=2, dim=-1).to(self.device)
            # print(f'gradient norm {norm_t:.5f}')
            alpha = torch.tensor(self.alpha).to(self.device)
            alpha = torch.min(alpha, self.max_translation_step / norm_t)

            # print(f'translation step: {torch.norm(alpha*delta_t, p=2, dim=-1)*1000:.2f} [mm]')
            self._translation = self._translation - alpha * delta_t

        if self.refine_orientation:
            # todo: restrict angular change as well? perhaps by 5 degree? this is a bit more effort though.
            # alpha = 5 * self.alpha
            alpha = 0.3
            # previous_pose = self.get_pose()
            self._orientation = self._orientation - alpha * self._orientation.grad
            # self.print_orientation_change(previous_pose)

    def zero_grad(self):
        # simply reinitialise, this will construct new tensors
        self.set_initial_pose(self.get_pose())

    def print_orientation_change(self, previous_pose):
        current_rot = self.get_pose().clone().detach().cpu().numpy()[:3, :3]
        previous_rot = previous_pose.clone().detach().cpu().numpy()[:3, :3]

        delta_rot = np.dot(previous_rot, current_rot.T)
        theta = (np.trace(delta_rot) - 1) / 2
        theta = np.minimum(np.maximum(theta, -1), 1)  # clamp to [-1, 1] to avoid numerical issues
        angle = np.arccos(theta) * (180 / np.pi)
        print(f'angular change: {angle:.2f} degrees')


class ConfigRefinement:
    def __init__(self, device, alpha=0.1, config_refinement=True, bounds=(0.1, 1.0)):
        self.device = device
        self.alpha = alpha
        self.config_refinement = config_refinement
        self.bounds = bounds
        self._config = None

    def get_config(self):
        return self._config

    def set_initial_config(self, config):
        if not isinstance(config, torch.Tensor):
            config = torch.from_numpy(np.asarray(config))
            if len(config.shape) == 0:
                config = config.unsqueeze(0)

        self._config = config.clone().detach().to(self.device).float()
        self._config.requires_grad = self.config_refinement

    def step(self):
        if self.config_refinement:
            # print(f'config step: {torch.norm(self.alpha * self._config.grad, p=2, dim=-1):.5f}')
            self._config = self._config - self.alpha * self._config.grad
            self._config = torch.clamp(self._config, min=self.bounds[0], max=self.bounds[1])
            # print(f'new config is {self.get_config()}')

    def zero_grad(self):
        self.set_initial_config(self.get_config())


class GraspRefinement:
    """ class to perform the refinement

    Args:
         model: ConvSDFNetInterface
         gripper_decoder: a gripper decoder model
         pose_refiner: an instance of PoseRefinement
         config_refiner: an instance of ConfigRefinement
         collision_avoidance_distance: distance from a surface that is aimed for to be collision-free in [m]
         collision_threshold: distance from a surface that is accepted to be collision-free in [m]
         contact_distance: max distance from a surface to be considered as being in contact in [m]
    """
    def __init__(self, model, gripper_decoder, pose_refiner=None, config_refiner=None,
                 collision_avoidance_distance=0.02, collision_threshold=0.005, contact_distance=0.005,
                 lambda_collision=5, lambda_contact=0.1, lambda_grasp=0.01):
        assert hasattr(model, 'eval_grasps'), f'model does not have eval_grasps. backbone: {type(model)}'
        assert hasattr(model, 'eval_sdf'), f'model does not have eval_sdf. backbone: {type(model)}'

        self.model = model
        self.gripper_decoder = gripper_decoder
        self.device = model.device
        self.pose_refiner = pose_refiner or PoseRefinement(self.device)
        self.config_refiner = config_refiner or ConfigRefinement(self.device)

        # refinement parameters
        self.collision_avoidance_distance = collision_avoidance_distance
        self.collision_threshold = collision_threshold
        self.contact_distance = contact_distance

        # loss weights
        self.lambda_collision = lambda_collision
        self.lambda_contact = lambda_contact
        self.lambda_grasp = lambda_grasp

    def refine_grasps(self, scene_data, grasp_candidates, initial_configs, max_iter=20, early_stopping=True,
                      return_contact_and_success=False):
        """ Performs the iterative refinement for the given grasp candidates

        Args:
            scene_data (dict): test data item
            grasp_candidates (np.ndarray): Nx4x4 with N being the number of initial poses
            initial_configs (np.ndarray): NxC with C being gripper DOF
            max_iter (int): maximum iterations
            early_stopping (bool): if True, will stop as soon as grasp is considered successful
            return_contact_and_success (bool): if True, will return two more arrays of bools which indicate whether
                the refiner thought that there were two or more points in contact and the grasp was stable

        Returns:
            tuple of (np.ndarray (Nx4x4) of refined poses; ndarray (N, C) of gripper configs)
        """
        # use backbone to encode scene
        self.model.eval_scene_pointcloud(scene_data)

        refined_poses = np.zeros_like(grasp_candidates)
        refined_configs = np.zeros_like(initial_configs)
        had_contact = np.zeros(len(grasp_candidates), dtype=bool)
        was_successful = np.zeros(len(grasp_candidates), dtype=bool)

        for i in range(len(grasp_candidates)):  # todo: this could probably be batched???
            # initialise the tensors we want to optimise
            self.pose_refiner.set_initial_pose(grasp_candidates[i])
            self.config_refiner.set_initial_config(initial_configs[i])

            for it in range(max_iter):
                # generate gripper point cloud and contact points using gripper decoder
                joint_config = self.config_refiner.get_config()
                gd_prediction = self.gripper_decoder(joint_config)
                gripper_pc = gd_prediction['gripper_points'].squeeze(0)
                gripper_contact_points = gd_prediction['contact_points'].squeeze(0)

                # apply current pose to gripper pc
                current_pose = self.pose_refiner.get_pose()
                gripper_pc = transform_points(current_pose, gripper_pc)
                gripper_contact_points = transform_points(current_pose, gripper_contact_points)

                # prepare for backbone ConvONet (normalise to [-0.5, 0.5])
                pre_norm_tf = pre_normalisation_tf().to(self.device)
                gripper_pc = transform_points(pre_norm_tf, gripper_pc)
                gripper_contact_points = transform_points(pre_norm_tf, gripper_contact_points)

                # evaluate occupancy or SDF and compute losses
                sdf_gripper, sdf_contacts = self.model.eval_sdf([gripper_pc, gripper_contact_points])

                # we can define a distance at which we consider the gripper points to not be in collision
                sdf_range = self.model.cfg['data']['clamp_sdf']
                collision_eps = self.collision_avoidance_distance / sdf_range  # scale according to backbone config

                # basically make sure all gripper points are at least that far away from a surface
                clamped_sdf = torch.clamp_max(sdf_gripper, collision_eps)
                loss_collision = torch.nn.functional.l1_loss(
                    clamped_sdf, torch.full_like(sdf_gripper, fill_value=collision_eps), reduction='mean')

                # make sure gripper contact points are at sdf=0, i.e. at the surface
                loss_contact = torch.nn.functional.mse_loss(
                    sdf_contacts, torch.zeros_like(sdf_contacts), reduction='mean')

                # also check for grasp quality
                contact_eps = self.contact_distance / sdf_range
                n_contacts = torch.sum(torch.abs(sdf_contacts) <= contact_eps, dim=-1)
                points_in_contact = gripper_contact_points[n_contacts >= 2]  # need at least two contact points
                if len(points_in_contact) > 0:
                    qualities = self.model.eval_grasps(points_in_contact)
                    loss_quality = torch.nn.functional.l1_loss(qualities, torch.ones_like(qualities), reduction='mean')
                    had_contact[i] = True
                else:
                    loss_quality = torch.tensor(2)
                    had_contact[i] = False

                # compute the weighted total loss
                total_refinement_loss = \
                    self.lambda_collision * loss_collision + \
                    self.lambda_contact * loss_contact + \
                    self.lambda_grasp * loss_quality
                total_refinement_loss.backward(torch.ones_like(total_refinement_loss))

                # check if we need to refine further or we can stop early  # todo: this will not work if batched
                if early_stopping:
                    # condition 1: grasp is stable
                    if loss_quality.cpu().item() < 1.0:
                        # condition 2: no collision
                        n_occupied = torch.sum(sdf_gripper <= self.collision_threshold)
                        if n_occupied == 0:
                            # grasp is stable and collision-free, stop refinement
                            had_contact[i] = True
                            was_successful[i] = True
                            break

                # make the updates
                self.pose_refiner.step()
                self.pose_refiner.zero_grad()
                self.config_refiner.step()
                self.config_refiner.zero_grad()

            refined_poses[i] = self.pose_refiner.get_pose().detach().cpu().numpy()
            refined_configs[i] = self.config_refiner.get_config().detach().cpu().numpy()

        if return_contact_and_success:
            return refined_poses, refined_configs, had_contact, was_successful
        else:
            return refined_poses, refined_configs

