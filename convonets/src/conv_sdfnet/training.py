import torch

from ..common import compute_iou, sdf_to_occ, add_key
from ..conv_onet.training import Trainer
from gag_refine.dataset.ferrari_canny import compute_stats


class ConvSDFNetTrainer(Trainer):
    """
    Trainer object for the SDF variant of ConvONet.
    """
    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        super().__init__(model, optimizer, device, input_type, vis_dir, threshold, eval_sample)

        # check if our ConvSDFNet also has a GraspQualityNet, if so we train that jointly
        self.has_grasps = hasattr(model, 'grasp_quality_net') and model.grasp_quality_net is not None

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # print('training.eval_step: data dict', data.keys())
        # keys are:
        # 'points', 'points.occ', 'points.sdf', 'points_iou', 'points_iou.occ', 'points_iou.sdf',
        # 'inputs', 'inputs.normals', 'idx'
        # 'grasps', 'grasps.scores', 'grasps.contact_points' (optional)

        points = data.get('points').to(device)
        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        sdf_gt = data.get('points_iou.sdf').to(device)

        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # forward pass
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
            sdf_out = self.model.decode(points_iou, c)

        occ_out = sdf_to_occ(sdf_out)
        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (occ_out >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['sdf_iou'] = iou

        l1_loss = torch.nn.functional.l1_loss(sdf_out, sdf_gt, reduction='none')
        l1_loss = l1_loss.mean()  # skipping the sum here, as all the samples have different number of points
        eval_dict['sdf_l1_loss'] = l1_loss.item()

        if self.has_grasps:
            contact_points = data.get('grasps.contact_points').to(device)
            gt_scores = data.get('grasps.scores').to(device)
            with torch.no_grad():
                scores_out = self.model.predict_grasp_quality(contact_points, c)

            l1_loss_fc = torch.nn.functional.l1_loss(scores_out, gt_scores, reduction='none')
            l1_loss_fc = l1_loss_fc.mean()
            eval_dict['grasps_l1_loss'] = l1_loss_fc.item()

            # compute CCR, precision, recall
            stats = compute_stats(gt_scores, scores_out)
            for key, val in stats.items():
                eval_dict[f'grasps_{key}'] = val

        return eval_dict

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.compute_loss(data)
        losses['loss'].backward()
        self.optimizer.step()

        for key, val in losses.items():
            losses[key] = val.item()
        return losses

    def compute_loss(self, data):
        ''' Computes the loss. L1 loss for SDF values and grasp scores.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        # dict has: points, points.occ, points.sdf, inputs, inputs.normals
        p = data.get('points').to(device)
        gt_sdf = data.get('points.sdf').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        pred_sdf = self.model.decode(p, c, **kwargs)
        loss_sdf = torch.nn.functional.l1_loss(pred_sdf, gt_sdf, reduction='none')
        loss_sdf = loss_sdf.sum(dim=-1).mean()  # sum over points, average over the samples

        losses = {
            'loss': loss_sdf,
        }

        if self.has_grasps:
            contact_points = data.get('grasps.contact_points').to(device)
            gt_scores = data.get('grasps.scores').to(device)
            pred_scores = self.model.predict_grasp_quality(contact_points, c)
            loss_fc = torch.nn.functional.l1_loss(pred_scores, gt_scores, reduction='none')
            loss_fc = loss_fc.sum(dim=-1).mean()
            losses['loss_sdf'] = loss_sdf
            losses['loss_fc'] = loss_fc
            losses['loss'] = loss_sdf + 0.1 * loss_fc

        return losses
