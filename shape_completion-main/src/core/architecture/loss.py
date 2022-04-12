from distutils.log import error
from pyrsistent import b
import torch
from geom.mesh.op.gpu.base import batch_vnrmls, batch_fnrmls_fareas, batch_moments, batch_surface_volume, \
    vertex_velocity
from geom.mesh.op.gpu.dist import batch_l2_pdist
from util.strings import warn
from .collect_reconstruction_stats import collect_reconstruction_stats
import numpy as np
import statistics as st
import sys
import os
sys.path.insert(0,os.path.join('..','..'))
from visualize.get_objects_hardcoded_for_sets_base import get_segmentation_manger
from visualize.computation_manager import ErrorComputationDiffManger, get_valid_error_computations_type_list,get_valid_error_computations_type_list_for_flow


# from chamferdist import ChamferDist
# distpc1topc2, distpc2topc1, idx1, idx2 = chamferDist(pc1, pc2)
# ----------------------------------------------------------------------------------------------------------------------
#                           Full Losses (different architecture might have different losses)
# ----------------------------------------------------------------------------------------------------------------------
class BasicLoss:
    loss_instance_counter=0
    def __init__(self, hp, f):
        self.lambdas = list(hp.lambdas)
        self.body_part_volume_weights = list(hp.body_part_volume_weights)
        self.shape_diff = ShapeDiffLoss(hp, f)
        self.f = f
        self.best = {}
        self.worst = {}
        self.mean = {}
        self.iter = 0
        BasicLoss.loss_instance_counter += 1
        assert BasicLoss.loss_instance_counter!=2, f"loss instance count {BasicLoss.loss_instance_counter}"
        segmentation_manger=get_segmentation_manger()
        self.organs_to_lambdas = {
            "RightArm": self.body_part_volume_weights[0],
            "LeftArm": self.body_part_volume_weights[1],
            "Head": self.body_part_volume_weights[2],
            "RightLeg": self.body_part_volume_weights[3],
            "LeftLeg": self.body_part_volume_weights[4]
        }
        
        self.organs_to_lambdas = {k:v  for (k,v) in self.organs_to_lambdas.items() if v>0}

        segmentation_manger_backwards=get_segmentation_manger(organs=list(self.organs_to_lambdas.keys()))

        # self._err_manger_logging=ErrorComputationDiffManger(f=f,segmentation_manger=segmentation_manger)
        self._err_manger_loss=ErrorComputationDiffManger(f=f,segmentation_manger=segmentation_manger_backwards,computation_type_list=get_valid_error_computations_type_list_for_flow())

    def closest_to_mean(self,array,mean):
        if not isinstance(array,np.ndarray):
            array = np.array(array)
        distance = abs(array-mean)
        return np.argmin(distance)

    def compute_loss_start(self):
        self.best = {}
        self.worst = {}
        self.mean = {}
        self.iter = 0


    def _get_mean_by_metric(self,metric,stats,gt_hi,tp_hi):
        new_mean = np.mean(stats[metric])
        if metric in self.mean:
            new_mean = (new_mean + self.mean[metric][2]*self.iter)/(self.iter+1)
        indice = self.closest_to_mean(stats[metric],new_mean)
        return (
            gt_hi[indice], 
            tp_hi[indice], 
            stats[metric][indice])


    def compute_loss_end(self, gt_hi, tp_hi, gt, masks, tp, comp, face_override=False):
        #return collect_reconstruction_stats(gt, masks, tp, comp, self.f)
        stats =  collect_reconstruction_stats(gt, masks, tp, comp,self.f)
        
        if len(gt.shape) > 3:
            gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])
        if len(tp.shape) > 3:
            tp = tp.reshape(-1, tp.shape[-2], tp.shape[-1])

        best_vals = {}
        best_vals['Comp-GT Vertex L2'] =  (
            gt_hi[np.argmin(stats['Comp-GT Vertex L2'])],
             tp_hi[np.argmin(stats['Comp-GT Vertex L2'])],
              min(stats['Comp-GT Vertex L2']))
        best_vals['Comp-GT Volume L1'] = (
            gt_hi[np.argmin(stats['Comp-GT Volume L1'])], 
            tp_hi[np.argmin(stats['Comp-GT Volume L1'])], 
            min(stats['Comp-GT Volume L1']))
        best_vals['TP-GT Vertex L2'] = (
            gt_hi[np.argmin(stats['TP-GT Vertex L2'])], 
            tp_hi[np.argmin(stats['TP-GT Vertex L2'])], 
            min(stats['TP-GT Vertex L2']))
        if len(self.best) != 0:
            if best_vals['Comp-GT Volume L1'][2] < self.best['Comp-GT Volume L1'][2]:
                self.best['Comp-GT Volume L1'] = best_vals['Comp-GT Volume L1']
            if best_vals['TP-GT Vertex L2'][2] < self.best['TP-GT Vertex L2'][2]:
                self.best['TP-GT Vertex L2'] = best_vals['TP-GT Vertex L2']
            if best_vals['Comp-GT Vertex L2'][2] < self.best['Comp-GT Vertex L2'][2]:
                self.best['Comp-GT Vertex L2'] = best_vals['Comp-GT Vertex L2']
        else:
            self.best = best_vals

        worst_vals = {}
        worst_vals['Comp-GT Vertex L2'] =  (
            gt_hi[np.argmax(stats['Comp-GT Vertex L2'])],
             tp_hi[np.argmax(stats['Comp-GT Vertex L2'])],
              max(stats['Comp-GT Vertex L2']))
        worst_vals['Comp-GT Volume L1'] = (
            gt_hi[np.argmax(stats['Comp-GT Volume L1'])], 
            tp_hi[np.argmax(stats['Comp-GT Volume L1'])], 
            max(stats['Comp-GT Volume L1']))
        worst_vals['TP-GT Vertex L2'] = (
            gt_hi[np.argmax(stats['TP-GT Vertex L2'])], 
            tp_hi[np.argmax(stats['TP-GT Vertex L2'])], 
            max(stats['TP-GT Vertex L2']))
        if len(self.worst) != 0:
            if worst_vals['Comp-GT Volume L1'][2] > self.worst['Comp-GT Volume L1'][2]:
                self.worst['Comp-GT Volume L1'] = worst_vals['Comp-GT Volume L1']
            if worst_vals['TP-GT Vertex L2'][2] > self.worst['TP-GT Vertex L2'][2]:
                self.worst['TP-GT Vertex L2'] = worst_vals['TP-GT Vertex L2']
            if worst_vals['Comp-GT Vertex L2'][2] > self.worst['Comp-GT Vertex L2'][2]:
                self.worst['Comp-GT Vertex L2'] = worst_vals['Comp-GT Vertex L2']
        else:
            self.worst = worst_vals
        
        self.mean['Comp-GT Vertex L2'] = self._get_mean_by_metric('Comp-GT Vertex L2',stats,gt_hi,tp_hi)
        self.mean['Comp-GT Volume L1'] = self._get_mean_by_metric('Comp-GT Volume L1',stats,gt_hi,tp_hi)
        self.mean['TP-GT Vertex L2'] = self._get_mean_by_metric('TP-GT Vertex L2',stats,gt_hi,tp_hi)
        self.iter += 1

        return stats

    def return_best_stats(self):
        return self.best

    def return_worst_stats(self):
        return self.worst

    def return_mean_stats(self):
        return self.mean

    def compute(self, x, network_output):
        """
        :param x: The input batch dictionary
        :param network_output: The lightning output
        :return: loss
        """
        completion_gt = x['gt']

        completion_rec = network_output['completion_xyz']
        if 'gt_f' in x:  # TODO
            face_override = True
        else:
            face_override = False
        loss_dict_comp = self.shape_diff.compute(shape_1=completion_gt, shape_2=completion_rec, w=1,
                                                 face_override=face_override)
        # TODO calculate mask: w, w.r.t to mask penalty and distnat vertices (only for completion)
        loss_dict_comp = {f'{k}_comp': v for k, v in loss_dict_comp.items()}

        loss_dict = loss_dict_comp
        loss_dict.update(total_loss=loss_dict['total_loss_comp'])

        #loss segments
        shape1=completion_gt.reshape(-1, completion_gt.shape[-2], completion_gt.shape[-1])[:,:,:3]
        shape2=completion_rec.reshape(-1, completion_rec.shape[-2], completion_rec.shape[-1])[:,:,:3]
        # errors_to_log=self._err_manger_logging.get_compute_errors_dict(shape_1=shape1.detach(),shape_2=shape2.detach())
        errors_to_loss = self._err_manger_loss.get_compute_errors_dict(shape_1=shape1,shape_2=shape2)
        # if self.lambdas[6] != 0:
        #     errors_to_loss['Full volume error'] *= self.lambdas[6]
        #     loss_dict['total_loss']+=errors_to_log['Full volume error']
        
        for organ,l_val in self.organs_to_lambdas.items():
            loss_dict['total_loss']+=l_val*errors_to_loss[f'{organ} volume error']
            loss_dict[f"{organ}_volume_error"]=errors_to_loss[f'{organ} volume error']
        loss_dict.update(total_loss=loss_dict['total_loss_comp'])

        # loss_dict.update(errors_to_log)
        # shape1=None
        # shape2=None
            #compute and area and volume losses too
        return loss_dict


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Loss Terms
# ----------------------------------------------------------------------------------------------------------------------
# Relies on the fact the shapes have the same connectivity
class ShapeDiffLoss:
    def __init__(self, hp, f):
        # Copy over from the hyper-params - Remove ties to the hp container for our own editing
        self.lambdas = list(hp.lambdas)
        self.mask_penalties = list(hp.mask_penalties)
        self.dist_v_penalties = list(hp.dist_v_penalties)
        self.centralize_com = hp.centralize_com
        # For CUDA:
        self.dev = hp.dev  # TODO - Might be problematic for multiple GPUs
        self.non_blocking = hp.NON_BLOCKING
        self.def_prec = getattr(torch, hp.UNIVERSAL_PRECISION)

        # Handle Faces:
        if f is not None and (self.lambdas[1] > 0 or self.lambdas[4] > 0 or self.lambdas[5] > 0 or self.lambdas[6] > 0):
            self.torch_f = torch.from_numpy(f).long().to(device=self.dev, non_blocking=self.non_blocking)

        # Sanity Check - Input Channels:
        # TODO: we should have a transformation block operating on the initial data,
        #  adding input channels, scaling, rotating etc.
        # TODO: hp.in_channels should be defined with respect to transformed input data.
        # For example: the initial data might not have normals (initial input channels = 3),
        # however in the input pipeline
        # we add normals (before the lightning), making the input channel = 6.
        # Now, assume we want to calculate normal loss.
        # If hp.in_channels refers to the initial input channels then the logic below won't work (the assert will fail).
        if self.lambdas[1] > 0 or self.lambdas[4]:
            assert hp.in_channels >= 6, "Only makes sense to lightning normal losses with normals available"
        if self.lambdas[2] > 0:
            assert hp.in_channels >= 12, "Only makes sense to lightning moment losses with moments available"

        # Sanity Check - Destroy dangling mask_penalties/distant_v_penalties
        for i, lamb in enumerate(self.lambdas[0:3]):  # TODO - Implement 0:5
            if lamb <= 0:
                self.mask_penalties[i] = 0
                self.dist_v_penalties[i] = 0

        # Sanity Check - Check validity of the penalties:
        for i in range(len(self.dist_v_penalties)):
            if 0 < self.dist_v_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.dist_v_penalties[i]}.\nPlease use 0 or 1 to remove this specific loss lightning')
            if 0 < self.mask_penalties[i] < 1:
                warn(f'Found an invalid penalty in the distant vertex arg set: at {i} with val '
                     f'{self.mask_penalties[i]}.\nPlease use 0 or 1 to remove this specific loss lightning')

        # Micro-Optimization - Reduce movement to the GPU:
        if [p for p in self.dist_v_penalties if p > 1]:  # if using_distant_vertex
            self.dist_v_ones = torch.ones((1, 1, 1), device=self.dev, dtype=self.def_prec)

    def compute(self, shape_1, shape_2, w, face_override=False):
        """
        :param shape_1: first batch of shapes [b x nv x d] - for which all fields are known
        :param shape_2: second batch shapes [b x nv x 3] - for which only xyz fields are known
        :param w: a set of weighting functions over the vertices
        :return: The loss, as dictionary
        """
        loss = torch.zeros(1, device=self.dev, dtype=self.def_prec)
        loss_dict = {}
        shape1_sequential = None
        shape2_sequential = None
        shape_of_1 = shape_1.shape
        if self.centralize_com:
            com_shape_1= torch.mean(shape_1, dim=1, keepdim=True)
            com_shape_1[:,3:]=0
            shape_1=shape_1-com_shape_1
            com_shape_2= torch.mean(shape_2, dim=1, keepdim=True)
            com_shape_2[:,3:]=0
            shape_2=shape_2-com_shape_2
        if len(shape_of_1) > 3:
            shape1_sequential = shape_1
            shape2_sequential = shape_2.reshape(shape_of_1[0], shape_of_1[1], shape_of_1[2], -1)
            shape_1 = shape_1.reshape(-1, shape_1.shape[-2], shape_1.shape[-1])
            shape_2 = shape_2.reshape(-1, shape_2.shape[-2], shape_2.shape[-1])
        for i, lamb in enumerate(self.lambdas):
            if lamb > 0:
                if i == 0:  # XYZ
                    if not face_override:
                        loss_xyz = self._l2_loss(shape_1[:, :, 0:3], shape_2, lamb=lamb, vertex_mask=w)
                        loss_dict['xyz'] = loss_xyz
                        loss += loss_xyz
                elif i == 1:
                    if not face_override:  # Normals
                        need_f_area = self.lambdas[5] > 0
                        out = batch_vnrmls(shape_2, self.torch_f, return_f_areas=need_f_area)
                        vnb_2, is_valid_vnb_2 = out[0:2]
                        if need_f_area:
                            f_area_2 = out[2]
                        loss_normals = self._l2_loss(shape_1[:, :, 3:6], vnb_2, lamb=lamb,
                                                     vertex_mask=w * is_valid_vnb_2.unsqueeze(2))
                        loss_dict['normals'] = loss_normals
                        loss += loss_normals
                elif i == 2:  # Moments:
                    loss_moments = self._l2_loss(shape_1[:, :, 6:12], batch_moments(shape_2), lamb=lamb, vertex_mask=w)
                    loss_dict['moments'] = loss_moments
                    loss += loss_moments
                elif i == 3:  # Euclidean Distance Matrices (validated)
                    loss_euc_dist = self._l2_loss(batch_l2_pdist(shape_1[:, :, 0:3]),
                                                  batch_l2_pdist(shape_2), lamb=lamb)
                    loss_dict['EucDist'] = loss_euc_dist
                    loss += loss_euc_dist
                elif i == 4:  # Euclidean Distance Matrices with normals (defined on Gauss map)
                    try:
                        vnb_2
                    except NameError:
                        need_f_area = self.lambdas[5] > 0
                        out = batch_vnrmls(shape_2, self.torch_f, return_f_areas=need_f_area)
                        vnb_2, is_valid_vnb_2 = out[0:2]
                        if need_f_area:
                            f_area_2 = out[2]

                    loss_euc_dist_gauss = self._l2_loss(batch_l2_pdist(shape_1[:, :, 3:6]),
                                                        batch_l2_pdist(vnb_2), lamb=lamb)
                    loss_dict['EucDistGauss'] = loss_euc_dist_gauss
                    loss += loss_euc_dist_gauss
                elif i == 5:  # Face Areas
                    f_area_1 = batch_fnrmls_fareas(shape_1[:, :, 0:3], self.torch_f, return_normals=False)
                    try:
                        f_area_2
                    except NameError:
                        f_area_2 = batch_fnrmls_fareas(shape_2, self.torch_f, return_normals=False)
                    loss_areas = self._l2_loss(f_area_1, f_area_2, lamb=lamb, vertex_mask=w)
                    loss_dict['Areas'] = loss_areas
                    loss += loss_areas
                elif i == 6:  # Volume:
                    continue
                    # f_volume_1 = batch_surface_volume(shape_1[:, :, 0:3], self.torch_f.unsqueeze(0))

                    # f_volume_2 = batch_surface_volume(shape_2, self.torch_f.unsqueeze(0))
                    # loss_volumes = torch.linalg.norm(f_volume_1 - f_volume_2)
                    # # loss_volumes = self._l2_loss(f_volume_1, f_volume_2, lamb=lamb, vertex_mask=w)
                    # loss_dict['Volumes'] = loss_volumes
                    # loss += lamb*loss_volumes
                elif i == 7:
                    assert len(shape_of_1) > 3, "Dataset should be configured for sequential data"
                    loss_velocity = lamb * vertex_velocity(shape1_sequential[:, :, :, 0:3], shape2_sequential)
                    loss_dict['Velocity'] = loss_velocity
                    loss += loss_velocity
                # TODO: implement chamfer distance loss
                else:
                    raise AssertionError

        loss_dict['total_loss'] = loss
        return loss_dict

    #def compute_loss_end(self, gt, masks, tp, comp, w, face_override=False):
    #   stats =  collect_reconstruction_stats(gt, masks, tp, comp)
    #    best = {}
    #    best['mean_error'] = min(stats['mean_error'])
    #    best['volume_error'] = min(stats['volume_error'])
    #    best['template_mean_error'] = min(stats['template_mean_error'])
    #    #best_mean_index = stats['mean_error']).index(min(stats['mean_error']))
    #    #best_tp_by_mean = tps[best_mean_index]
    #    self.best = best

        return stats

    def _mask_part_weight(self, mask_b, nv):
        """
        :param mask_b: A list of mask indices as numpy arrays
        :param nv: The number of vertices
        :return w: a weight function admitting 1 on the part and 0 outside the part
        """
        b = len(mask_b)
        w = torch.zeros((b, nv, 1), dtype=self.def_prec)
        for i in range(b):
            w[i, mask_b[i], :] = 1
        return w.to(device=self.dev, non_blocking=self.non_blocking)  # Transfer after looping

    def _mask_penalty_weight(self, mask_b, nv, p):
        """ TODO - This function was never checked
        :param mask_b: A list of mask indices as numpy arrays
        :param nv: The number of vertices
        :param p: Additional weight multiplier for the mask vertices - A scalar > 1
        """
        if p <= 1:
            return 1
        b = len(mask_b)
        w = torch.ones((b, nv, 1), dtype=self.def_prec)
        for i in range(b):
            w[i, mask_b[i][0], :] = p
        return w.to(device=self.dev, non_blocking=self.non_blocking)  # Transfer after looping

    def _distant_vertex_weight(self, gtb_xyz, tpb_xyz, p):
        """ TODO - This function was never checked
        :param gtb_xyz: ground-truth batched tensor [b x nv x 3]
        :param tpb_xyz: template batched tensor [b x nv x 3]
        :param p: Additional weight multiplier for the far off vertices - A scalar > 1
        This function returns a bxnvx1 point-wise weight. For vertices that are similar between gt & tp - return 1.
        For "distant" vertices - return some cost greater than 1.
        Defines the point-wise difference as: d = ||gtb_xyz - tpb_xyz|| - a  [b x nv x 1] vector
        Normalize d by its mean: dhat = d/mean(d)
        Far-off vertices are defined as vertices for which dhat > 1 - i.e.,
        the difference is greater than the mean vertices
        The weight function w is defined by W_i = max(dhat_i,1) * lamb
        """
        if p <= 1:
            return 1
        d = torch.norm(gtb_xyz - tpb_xyz, dim=2, keepdim=True)
        d /= torch.mean(d, dim=1, keepdim=True)  # dhat
        w = torch.max(d, self.dist_v_ones)
        w[w > 1] *= p
        return w  # Promised tensor

    @staticmethod
    def _l2_loss(v1b, v2b, lamb, vertex_mask=1):
        #assert False, f"v1 shape {v1b.shape} v2 shape {v2b.shape}"
        return lamb * torch.mean(vertex_mask * ((v1b - v2b) ** 2))


class CodeLoss:
    def __init__(self):
        pass

    @staticmethod
    def compute(code_1, code_2):
        x = torch.norm(code_1, dim=1)
        y = torch.norm(code_2, dim=1)
        z = torch.stack((x, y), dim=1)
        min_norm, _ = torch.min(z, dim=1)
        d = torch.norm(code_1 - code_2, dim=1)
        loss = {"code_loss": torch.mean(d / min_norm)}
        return loss
