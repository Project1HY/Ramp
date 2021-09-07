import torch
from geom.mesh.op.gpu.base import batch_vnrmls, batch_fnrmls_fareas, batch_moments
from geom.mesh.op.gpu.dist import batch_l2_pdist
from util.strings import warn


# from chamferdist import ChamferDist
# distpc1topc2, distpc2topc1, idx1, idx2 = chamferDist(pc1, pc2)
# ----------------------------------------------------------------------------------------------------------------------
#                           Full Losses (different architecture might have different losses)
# ----------------------------------------------------------------------------------------------------------------------
class BasicLoss:
    def __init__(self, hp, f):
        self.shape_diff = ShapeDiffLoss(hp, f)

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
        loss_dict_comp = self.shape_diff.compute(shape_1=completion_gt, shape_2=completion_rec, w=1,face_override=face_override)
        # TODO calculate mask: w, w.r.t to mask penalty and distnat vertices (only for completion)
        loss_dict_comp = {f'{k}_comp': v for k, v in loss_dict_comp.items()}

        loss_dict = loss_dict_comp
        loss_dict.update(total_loss=loss_dict['total_loss_comp'])
        return loss_dict


# class TBasedLoss:
#     def __init__(self, hp, f):
#         self.shape_diff = ShapeDiffLoss(hp, f)
#         self.code_loss = CodeLoss()
#
#     def compute(self, x, network_output):
#         # input retrieval
#         completion_gt = x['gt']
#         full = x['tp']
#         part_idx = x['gt_mask']
#
#         # output retrieval
#         completion_rec = network_output['completion_xyz']
#         part_rec = network_output['part_rec']
#         full_rec = network_output['full_rec']
#         gt_rec = network_output['gt_rec']
#         comp_code = network_output['comp_code']
#         gt_code = network_output['gt_code']
#
#         # weights calculation
#         nv = completion_gt.shape[1]
#         w_part = self.shape_diff._mask_part_weight(part_idx, nv)
#
#         loss_dict_comp = self.shape_diff.compute(completion_gt, completion_rec, w=1)
#         # TODO calculate mask: w, w.r.t to mask penalty (only for completion)
#         loss_dict_part = self.shape_diff.compute(completion_gt, part_rec, w=w_part)
#         loss_dict_full = self.shape_diff.compute(full, full_rec, w=1)
#         loss_dict_gt = self.shape_diff.compute(gt_rec, completion_rec, w=1)
#         # bring completion_rec close to gt_rec (gt_rec always better than completion_rec in template based decoder)
#
#         loss_comp = {f'{k}_comp': v for k, v in loss_dict_comp.items()}
#         loss_part = {f'{k}_part': v for k, v in loss_dict_part.items()}
#         loss_full = {f'{k}_full': v for k, v in loss_dict_full.items()}
#         loss_gt = {f'{k}_gt': v for k, v in loss_dict_gt.items()}
#
#         loss_code = self.code_loss.compute(comp_code, gt_code)
#
#         loss_dict = loss_comp
#         loss_dict.update(loss_part)
#         loss_dict.update(loss_full)
#         loss_dict.update(loss_gt)
#         loss_dict.update(loss_code)
#         loss_dict.update(total_loss=loss_dict['total_loss_comp'] + loss_dict['total_loss_part'] +
#                                     loss_dict['total_loss_full'] + loss_dict['total_loss_gt'])
#         return loss_dict


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

        # For CUDA:
        self.dev = hp.dev  # TODO - Might be problematic for multiple GPUs
        self.non_blocking = hp.NON_BLOCKING
        self.def_prec = getattr(torch, hp.UNIVERSAL_PRECISION)

        # Handle Faces:
        if f is not None and (self.lambdas[1] > 0 or self.lambdas[4] > 0 or self.lambdas[5] > 0):
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

    def compute(self, shape_1, shape_2, w,face_override=False):
        """
        :param shape_1: first batch of shapes [b x nv x d] - for which all fields are known
        :param shape_2: second batch shapes [b x nv x 3] - for which only xyz fields are known
        :param w: a set of weighting functions over the vertices
        :return: The loss, as dictionary
        """
        loss = torch.zeros(1, device=self.dev, dtype=self.def_prec)
        loss_dict = {}
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
                    pass
                # TODO: implement chamfer distance loss
                else:
                    raise AssertionError

        loss_dict['total_loss'] = loss
        return loss_dict

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
