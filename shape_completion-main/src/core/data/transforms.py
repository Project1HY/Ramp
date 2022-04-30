import random
import numbers
from itertools import repeat
from collections.abc import Sequence
from geom.mesh.op.cpu.base import vertex_normals, vertex_moments, \
    padded_part_by_mask, flip_vertex_mask, normalize_to_l2_ball, estimate_vertex_normals
from geom.mesh.op.cpu.descriptor import heat_kernel_signature
from geom.matrix.cpu import random_sign_vector
import numpy as np
import math

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Transforms- Abstract
# ----------------------------------------------------------------------------------------------------------------------
from geom.mesh.vis.base import plot_mesh_montage


class Transform:
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PostCompilerTransform(Transform):
    pass


class PreCompilerTransform(Transform):
    pass


class SystemUtilTransform(Transform):
    pass


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def append(self, new_transform):
        self.transforms.append(new_transform)

    def insert(self, index, new_transform):
        self.transforms.insert(index, new_transform)

    def __repr__(self):
        string_t = [str(t) for t in self.transforms]
        return '(' + ",".join(string_t) + ')'


# ----------------------------------------------------------------------------------------------------------------------
#                                               Special Transforms
# ----------------------------------------------------------------------------------------------------------------------

class AlignChannels(SystemUtilTransform):
    def __init__(self, keys, n_channels):
        self.keys = keys
        self.n_channels = n_channels

    def __call__(self, x):
        for k in self.keys:
            x[k] = align_channels(x[k], x[f'{k}_f'], self.n_channels)
        return x


class PartCompiler(SystemUtilTransform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, x):
        # Done last, since we might transform the mask
        for (part_key, mask_key, full_key) in self.keys:
            x[part_key] = padded_part_by_mask(x[mask_key], x[full_key])
        return x


class RemoveFaces(SystemUtilTransform):
    def __call__(self, x):
        x.pop('gt_f', None)
        x.pop('tp_f', None)
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Normalization Transforms
# ----------------------------------------------------------------------------------------------------------------------

class AppendHKS(PreCompilerTransform):
    def __init__(self, keys=('gt', 'tp'), t=(5e-3, 1e1, 10), k=200):
        self._keys = keys
        self._t = t
        self._k = k

    def __call__(self, x):
        for k in self._keys:
            x[f'{k}_hks'] = heat_kernel_signature(x[k][:, 0:3], x[f'{k}_f'], t=self._t, k=self._k)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self._keys},t={self._t},k={self._k})'


class NormalizeScale(PreCompilerTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self, slicer=slice(0, 3), keys=('gt', 'tp')):
        self._slicer = slicer
        self._keys = keys
        self.center = Center(slicer, keys)

    def __call__(self, x):
        x = self.center(x)

        for k in self._keys:
            x[k] *= (1 / x[k].abs().max()) * 0.999999

        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(channels={self._slicer},keys={self._keys})'


class Center(PreCompilerTransform):
    def __init__(self,seg_manager, slicer=slice(0, 3), keys=('gt', 'tp')):
        self._slicer = slicer
        self._keys = keys

    def __call__(self, x):
        assert False,f"x is {x}"
        for k in self._keys:
            center_offset = x[k][:, self._slicer].mean(axis=0, keepdims=True)
            x[k][:, self._slicer] -= center_offset
            if k == 'gt' and 'gt_world_joints' in x.keys():
                joint_trans = []
                for trans in x['gt_world_joints']:
                    trans[0:3, 3] -= center_offset.squeeze()
                    joint_trans += [trans]
                x['gt_world_joints'] = np.array(joint_trans)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(channels={self._slicer},keys={self._keys})'

class CenterTorso(PreCompilerTransform):
    def __init__(self,seg_manager ,slicer=slice(0, 3),keys=('gt', 'tp')):
        self._slicer = slicer
        self._keys = keys
        self.seg_manager = seg_manager
        self.segmentation = self.seg_manager.get_vertex_segs()['Torso']
    def __call__(self, x):
        for k in self._keys:
            com_copy = x[k][:,self._slicer]
            com_copy = com_copy[self.segmentation]
            center_offset = x[k][:, self._slicer].mean(axis=0, keepdims=True)
            x[k][:, self._slicer] -= center_offset
        return x        

    def __repr__(self):
        return self.__class__.__name__ + f'(channels={self._slicer},keys={self._keys})'


class L2BallNormalize(PreCompilerTransform):
    def __init__(self, keys=('gt', 'tp')):
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            x[k][:, 0:3] = normalize_to_l2_ball(x[k][:, 0:3])
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self._keys})'


class UniformVertexScale(PreCompilerTransform):
    def __init__(self, scale, keys=('gt', 'tp')):
        self._scale = scale
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            x[k][:, 0:3] = x[k][:, 0:3] * self._scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(scale={self._scale},keys={self._keys})'


class AlignPose(PreCompilerTransform):
    pass
    # TODO - using PCA


# ----------------------------------------------------------------------------------------------------------------------
#                                        Data Augmentation Transforms - Post Compiler
# ----------------------------------------------------------------------------------------------------------------------
class RandomizeNormalDirections(PostCompilerTransform):
    def __init__(self, keys=('gt_part',)):
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            assert x[k].shape[1] >= 6, "Could not find normal field"
            rs = random_sign_vector(x[k].shape[0])[:, np.newaxis]
            x[k][:, 3:6] *= rs
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self._keys})'


class ReplaceNormalsWithApproximation(PostCompilerTransform):
    def __init__(self, keys=('gt_part',), k=7, smoothing_iter=0):  # Params as taken from the grid search for scans
        self._keys = keys
        self._smoothing_iter = smoothing_iter
        self._k = k

    def __call__(self, x):
        for k in self._keys:
            dims = x[k].shape[1]
            v = x[k][0:3]
            if dims > 6:
                #Oshri Patch to estimate normalson up-padded vertices
                #x[part_key] = padded_part_by_mask(x[mask_key], x[full_key])
                x[k] = np.concatenate([v, estimate_vertex_normals(v), x[k][6:]], axis=1)
            else:
                x[k] = np.concatenate([v, estimate_vertex_normals(v)], axis=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self._keys},k={self._k},smoothing_iter={self._smoothing_iter})'


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Transforms
# ----------------------------------------------------------------------------------------------------------------------

class RandomMaskFlip(PreCompilerTransform):
    def __init__(self, prob, keys=('gt',)):  # Probability of mask flip
        self._prob = prob
        self._keys = keys
        self._mask_keys = [k + '_mask' for k in keys]  # TODO - Review for more than one mask

    def __call__(self, x):
        for (k, mk) in zip(self._keys, self._mask_keys):
            if random.random() < self._prob:
                nv = x[k].shape[0]
                x[mk] = flip_vertex_mask(nv, x[mk])
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self._prob},keys={self._keys})'


class RandomMaskDecimation(PreCompilerTransform):
    def __init__(self, frac, keys=('gt',)):  # Probability of mask flip
        if isinstance(frac,(tuple,list)):
            self._low = frac[0]
            self._high = frac[1]
        else:
            self._low = 0
            self._high = frac

        self._keys = keys
        self._mask_keys = [k + '_mask' for k in keys]  # TODO - Review for more than one mask

    def __call__(self, x):
        for (k, mk) in zip(self._keys, self._mask_keys):
            num_mask_vi = x[mk].shape[0]
            frac = np.random.uniform(self._low,self._high)
            num_to_remove = int(num_mask_vi * frac)
            assert num_to_remove < num_mask_vi, "Decimated the entire mask - trying upping decimation frac"
            if num_to_remove > 0:
                keepers_i = sorted(np.random.choice(range(num_mask_vi), size=num_mask_vi - num_to_remove, replace=False))
                x[mk] = x[mk][keepers_i]
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(frac=({self._low},{self._high}),keys={self._keys})'


class RandomScale(PreCompilerTransform):
    """
    scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
    is randomly sampled from the range
    """

    def __init__(self, scales, keys=('gt', 'tp')):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self._scales = scales
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            scale = np.random.uniform(*self._scales)
            x[k][:, :3] *= scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(scales={self._scales},keys={self._keys})'


class RandomRotate(PreCompilerTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, axis=0, keys=('gt', 'tp')):
        if not isinstance(degrees, tuple):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        assert axis in [0, 1, 2]
        self._degrees = degrees
        self._axis = axis
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            degree = math.pi * random.uniform(*self._degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)

            if self._axis == 0:
                rot = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            elif self._axis == 1:
                rot = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            else:
                rot = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            x[k][:, :3] = np.matmul(x[k][:, :3], rot)
            if x[k].shape[1] >= 6:
                x[k][:, 3:6] = np.matmul(x[k][:, 3:6], rot)  # Rotate normals as well

    def __repr__(self):
        return self.__class__.__name__ + f'(degrees={self._degrees},axis={self._axis},keys={self._keys})'


class RandomTranslate(PreCompilerTransform):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    WARNING: After this operation, vertex normals are not going to fit- TODO
    """

    def __init__(self, translate, keys=('gt', 'tp')):
        self._translate = translate
        self._keys = keys

    def __call__(self, x):
        for k in self._keys:
            nv, t = x[k].shape[0], self._translate
            if isinstance(t, numbers.Number):
                t = list(repeat(t, times=3))
            assert len(t) == 3

            ts = []
            for d in range(3):
                ts.append(np.random.uniform(low=-abs(t[d]), high=abs(t[d]), size=(nv,)))
                # ts.append(x[k].empty_like(n).uniform_(-abs(t[d]), abs(t[d])))

            x[k][:, :3] += np.stack(ts, axis=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(translate={self._translate},keys={self._keys})'


class RandomGaussianNoise(PreCompilerTransform):
    # WARNING: After this operation, vertex normals are not going to fit- TODO
    # STD - Either a range or a float
    # Output keys must be same as keys or completely new
    def __init__(self, std, keys=('gt',), okeys=('gt',), slicer=slice(0, 3)):
        if isinstance(std, Sequence):
            std = tuple(std)
        self._std = std
        self._keys = keys
        self._okeys = okeys
        self._slicer = slicer

    def __call__(self, x):
        for ok, k in zip(self._okeys, self._keys):
            arr = x[k][:, self._slicer]
            std = np.random.uniform(low=self._std[0], high=self._std[1]) if isinstance(self._std, tuple) else self._std
            noise = (std * np.random.standard_normal(arr.shape)).astype(arr.dtype)
            if ok == k:
                x[k][:, self._slicer] = arr + noise
            else:
                x[ok] = arr + noise
        return x

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(std={self._std},keys={self._keys},okeys={self._okeys},slicer={self._slicer})'


class RandomInputDropout(PreCompilerTransform):
    # TODO - Correct this
    def __init__(self, max_dropout_ratio=0.875):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return pc


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Utils
# ----------------------------------------------------------------------------------------------------------------------

def align_channels(v, f, req_in_channels):
    available_in_channels = v.shape[1]
    if available_in_channels > req_in_channels:
        return v[:, 0:req_in_channels]
    else:
        combined = [v]
        if req_in_channels >= 6 > available_in_channels:
            combined.append(vertex_normals(v, f))
        if req_in_channels >= 12 > available_in_channels:
            combined.append(vertex_moments(v))

        return np.concatenate(combined, axis=1)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def _test_transforms():
    from data.sets import DatasetMenu
    from geom.mesh.vis.base import plot_mesh
    ds = DatasetMenu.order('FaustPyProj')
    # single_ldr = ds.loaders(s_nums=1000, s_shuffle=True,
    #                         s_transform=[RandomGaussianNoise((0, 0.05), okeys=('gt_noise',))],
    #                         n_channels=6, method='rand_f2f', batch_size=1, device='cpu-single')
    single_ldr = ds.loaders(s_nums=1000, s_shuffle=True, s_transform=[AppendHKS()], n_channels=6, method='rand_f2p',
                            batch_size=1,
                            device='cpu-single')

    for dp in single_ldr:
        dp['gt'] = dp['gt'].squeeze()
        gt = dp['gt']
        # mask = dp['gt_mask'][0]
        # gt_part = gt[mask, :]
        trans = RandomTranslate(0.01, keys=['gt'])
        # print(trans)
        # v = gt_part[:, :3]
        # n = gt_part[:, 3:6]
        # _, f = trunc_to_vertex_mask(gt[:, :3], ds.faces(), mask)
        plot_mesh_montage(vb=[gt[:, :3], dp['gt_noise']], fb=ds.faces(), strategy='spheres')
        # dp = trans(dp)
        v = gt[:, :3]
        n = gt[:, 3:6]
        plot_mesh(v=v, n=n, f=ds.faces(), strategy='mesh')
        break


if __name__ == '__main__':
    _test_transforms()
