import re
import sys
from abc import ABC
from copy import deepcopy
from pathlib import Path

import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from tqdm import tqdm

from data.transforms import *
from geom.mesh.io.base import read_mesh, read_npz_mask
from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask
from collections.abc import Mapping
from geom.mesh.vis.base import plot_mesh
from util.container import split_frac, to_list, list_dup
from util.fs import pkl_load
from util.func import handles_scalars
from util.strings import warn, banner
from util.time import time_me
from util.torch.data import determine_worker_num, ReconstructableLoader, ParametricLoader, SubsetChoiceSampler

# from torch.utils.data.distributed import DistributedSampler
NP_STR_OBJ_ARRAY_PATTERN = re.compile(r'[SaUO]')


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Skinning
# ----------------------------------------------------------------------------------------------------------------------

class HomogenicMatCfg:
    NONE = 0  # No matrix
    ORIGINAL = 1  # The entire matrix, 4x4
    TRUNC = 2  # The truncated matrix, 3x4, removing the degenerate homogenic row
    ONLY_ROTATION = 3  # Only the rotation matrix
    ONLY_TRANSLATION = 4  # Only the translation vector


class SkinningConfiguration:
    def __init__(self, gt_world_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                 gt_local_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                 template_world_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                 template_local_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                 need_inverse_bind_mats=HomogenicMatCfg.ORIGINAL,
                 need_skinning_weights=True, need_rest_pose_vertices=True, need_part=True):
        """
        For each of the following, insert an integer from HomogenicMatCfg. If NONE is used, no
        matrices are supplied by the loader.

        Expected sizes:
            [batch_size x num_joints x 4x4] for ORIGINAL
            [batch_size x num_joints x 3x4] for TRUNC
            [batch_size x num_joints x 3x3] for ONLY_ROTATION
            [batch_size x num_joints x 3] for ONLY_TRANSLATION

        :param int gt_world_joint_mats_cfg: The world matrices for each joint on the ground truth shape
        :param int gt_local_joint_mats_cfg: The local matrices for each joint on the ground truth shape
        :param int template_world_joint_mats_cfg: The world matrices for each joint on the template shape
        :param int template_local_joint_mats_cfg: The local matrices for each joint on the template truth shape
        :param int need_inverse_bind_mats: The inverse binding matrices for each joint on the bind shape

        # Note: we currently presume that the template and ground truth stem from the same subject, i.e., then have the
        same inverse binding matrices, skinning weights, rest pose vertices etc.
        # At least for the MIXAMO datasets, the joint tree structure is *exactly* the same for all subjects.

        :param bool need_skinning_weights:
            Whether to return the skinning weights for template & ground truth
            Expected size: [batch_size x num_joints x num_vertices]

        :param bool need_rest_pose_vertices:
            Whether to return the rest post vertices.
            Expected size: [batch_size x num_vertices x 3]
        """
        self.need_gt_joint_pickle = gt_world_joint_mats_cfg or gt_local_joint_mats_cfg
        self.need_tp_joint_pickle = template_world_joint_mats_cfg or template_local_joint_mats_cfg

        # Save variables:
        self.inv_bind_mats_cfg = need_inverse_bind_mats
        self.gt_global_joint_cfg = gt_world_joint_mats_cfg
        self.gt_local_joint_cfg = gt_local_joint_mats_cfg
        self.tp_global_joint_cfg = template_world_joint_mats_cfg
        self.tp_local_joint_cfg = template_local_joint_mats_cfg

        self.need_skinning_weights = need_skinning_weights
        self.need_res_pose_verts = need_rest_pose_vertices
        self.need_part = need_part


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Deformations
# ----------------------------------------------------------------------------------------------------------------------
class Deformation:
    def __init__(self, n_expected_proj):
        self._n_expected_proj = n_expected_proj

    def num_projections(self):
        return self._n_expected_proj

    def __repr__(self):
        return f'{self.__class__.__name__}(proj={self._n_expected_proj})'

    @staticmethod
    def dataset_name_suffix():
        raise NotImplementedError

    @staticmethod
    def name():
        raise NotImplementedError


class AzimuthalProjection(Deformation):
    def __init__(self, n_expected_proj=10):
        super().__init__(n_expected_proj)

    @staticmethod
    def dataset_name_suffix():
        return 'Proj'

    @staticmethod
    def name():
        return 'azimuthal_projections'


class SemanticCut(Deformation):
    def __init__(self, n_expected_proj=10):
        super().__init__(n_expected_proj)

    @staticmethod
    def dataset_name_suffix():
        return 'SemCuts'

    @staticmethod
    def name():
        return 'semantic_cuts'


class OrbitalProjection(Deformation):
    def __init__(self, n_expected_proj=110):
        super().__init__(n_expected_proj)

    @staticmethod
    def dataset_name_suffix():
        return 'EleProj'

    @staticmethod
    def name():
        return 'orbital_projections'


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class HitIndexedDataset(ABC):
    def __init__(self):
        # Construct the hit
        self._hit = self._construct_hit()

    def data_summary(self, with_tree=False):
        banner('Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Number of Indexed Elements: {self.num_indexed_files()}')
        print(f'* Direct Filepath: {self.data_directory()}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()

    def data_directory(self):
        raise NotImplementedError

    def report_index_tree(self):
        print(self._hit)

    def name(self):
        return self.__class__.__name__

    def num_indexed_files(self):
        return self._hit.num_indexed()

    def validate_dataset(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError

    def show_sample(self, num_samples):
        raise NotImplementedError

    def loaders(self):
        raise NotImplementedError

    def _construct_hit(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionDataset(HitIndexedDataset, ABC):
    DEFINED_SAMP_METHODS = ('full', 'part', 'f2p', 'rand_f2p', 'frand_f2p', 'p2p', 'rand_p2p', 'frand_p2p',
                            'rand_ff2p', 'rand_ff2pp', 'rand_f2p_seq', 'rand_f2f', 'const_tp_rand_f2p')
    LOADER_CLASS = ReconstructableLoader

    @classmethod
    def defined_methods(cls):
        return cls.DEFINED_SAMP_METHODS

    def name(self, short=False):
        if short:
            return self.__class__.__name__
        else:
            return f'{self.__class__.__name__}{self._deformation.dataset_name_suffix()}'

    def __init__(self, deformation: Deformation, data_dir_override, cls, suspected_corrupt=False):
        # Append Info:
        self._suspected_corrupt, self._deformation = suspected_corrupt, deformation
        self._f, self._n_v = None, None  # Place holder for same-face/same number of vertices - dataset
        from cfg import DANGEROUS_MASK_THRESH, UNIVERSAL_PRECISION
        self._def_precision = getattr(np, UNIVERSAL_PRECISION)
        self._mask_thresh = DANGEROUS_MASK_THRESH

        # Handle Data Directory
        if data_dir_override is None:
            from cfg import PRIMARY_DATA_DIR
            self._data_dir = (PRIMARY_DATA_DIR / cls / self.name(short=True)).resolve()
        else:
            # TODO: change back from this value yiftach
            # self._data_dir = Path(data_dir_override).resolve()
            self._data_dir = Path(data_dir_override)
        # assert self._data_dir.is_dir(), f"Data dir of {self.name()} is invalid: \nCould not find {self._data_dir}"

        # Set all other directories:
        self._proj_dir = self._data_dir / deformation.name()
        self._full_dir = self._data_dir / 'full'
        self._index_dir = self._data_dir / 'index'
        # assert self._proj_dir.is_dir(), f"Projection dir of {self.name()} is invalid: \nCould not find {self._proj_dir}"
        # assert self._full_dir.is_dir(), f"Full Shape dir of {self.name()} is invalid: \nCould not find {self._full_dir}"

        # Deformation Specfic loading support:
        self._hi2proj_path_func = getattr(self, f'_hi2proj_path_{self._deformation.name()}', self._hi2proj_path_default)
        self._hi2full_path_func = getattr(self, f'_hi2full_path_{self._deformation.name()}', self._hi2full_path_default)
        self._proj_path2data_func = getattr(self, f'_proj_path2data_{self._deformation.name()}',
                                            self._proj_path2data_default)
        self._full_path2data_func = getattr(self, f'_full_path2data_{self._deformation.name()}',
                                            self._full_path2data_default)

        # Init the hit
        super().__init__()
        self._hit.init_cluster_hi_list()
        self._hit_in_memory = self._hit.in_memory()
        self._tup_index_map = None  # TODO - Revise index map for datasets that are not in-memory
        assert self._deformation.num_projections() * self.num_full_shapes() == self.num_projections(), \
            "Index coherence error - check the right file is being read"

    def data_directory(self):
        return self._data_dir

    def num_indexed_files(self):  # Override
        return self.num_projections() + self.num_full_shapes()

    def num_projections(self):
        return self._hit.num_indexed()

    def num_full_shapes(self):
        return self._hit.num_index_clusters()

    def subject_names(self):
        return tuple(self._hit._hit.keys())

    def num_datapoints_by_method(self, method):
        assert method in self.DEFINED_SAMP_METHODS
        if method in ['full', 'rand_f2f', 'rand_f2j']:
            return self.num_full_shapes()
        elif method == 'part':
            return self.num_projections()
        elif method == 'f2p' or method == 'p2p':
            assert self._hit_in_memory, "Full tuple indexing will take too much time"  # TODO - Can this be fixed?
            if self._tup_index_map is None:
                self._build_tupled_index()  # TODO - Revise this for P2P
            return len(self._tup_index_map)
        else:
            return self.num_projections()  # This is big enough, but still a lie

    def data_summary(self, with_tree=False):
        v_str = "No" if self._n_v is None else f"Yes :: {self._n_v} vertices"
        f_str = "No" if self._f is None else f"Yes :: {self._f.shape[0]} faces"

        banner('Full-Part Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Deformation Type: {self._deformation.name()} [{self._deformation.num_projections()} per frame]')
        print(f'* Index in memory: {"Yes" if self._hit_in_memory else "No"}')
        print(f'* Number of Full Shapes: {self.num_full_shapes()}')
        print(f'* Number of Projections {self.num_projections()}')
        print(f'* Uniform Face Array: {f_str}')
        print(f'* Uniform Number of Vertices: {v_str}')
        print(f'* Direct Filepath: {self._data_dir}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()

    @time_me
    def validate_dataset(self):
        banner(f'Validation of dataset {self.name()} :: {self.num_projections()} Projections '
               f':: {self.num_full_shapes()} Full Shapes')
        for si in tqdm(range(self.num_projections()), file=sys.stdout, dynamic_ncols=True):
            hi = self._hit.si2hi(si)
            fp = self._hi2proj_path_func(hi)
            try:
                #data = self._proj_path2data_func(fp)
                data = self._full_path2data_default(fp)
            except Exception as e:
                print(f"Missing projection {fp.resolve()} in dataset {self.name()}",e)
        print(f'Projection Validation -- PASSED --')
        for si in tqdm(range(self.num_full_shapes()), file=sys.stdout, dynamic_ncols=True):
            chi = self._hit.csi2chi(si)
            fp = self._hi2full_path_func(chi)
            try:
                #data = self._full_path2data_func(fp)
                data = self._full_path2data_default(fp)
            except Exception as e:
                print(f"Missing full subject {fp.resolve()} in dataset {self.name()}",e)

    def sample(self, num_samples=10, transforms=None, method='f2p', n_channels=6):
        total_points = self.num_datapoints_by_method(method)
        if num_samples > total_points:
            warn(f"Requested {num_samples} samples when dataset only holds {total_points}. "
                 f"Returning the latter")
        ldr = self._loader(ids=None, batch_size=num_samples, device='cpu-single',
                           transforms=transforms, method=method, n_channels=n_channels, set_size=None)
        return next(iter(ldr))

    def show_sample(self, num_samples=4, strategy='mesh', with_vnormals=False, method='f2p'):
        raise NotImplementedError  # TODO

    def loaders(self, s_nums=None, s_shuffle=True, s_transform=None, split=(1,), s_dynamic=False,
                global_shuffle=False, batch_size=16, device='cuda', method='f2p', n_channels=6):
        """
        # s for split
        :param split: A list of fracs summing to 1: e.g.: [0.9,0.1] or [0.8,0.15,0.05]. Don't specify anything for a
        single loader
        :param s_nums: A list of integers: e.g. [1000,50] or [1000,5000,None] - The number of objects to take from each
        range split. If None, it will take the maximal number possible.
        WARNING: Remember that the data loader will ONLY load from these examples unless s_dynamic[i] == True
        :param s_dynamic: A list of booleans: If s_dynamic[i]==True, the ith split will take s_nums[i] examples at
        random from the partition [which usually includes many more examples]. On the next load, will take another
        random s_nums[i] from the partition. If False - will take always the very same examples. Usually, we'd want
        s_dynamic to be True only for the training set.
        :param s_shuffle: A list of booleans: If s_shuffle[i]==True, the ith split will be shuffled before truncations
        to s_nums[i] objects
        :param s_transform: A list - s_transforms[i] is the transforms for the ith split
        :param global_shuffle: If True, shuffles the entire set before split
        :param batch_size: Integer > 0
        :param device: 'cuda' or 'cpu' or 'cpu-single' or pytorch device
        :param method: One of ('full', 'part', 'f2p', 'rand_f2p','frand_f2p', 'p2p', 'rand_p2p','frand_p2p')
        :param n_channels: One of cfg.SUPPORTED_IN_CHANNELS - The number of channels required per datapoint
        :return: A list of (loaders,num_samples)
        """
        # Handle inpput arguments:
        s_shuffle = to_list(s_shuffle)
        s_dynamic = to_list(s_dynamic)
        s_nums = to_list(s_nums)
        if s_transform is None or not s_transform:
            s_transform = [None] * len(split)
        elif not isinstance(s_transform[0], Sequence):
            s_transform = list_dup(s_transform, len(split))
            # Transforms must be a list, all others are non-Sequence
        assert sum(split) == 1, "Split fracs must sum to 1"
        # TODO - Clean up this function, add in smarter defaults, simplify
        if (method == 'f2p' or method == 'p2p') and not self._hit_in_memory:
            method = 'rand_' + method
            warn(f'Tuple dataset index is too big for this dataset. Reverting to {method} instead')
        if (method == 'frand_f2p' or method == 'frand_p2p') and len(split) != 1:
            raise ValueError("Seeing the fully-rand methods have no connection to the partition, we cannot support "
                             "a split dataset here")
        # Logic:
        ids = list(range(self.num_datapoints_by_method(method)))
        if global_shuffle:
            np.random.shuffle(ids)  # Mixes up the whole set

        n_parts = len(split)
        ids = split_frac(ids, split)
        loaders = []
        for i in range(n_parts):
            set_ids, req_set_size, do_shuffle, transforms, is_dynamic = ids[i], s_nums[i], \
                                                                        s_shuffle[i], s_transform[i], s_dynamic[i]
            if req_set_size is None:
                req_set_size = len(set_ids)
            eff_set_size = min(len(set_ids), req_set_size)
            if eff_set_size != req_set_size:
                warn(f'At Loader {i + 1}/{n_parts}: Requested {req_set_size} objects while set has only {eff_set_size}.'
                     f' Reverting to latter')
            if do_shuffle:
                np.random.shuffle(set_ids)  # Truncated sets may now hold different ids
            if not is_dynamic:  # Truncate only if not dynamic
                set_ids = set_ids[:eff_set_size]  # Truncate
            recon_stats = {
                'dataset_name': self.name(),
                'batch_size': batch_size,
                'split': split,
                'id_in_split': i,
                'set_size': eff_set_size,
                'transforms': str(transforms),
                'global_shuffle': global_shuffle,
                'partition_shuffle': do_shuffle,
                'method': method,
                'n_channels': n_channels,
                'in_memory_index': self._hit_in_memory,
                'is_dynamic': is_dynamic
            }

            ldr = self._loader(method=method, transforms=transforms, n_channels=n_channels, ids=set_ids,
                               batch_size=batch_size, device=device, set_size=eff_set_size)
            ldr.init_recon_table(recon_stats)
            loaders.append(ldr)

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    def _loader(self, method, transforms, n_channels, ids, batch_size, device, set_size):

        # Handle Device:
        device = str(device).split(':')[0]  # Compatible for both strings & pytorch devs
        assert device in ['cuda', 'cpu', 'cpu-single']
        pin_memory = (device == 'cuda')
        if device == 'cpu-single':
            n_workers = 0
        else:
            n_workers = determine_worker_num(len(ids), batch_size)

        # Compile Sampler:
        if ids is None:
            ids = range(self.num_datapoints_by_method(method))
        if set_size is None:
            set_size = len(ids)
        assert len(ids) > 0, "Found loader with no data samples inside"
        sampler_length = min(set_size, len(ids))  # Allows for dynamic partitions
        # if device == 'ddp': #TODO - Add distributed support here. What does num_workers need to be?
        # data_sampler == DistributedSampler(dataset,num_replicas=self.num_gpus,ranks=self.logger.rank)

        #SequentialSampler
        data_sampler = SubsetChoiceSampler(ids, sampler_length)
        #data_sampler = SequentialSampler(ids)

        # Compiler Transforms:
        transforms = self._transformation_finalizer_by_method(method, transforms, n_channels)

        return self.LOADER_CLASS(FullPartTorchDataset(self, transforms, method), batch_size=batch_size,
                                 sampler=data_sampler, num_workers=n_workers, pin_memory=pin_memory,
                                 collate_fn=completion_collate, drop_last=True)

    def _datapoint_via_full(self, csi):
        return self._full_dict_by_hi(self._hit.csi2chi(csi))

    def _full_dict_by_hi(self, hi):
        v = self._full_path2data_func(self._hi2full_path_func(hi))
        if isinstance(v, tuple):
            v, f = v[0], v[1]
        else:
            f = self._f
        v = v.astype(self._def_precision)
        return {'gt_hi': hi, 'gt': v, 'gt_f': f}

    def _mask_by_hi(self, hi):
        mask = self._proj_path2data_func(self._hi2proj_path_func(hi))
        if len(mask) < self._mask_thresh:
            warn(f'Found mask of length {len(mask)} with id: {hi}')
        return mask

    def _datapoint_via_part(self, si):
        hi = self._hit.si2hi(si)
        d = self._full_dict_by_hi(hi)
        d['gt_mask'] = self._mask_by_hi(hi)
        return d

    # @time_me
    def _build_tupled_index(self):
        # TODO - Revise index map for datasets that are not in-memory
        tup_index_map = []
        for i in range(self.num_projections()):
            for j in range(self.num_full_shapes()):
                if self._hit.csi2chi(j)[0] == self._hit.si2hi(i)[0]:  # Same subject
                    tup_index_map.append((i, j))
        self._tup_index_map = tup_index_map
        # print(convert_bytes(sys.getsizeof(tup_index_map)))

    def _tupled_index_map(self, si):
        return self._tup_index_map[si]

    def _datapoint_via_f2p(self, si):
        si_gt, si_tp = self._tupled_index_map(si)
        tp_dict = self._datapoint_via_full(si_tp)
        gt_dict = self._datapoint_via_part(si_gt)
        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
        return gt_dict

    def _datapoint_via_rand_f2f(self, si):
        gt_dict = self._datapoint_via_full(si)
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
        return gt_dict

    def _datapoint_via_frand_f2p(self, _):
        # gt_dict = self._datapoint_via_part(si)  # si is gt_si
        gt_hi = self._hit.random_path_from_partial_path()
        gt_dict = self._full_dict_by_hi(gt_hi)
        gt_dict['gt_mask'] = self._mask_by_hi(gt_hi)
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
        return gt_dict

    def _datapoint_via_rand_f2p(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        # tp_hi = gt_dict['gt_hi'][:-1]
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
        return gt_dict

    def _datapoint_via_rand_f2p_seq(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path(gt_dict['gt_hi'][:2])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'], tp_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
        return gt_dict

    def _datapoint_via_rand_ff2p(self, si):
        # TODO - revise for scans
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp1'], gt_dict['tp1_hi'] = tp_dict['gt'], tp_dict['gt_hi']

        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp2'], gt_dict['tp2_hi'] = tp_dict['gt'], tp_dict['gt_hi']

        return gt_dict

    def _datapoint_via_rand_ff2pp(self, si):
        # TODO - revise for scans
        ff2p_dict = self._datapoint_via_rand_ff2p(si)
        # Change gt_mask -> gt_mask1, gt_hi->gt_hi1
        ff2p_dict['gt_mask1'] = ff2p_dict['gt_mask']
        ff2p_dict['gt_hi1'] = ff2p_dict['gt_hi']
        del ff2p_dict['gt_mask'], ff2p_dict['gt_hi']
        # Add in another mask:
        gt_hi2 = self._hit.random_path_from_partial_path(ff2p_dict['gt_hi1'][:-1])  # All but proj id
        ff2p_dict['gt_mask2'] = self._mask_by_hi(gt_hi2)
        ff2p_dict['gt_hi2'] = gt_hi2
        return ff2p_dict

    def _datapoint_via_p2p(self, si):
        # TODO - revise for scans
        si_gt, si_tp = self._tupled_index_map(si)
        tp_dict = self._datapoint_via_part(si_tp)
        gt_dict = self._datapoint_via_part(si_gt)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    def _datapoint_via_rand_p2p(self, si):
        # TODO - revise for scans
        gt_dict = self._datapoint_via_part(si)  # si is the gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])
        tp_dict = self._full_dict_by_hi(tp_hi)
        tp_dict['gt_mask'] = self._mask_by_hi(tp_hi)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    def _datapoint_via_frand_p2p(self, _):
        # TODO - revise for scans
        # gt_dict = self._datapoint_via_part(si)  # si is the gt_si
        gt_hi = self._hit.random_path_from_partial_path()
        gt_dict = self._full_dict_by_hi(gt_hi)
        gt_dict['gt_mask'] = self._mask_by_hi(gt_hi)
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])
        tp_dict = self._full_dict_by_hi(tp_hi)
        tp_dict['gt_mask'] = self._mask_by_hi(tp_hi)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    # noinspection PyTypeChecker
    def _transformation_finalizer_by_method(self, method, transforms, n_channels):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, list):
            transforms = [transforms]
            for t in transforms:
                assert isinstance(t, Transform)

        if method == 'full' or method == 'rand_f2j':
            align_keys, compiler_keys = ['gt'], None
        elif method == 'part' or method == 'rand_f2jp':
            align_keys, compiler_keys = ['gt'], [['gt_part', 'gt_mask', 'gt']]
        elif method == 'rand_f2f':
            align_keys, compiler_keys = ['gt', 'tp'], None
        elif method in ['f2p', 'rand_f2p', 'frand_f2p', 'rand_f2p_seq', 'const_tp_rand_f2p']: # TODO
            align_keys, compiler_keys = ['gt', 'tp'], [['gt_part', 'gt_mask', 'gt']]
        elif method in ['p2p', 'rand_p2p', 'frand_p2p']:
            align_keys, compiler_keys = ['gt', 'tp'], [['gt_part', 'gt_mask', 'gt'], ['tp_part', 'tp_mask', 'tp']]
        elif method == 'rand_ff2p':
            align_keys, compiler_keys = ['gt', 'tp1', 'tp2'], [['gt_part', 'gt_mask', 'gt']]
        elif method == 'rand_ff2pp':
            align_keys, compiler_keys = ['gt', 'tp1', 'tp2'], \
                                        [['gt_part1', 'gt_mask1', 'gt'], ['gt_part2', 'gt_mask2', 'gt']]
        else:
            raise AssertionError

        if compiler_keys is None:
            ordered_transforms = [AlignChannels(keys=align_keys, n_channels=n_channels)] + transforms
        else:
            ordered_transforms = [AlignChannels(keys=align_keys, n_channels=n_channels)] + \
                                 [t for t in transforms if isinstance(t, PreCompilerTransform)] + \
                                 [PartCompiler(compiler_keys)] + \
                                 [t for t in transforms if isinstance(t, PostCompilerTransform)]

        if self._f is not None:
            ordered_transforms.append(RemoveFaces())

        return Compose(ordered_transforms)

    def full_shape_by_si(self, si, verts_only=True):
        chi = self._hit.csi2chi(si)
        return self.full_shape_by_hi(chi, verts_only)

    def full_shape_by_hi(self, hi, verts_only=True):
        fp = self._hi2full_path_func(hi)
        return read_mesh(fp, verts_only=verts_only)

    def deformation_mask_by_hi(self, hi):
        return self._proj_path2data_func(self._hi2proj_path_func(hi))

    def partial_shape_by_hi(self, hi, verts_only=True):
        full_fp = self._hi2full_path_func(hi)
        v, f = read_mesh(full_fp)
        mask = self._proj_path2data_func(self._hi2proj_path_func(hi))
        if verts_only:
            return v[mask, :]
        else:
            return trunc_to_vertex_mask(v, f, mask)

    def full_shape_iterator(self):
        for si in range(self.num_full_shapes()):
            chi = self._hit.csi2chi(si)
            fp = self._hi2full_path_func(chi)
            yield self._full_path2data_func(fp)

    def _construct_hit(self):
        # Default Implementation
        # _index_dir
        # self._index_dir = "~/mnt/Mano/data/DFaust/DFaust/"
        #TODO: change back yiftach
        import pathlib
        hit_fp = list([r"/Users/yiftachedelstain/Development/Technion/Project/Ramp/shape_completion-main/DFaust_azimuthal_projections_hit.pkl"])
        if len(hit_fp) != 1:
            raise AssertionError(f"Could not find hit file in for deformation {self._deformation.name()} "
                                 f"in index directory:\n{self._index_dir}")
        else:
            return pkl_load(hit_fp[0])

    def _hi2proj_path_default(self, hi):
        raise NotImplementedError

    def _hi2full_path_default(self, hi):
        raise NotImplementedError

    def _proj_path2data_default(self, fp):
        return read_npz_mask(fp)

    def _full_path2data_default(self, fp):
        return read_mesh(fp)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class FullPartTorchDataset(Dataset):
    SAME_SHAPE_RETRY_BOUND = 3
    RETRY_BOUND = SAME_SHAPE_RETRY_BOUND * 7  # Try to retry atleast 7 shapes before dying

    # Note that changes to Dataset will be seen in any loader derived from it before
    # This should be taken into account when decimating the Dataset index
    def __init__(self, ds_inst, transforms, method):
        self._ds_inst = ds_inst
        self._transforms = transforms
        self._method = method
        self.get_func = getattr(self._ds_inst, f'_datapoint_via_{method}')
        self.self_len = self._ds_inst.num_datapoints_by_method(self._method)
        self.use_unsafe_meth = not self._ds_inst._suspected_corrupt

    def __len__(self):
        return self.self_len

    def __getitem__(self, si):
        if self.use_unsafe_meth:
            return self._transforms(self.get_func(si))
        else:  # This is a hack to enable reloading
            global_retries = 0
            local_retries = 0
            while 1:
                try:
                    return self._transforms(self.get_func(si))
                except Exception as e:
                    global_retries += 1
                    local_retries += 1
                    if global_retries == self.RETRY_BOUND:
                        raise e
                    if local_retries == self.SAME_SHAPE_RETRY_BOUND:
                        local_retries = 0
                        si += 1  # TODO - Find better way
                        if si == self.self_len:  # Double back
                            si = 0

class FullPartSequentialTorchDataset(Dataset):
    SAME_SHAPE_RETRY_BOUND = 3
    RETRY_BOUND = SAME_SHAPE_RETRY_BOUND * 7  # Try to retry atleast 7 shapes before dying

    # Note that changes to Dataset will be seen in any loader derived from it before
    # This should be taken into account when decimating the Dataset index
    def __init__(self, ds_inst, transforms, method):
        self._ds_inst = ds_inst
        self._transforms = transforms
        self._method = method
        self.get_func = getattr(self._ds_inst, f'_datapoint_via_path_tup')
        self.self_len = self._ds_inst.num_datapoints_by_method(self._method)
        self.use_unsafe_meth = not self._ds_inst._suspected_corrupt

    def __len__(self):
        return self.self_len

    def __getitem__(self, pose_tup):
        if self.use_unsafe_meth:
            return self._transforms(self.get_func(pose_tup))
        else:  # This is a hack to enable reloading
            global_retries = 0
            local_retries = 0
            while 1:
                try:
                    return self._transforms(self.get_func(si))
                except Exception as e:
                    global_retries += 1
                    local_retries += 1
                    if global_retries == self.RETRY_BOUND:
                        raise e
                    if local_retries == self.SAME_SHAPE_RETRY_BOUND:
                        local_retries = 0
                        si += 1  # TODO - Find better way
                        if si == self.self_len:  # Double back
                            si = 0

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class ParametricCompletionDataset(CompletionDataset, ABC):
    NULL_SHAPE_SI = 0
    LOADER_CLASS = ParametricLoader

    # This adds the assumption that each mesh has the same connectivity, and the same number of vertices
    def __init__(self, n_verts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #TODO yiftach change back
        self._f = pkl_load("/Users/yiftachedelstain/Development/Technion/Project/Ramp/shape_completion-main/face_template.pkl")
        # self._f.flags.writeable = False  # TODO - Turned this off due to a PyTorch warning on tensor support

        self._n_v = n_verts
        self._null_shape, self._num_pose_per_sub = None, None
        self._null_shape_hi = self._hit.csi2chi(self.NULL_SHAPE_SI)

    def faces(self):
        return self._f

    def num_verts(self):
        return self._n_v

    def num_faces(self):
        return self._f.shape[0]

    def null_shape_hi(self):
        return self._null_shape_hi

    def null_shape(self, n_channels=3):
        # Cache shape:
        if self._null_shape is None or self._null_shape.shape[1] != n_channels:
            self._null_shape = align_channels(self._datapoint_via_full(self.NULL_SHAPE_SI)['gt'], self._f, n_channels)
            self._null_shape.flags.writeable = False

        return self._null_shape

    @handles_scalars
    def full_shape_distance_mat(self, statistics=(np.mean, np.max)):

        n = self.num_full_shapes()
        stat_mats = [np.zeros((n, n)) for _ in range(len(statistics))]
        for i in tqdm(range(1, n), file=sys.stdout, desc='Computing distance statistics'):
            full_shape_i = self.full_shape_by_si(i)
            for j in range(i):
                full_shape_j = self.full_shape_by_si(j)
                for stat_i, stat in enumerate(statistics):
                    stat_mats[stat_i][i, j] = stat(np.abs(full_shape_i - full_shape_j))

        for mat in stat_mats:
            mat += mat.transpose()
        return stat_mats

    def _datapoint_via_const_tp_rand_f2p(self, si):
        gt_dict = self._datapoint_via_part(si)
        gt_dict['tp'], gt_dict['tp_hi'] = self.null_shape(), self.null_shape_hi()
        return gt_dict

    def plot_null_shape(self, strategy='mesh', with_vnormals=False):
        null_shape = self.null_shape(n_channels=6)
        n = null_shape[:, 3:6] if with_vnormals else None
        plot_mesh(v=null_shape[:, :3], f=self._f, n=n, strategy=strategy)

    def _full_path2data_default(self, fp):
        return read_mesh(fp, verts_only=True)




# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class ParametricSkinningDataset(ParametricCompletionDataset, ABC):
    DEFINED_SAMP_METHODS = ('rand_f2j', 'rand_f2jp', 'rand_f2p', 'f2p', 'skinning')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta_data_fp = self._index_dir / 'subject_joint_meta'
        import pickle
        joints_metadata = pickle.load(open(meta_data_fp / 'joints_metadata.pkl', 'rb'))
        self.joints_order = joints_metadata['joints_orders']

        self._orig_j2i, self._n_j, self._directed_joint_edges, self._directed_joint_adj = [None] * 4
        self._subject_meta_data = {n: self._preprocess_meta_data(pkl_load(meta_data_fp / f'{n}.pkl'))
                                   for n in self.subject_names()}

        self._skinning_cfg = SkinningConfiguration()  # Use Default config
        self._j2i = {k.replace('mixamorig_', ''): v for k, v in self._orig_j2i.items()}  # Shorten orig_j2i:
        self._i2j = {v: k for k, v in self._j2i.items()}



    def directed_joint_adjacency_matrix(self):
        return self._directed_joint_adj

    def directed_joint_edge_list(self):
        return self._directed_joint_edges

    def joint_index_to_name_map(self):
        return deepcopy(self._i2j)

    def _preprocess_meta_data(self, raw_meta):

        if self._orig_j2i is None:
            # [1] Set the global joint mappings on first subject meta data
            self._orig_j2i = raw_meta['joint_name2id']
            self._n_j = len(self._orig_j2i)  # Note - subject meta data dict ordering is *different* from sub to sub

            # [2] Translate joints_tree_hierarchy to a np.ndarray - TODO - tree information is identical for all subs
            edges = np.array([[self._orig_j2i[e[0]], self._orig_j2i[e[1]]] for e in raw_meta['joints_tree_hierarchy']])
            ne = edges.shape[0]
            assert ne == self._n_j - 1
            self._directed_joint_edges = edges
            self._directed_joint_edges.flags.writeable = False
            # [3] Compute joint adjacency matrix
            self._directed_joint_adj = csr_matrix((np.ones((ne,), dtype=np.int32), (edges[:, 0], edges[:, 1])),
                                                  (self._n_j, self._n_j), dtype=np.int32)
            # TODO - anyway to make this immutable?

        # [4] Concat inv_bind matrices
        # [5] Copy rest_vertices as is
        ret = {'inv_bind_mats': self._homogenic_mat_dict_to_numpy(raw_meta['inv_bind_matrices']),
               'rest_pose_verts': raw_meta['rest_vertices']}

        # [6] No need to copy faces - just assert they are equal to what we have - already validated for Mixamo
        # assert (np.array_equal(raw_subject_meta_data_dict['faces'], self._f))
        # [7] Concat skinning weights:
        ret['skinning_weights'] = raw_meta['reordered_weights']
        return ret

    def validate_dataset(self):
        super().validate_dataset()
        for si in tqdm(range(self.num_joint_files()), file=sys.stdout, dynamic_ncols=True):
            chi = self._hit.csi2chi(si)
            fp = self._hi2joint_path(chi)
            assert fp.is_file(), f"Missing full subject joint file {fp.resolve()} in dataset {self.name()}"
        print(f'Joint Validation -- PASSED --')

    def num_indexed_files(self):
        return self.num_projections() + self.num_full_shapes() + \
               self.num_subject_joint_meta_data_files() + self.num_joint_files()

    def num_subject_joint_meta_data_files(self):
        len(self._hit._hit.keys())

    def num_joints(self):
        return self._n_j

    def num_joint_files(self):
        return self.num_full_shapes()

    def set_skinning_config(self, cfg: SkinningConfiguration):
        self._skinning_cfg = deepcopy(cfg)

    def data_summary(self, with_tree=False):
        super().data_summary(with_tree=False)
        print(f'* Number of joints per mesh: {self.num_joints()}')
        print(f'* Number of joints files: {self.num_joint_files()}')
        print(f'* Number of joint meta data subject files: {self.num_subject_joint_meta_data_files()}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()

    def _datapoint_via_rand_f2j(self, si):
        gt_dict = self._datapoint_via_rand_f2f(si)
        joint_data = self._joint_path2data(self._hi2joint_path(gt_dict['gt_hi']))  # 'local' , 'global'
        # Extract the translation vectors from the world matrices
        gt_dict['gt_joint'] = np.array([joint_data['global'][joint][0:3, 3] for joint in self.joints_order])
        return gt_dict

    def _datapoint_via_rand_f2jp(self, si):
        gt_dict = self._datapoint_via_rand_f2p(si)
        joint_data = self._joint_path2data(self._hi2joint_path(gt_dict['gt_hi']))
        # Extract the translation vectors from the world matrices
        gt_dict['gt_joint'] = np.array([joint_data['global'][joint][0:3, 3] for joint in self.joints_order])
        # gt_dict['gt_joint'] = np.array([v[0:3, 3] for v in joint_data['global'].values()])
        return gt_dict

    def _datapoint_via_skinning(self, si):
        cfg = self._skinning_cfg
        if not cfg.need_part:
            x = self._datapoint_via_rand_f2f(si)
        else:
            x = self._datapoint_via_rand_f2p(si)

        meta_data = self._subject_meta_data[x['gt_hi'][0]]  # TODO - assuming here subject(tp) == subject(gt)

        # From per frame pickles:
        if cfg.need_gt_joint_pickle:
            gt_joint_d = self._joint_path2data(self._hi2joint_path(x['gt_hi']))
            self._set_homogenic_mat_by_cfg(x, 'gt_world_joints', gt_joint_d['global'], cfg.gt_global_joint_cfg)
            self._set_homogenic_mat_by_cfg(x, 'gt_local_joints', gt_joint_d['local'], cfg.gt_local_joint_cfg)
        if cfg.need_tp_joint_pickle:
            tp_joint_d = self._joint_path2data(self._hi2joint_path(x['tp_hi']))
            self._set_homogenic_mat_by_cfg(x, 'tp_world_joints', tp_joint_d['global'], cfg.tp_global_joint_cfg)
            self._set_homogenic_mat_by_cfg(x, 'tp_local_joints', tp_joint_d['local'], cfg.tp_local_joint_cfg)

        # From subject meta-data:
        self._set_homogenic_mat_by_cfg(x, 'inv_bind_mats', meta_data['inv_bind_mats'], cfg.inv_bind_mats_cfg)
        if cfg.need_res_pose_verts:
            x['rest_pose_verts'] = meta_data['rest_pose_verts']
        if cfg.need_skinning_weights:
            x['skinning_weights'] = meta_data['skinning_weights']
        return x

    def _homogenic_mat_dict_to_numpy(self, homo_mat_dict):
        return np.array([homo_mat_dict[joint] for joint in self.joints_order])

    def _set_homogenic_mat_by_cfg(self, d, tgt_key, matrix, cfg):
        if cfg == 0:  # NONE - Do nothing
            return
        else:
            if isinstance(matrix, dict):  # TODO - optimize for the common case?
                matrix = self._homogenic_mat_dict_to_numpy(matrix)
            if cfg == 1:  # ORIGINAL - place as is
                d[tgt_key] = matrix
            elif cfg == 2:  # TRUNC - truncate -> copy due to slice hashing
                d[tgt_key] = np.asarray(matrix[:, :3, :4])
            elif cfg == 3:  # ONLY_ROTATION - Only the rotation matrix
                d[tgt_key] = np.asarray(matrix[:, :3, :3])
            elif cfg == 4:  # ONLY_TRANSLATION - Only the translation vector
                d[tgt_key] = np.asarray(matrix[:, :3, 3])
            else:
                raise AssertionError(f'Unknown homogenic config option : {cfg}')

    @staticmethod
    def _joint_path2data(fp):
        return pkl_load(fp)

    def _hi2joint_path(self, hi):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyUnresolvedReferences
def completion_collate(batch, stop: bool = False):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if stop:
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if NP_STR_OBJ_ARRAY_PATTERN.search(elem.dtype.str) is not None:
                raise TypeError(
                    f"default_collate: batch must contain tensors, numpy arrays, "
                    f"numbers, dicts or lists; found {elem.dtype}")

            return completion_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Mapping):
        # A bit hacky - but works
        d = {}
        for k in elem:
            for suffix in ['_hi', '_mask', '_mask1', '_mask2', '_f']:  # TODO
                if k.endswith(suffix):
                    stop = True
                    break
            else:
                stop = False
            d[k] = completion_collate([d[k] for d in batch], stop)
        return d
        # return {key: default_collate([d[key] for d in batch],rec_level=1) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(completion_collate(samples) for samples in zip(*batch)))
    # elif isinstance(elem, container_abcs.Sequence):
    #     transposed = zip(*batch)
    #     return [completion_collate(samples) for samples in transposed]
    raise TypeError(
        f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem.dtype}")


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def _base_tester():
    from data.sets import DatasetMenu
    ds = DatasetMenu.order('DFaustProjSequential',data_dir_override=r"/home/adminpassis123/gipfs/Mano/data/DFaust/DFaust")
    ldr = ds.loaders(n_channels=6, method='rand_f2p', batch_size=10, device='cpu-single')
    # pbar = tqdm(total=len(ldr))
    i = 0
    for dp in tqdm(ldr):
        # pbar.update(1)
        # i=i+1
        pass
    # pbar.close()
    print("cool")


# TODO - Oshri, Haitham - Run me with a debugger step by step
def skinning_tutorial():
    from data.sets import DatasetMenu
    # Let's see what's on the menu:
    print('All implemented datasets: ', DatasetMenu.all_sets())

    # Let's filter out anything without skinning data:
    print('All datasets with skinning info: ', DatasetMenu.skinning_sets())

    # Let's order one of them:
    ds: ParametricSkinningDataset = DatasetMenu.order('MixamoSkinnedGaon9Proj')
    # Let's print a summary of the dataset:
    ds.data_summary(with_tree=True)
    # Each of the printed properties may be accessed directly:
    print('\nThe number of joints per mesh is: ', ds.num_joints())

    # We presume all meshes have the exact same bone connectivity - we can access this connectivity via an edge list:
    edges = ds.directed_joint_edge_list()  # Exactly n_joints - 1, seeing in a tree we have |V|-1 edges.
    print(edges)
    # Or via a sparse adjacency matrix of size n_joints x n_joints
    A = ds.directed_joint_adjacency_matrix()
    # We can map from the indices to the actual joint names via a dictionary:
    i2j = ds.joint_index_to_name_map()

    print(i2j)

    translated_edges = [[i2j[edge[0]], i2j[edge[1]]] for edge in edges]  # Translate each node index to node name
    print(translated_edges)

    # Now we turn to configure our loader. Please read the documentation of SkinningConfiguration before continuing  -
    # you may find it under the class definition. Please play with it to make sure you know how to configure the loader.
    # Remember - we presume both template and subject stem from the SAME subject, therefore, only one set of skinning
    # weights, inverse bind mats, rest pose etc are needed.
    cfg = SkinningConfiguration(gt_local_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                                gt_world_joint_mats_cfg=HomogenicMatCfg.NONE,
                                template_local_joint_mats_cfg=HomogenicMatCfg.ONLY_ROTATION,
                                template_world_joint_mats_cfg=HomogenicMatCfg.ONLY_TRANSLATION,
                                need_inverse_bind_mats=HomogenicMatCfg.TRUNC,
                                need_rest_pose_vertices=False,
                                need_skinning_weights=True)
    ds.set_skinning_config(cfg)  # If we don't do this - the default arguments of the SkinningConfiguration constructor
    # are used

    # Order a single loader of 100 tuples of (gt,tp,projection) with n_channels=6 and a batch size of 5.
    # We run this on a single cpu core, so we don't suffer from the parallelism initial overhead.
    # method == skinning must be used for the above to work.
    ldr = ds.loaders(s_nums=100, n_channels=6, method='skinning', batch_size=5, device='cpu-single')

    for datapoint in ldr:
        print(datapoint)  # Place a breakpoint right here to look at datapoint.


if __name__ == '__main__':
    _base_tester()
