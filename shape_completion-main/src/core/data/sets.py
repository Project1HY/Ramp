from data.base import *
from data.index import HierarchicalIndexTree
from geom.mesh.io.base import *
from geom.mesh.op.cpu.remesh import clean_mesh
import os
from torch.utils.data.sampler import Sampler
import random
from util.torch.data import SubsetChoiceSampler

try:
    # This snippet is needed to solve the unpickling error. See shorturl.at/jktw2
    if __name__ != '__main__':
        # noinspection PyUnresolvedReferences
        import __main__

        __main__.HierarchicalIndexTree = HierarchicalIndexTree
except ImportError:
    pass


# TODO - Consider adding Surreal Dataset
# TODO - Consider migrating to lmdb or h5py
# TODO - Allow extraction of an orbital subset by angle - its the same ids every time...

# ----------------------------------------------------------------------------------------------------------------------
#                                                       FAUST
# ----------------------------------------------------------------------------------------------------------------------

class Faust(ParametricCompletionDataset):
    def __init__(self, data_dir_override, deformation):
        self.num_proj_per_pose = deformation.num_projections()
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation,
                         cls='synthetic', suspected_corrupt=False)

    def _construct_hit(self):  # Override
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(10):
                hit[str(sub_id)][str(pose_id)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.off'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'tr_reg_0{hi[0]}{hi[1]}_{hi[2]}.npz'

    def _hi2proj_path_azimuthal_projections(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_00{hi[2]}.npz'


# ----------------------------------------------------------------------------------------------------------------------
#                                                       FAUST SCANS
# ----------------------------------------------------------------------------------------------------------------------

class FaustScanBase(CompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation, num_pose_per_sub=10):
        self.num_proj_per_pose = deformation.num_projections()
        self.num_pose_per_sub = num_pose_per_sub
        super().__init__(data_dir_override=data_dir_override, deformation=deformation,
                         cls='scan', suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(self.num_pose_per_sub):
                hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)


class FaustTrainScans(FaustScanBase):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, num_pose_per_sub=10)

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'tr_scan_{hi[1]:>03}.ply'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'tr_scan_{hi[1]:>03}' / f'tr_scan_{hi[1]:>03}_{hi[2]}.npz'

    def _full_path2data_default(self, fp):
        return clean_mesh(*read_ply(fp))


class FaustTestScans(FaustScanBase):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, num_pose_per_sub=20)

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'test_scan_{hi[1]:>03}.off'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'test_scan_{hi[1]:>03}' / f'test_scan_{hi[1]:>03}_{hi[2]}.npz'

    def _full_path2data_default(self, fp):
        return clean_mesh(*read_off(fp))


# ----------------------------------------------------------------------------------------------------------------------
#                                                       AMASS
# ----------------------------------------------------------------------------------------------------------------------

class AmassBase(ParametricCompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}_projectionID_{hi[2]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}.OFF'


class AmassTrain(AmassBase, ABC):
    pass


class AmassVald(AmassBase, ABC):
    pass


class AmassTest(AmassBase, ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
#                                                       SMAL
# ----------------------------------------------------------------------------------------------------------------------

class SMALBase(ParametricCompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation, num_poses_per_sub):
        self._num_poses_per_sub = num_poses_per_sub
        super().__init__(n_verts=3889, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in ['cats', 'dogs', 'horses', 'cows', 'hippos']:
            hit[sub_id] = {}
            for pose_id in range(self._num_poses_per_sub):
                hit[sub_id][str(pose_id)] = 110
        return HierarchicalIndexTree(hit, in_memory=True)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'{hi[1]}' / f'{hi[1]}_{hi[2]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'{hi[1]}.ply'


class SMALTrain(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=20, *args, **kwargs)


class SMALVald(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=10, *args, **kwargs)


class SMALTest(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=10, *args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust
# ----------------------------------------------------------------------------------------------------------------------
class DFaust(ParametricCompletionDataset):
    def __init__(self, deformation, data_dir_override=r"./DFaust/"):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)
        # self._full_dir = Path(r'D:\yift_hadas_data\DFAUST_VERT_PICK')
        self._full_dir = self._vert_pick
    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:>05}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>05}.npy'

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>05}_{hi[3]}.npz'


class DFaustSequential(ParametricCompletionDataset):
    NULL_SHAPE_SI = 0

    def __init__(self, data_dir_override, deformation, full_dir_override, n_verts=6890):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)
        self._hits = []
        # print("yiftach now is :",self._full_dir)
        self._full_dir = self._vert_pick

    def _get_datapoint_func(self, hit_index):
        _hit: HierarchicalIndexTree = self._hits[hit_index]

        def _datapoint_via_path_tup(path_tup):
            if path_tup[2] >= path_tup[-1]:
                # TODO: return zeros.
                gt_dict = {}
                tp_hi = _hit.random_path_from_partial_path([path_tup[0]])[:-1]  # Shorten hi by 1
                # tp_hi = gt_dict['gt_hi'][:-1]
                tp_dict = self._full_dict_by_hi(tp_hi)
                gt_dict['gt_f'] = tp_dict['gt_f']
                gt_dict['tp_hi'] = tp_dict['gt_hi']
                gt_dict['tp'] = tp_dict['gt']
                gt_dict['tp_f'] = tp_dict['gt_f']
                gt_dict['gt_hi'] = path_tup
                gt_dict['gt'] = np.zeros((6890, 3), dtype='float32')
                gt_dict['gt_mask'] = self._mask_by_hi(tp_hi + (path_tup[3],))
                gt_dict['gt_real_sample'] = False
                return gt_dict
            gt_dict = self._full_dict_by_hi(path_tup)
            tp_hi = _hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
            tp_dict = self._full_dict_by_hi(tp_hi)
            gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
            gt_dict['gt_mask'] = self._mask_by_hi(path_tup)
            gt_dict['gt_real_sample'] = True

            return gt_dict

        return _datapoint_via_path_tup

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:>05}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>05}.npy'

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>05}_{hi[3]}.npz'

    def loaders(self, s_nums=None, s_shuffle=True, s_transform=None, split=(.8, .1, .1), s_dynamic=False,
                global_shuffle=False, batch_size=16, device='cuda', method='f2p', n_channels=6, truncation_size=3):
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
        assert sum(split) == 1, "Split fracs must sum to 1"
        # TODO - Clean up this function, add in smarter defaults, simplify
        if (method == 'f2p' or method == 'p2p') and not self._hit_in_memory:
            method = 'rand_' + method
            warn(f'Tuple dataset index is too big for this dataset. Reverting to {method} instead')
        if (method == 'frand_f2p' or method == 'frand_p2p') and len(split) != 1:
            raise ValueError("Seeing the fully-rand methods have no connection to the partition, we cannot support "
                             "a split dataset here")
        # Logic:
        s_nums = to_list(s_nums)
        subjects = list(self._hit.get_id_union_by_depth(depth=1))
        subjects = split_frac(subjects, split)
        n_parts = len(split)
        loaders = []
        for i in range(n_parts):
            set_subjects, req_set_size, do_shuffle, transforms, is_dynamic = subjects[i], s_nums[i], None, None, None

            partial_hit = self._hit.keep_ids_by_depth(keepers=list(set_subjects), depth=1)
            self._hits = self._hits + [partial_hit]
            if req_set_size is None:
                req_set_size = partial_hit.num_indexed()
            eff_set_size = min(partial_hit.num_indexed(), req_set_size)
            if eff_set_size != req_set_size:
                warn(f'At Loader {i + 1}/{n_parts}: Requested {req_set_size} objects while set has only {eff_set_size}.'
                     f' Reverting to latter')
            # if not is_dynamic:  # Truncate only if not dynamic
            #     set_ids = set_ids[:eff_set_size]  # Truncate
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
            ids = list(range(eff_set_size))
            ldr = self._loader(method=method, transforms=transforms, n_channels=n_channels, ids=None,
                               batch_size=batch_size, device=device, set_size=eff_set_size, partial_hit=partial_hit,
                               truncation_size=truncation_size, hit_index=i)
            ldr.init_recon_table(recon_stats)
            loaders.append(ldr)

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    def _reorder_identity_tuple(self, tup):
        return *tup[:-3], tup[-1], tup[-2], tup[-3]

    def _loader(self, method, transforms, n_channels, ids, batch_size, device, set_size, partial_hit,
                truncation_size=3, hit_index=0):

        # Handle Device:
        device = str(device).split(':')[0]  # Compatible for both strings & pytorch devs
        assert device in ['cuda', 'cpu', 'cpu-single']
        pin_memory = (device == 'cuda')
        if device == 'cpu-single':
            n_workers = 0
        else:
            n_workers = determine_worker_num(10e7, batch_size)
        import itertools

        ids = partial_hit.get_path_union_by_depth(2)
        ids = [[ident + (i, 0,) for i in range(10)] for ident in ids]
        ids = list(itertools.chain(*ids))

        ids = [self._reorder_identity_tuple(tup) for tup in ids]
        # Compile Sampler:
        if set_size is None:
            set_size = len(ids)
        assert len(ids) > 0, "Found loader with no data samples inside"
        length = sum([dp[-1] for dp in ids])
        sampler_length = min(set_size, len(ids))  # Allows for dynamic partitions

        data_subsampler = SubsetChoiceSampler(ids, set_size)
        data_sampler = SequentialAnimationBatchSampler(data_subsampler, ids, batch_size=batch_size, set_size=set_size)
        # Compiler Transforms:
        transforms = self._transformation_finalizer_by_method(method, transforms, n_channels)

        return self.LOADER_CLASS(FullPartSequentialTorchDataset(self, transforms, method,hit_index),
                                 batch_sampler=data_sampler, num_workers=n_workers, pin_memory=pin_memory,
                                 collate_fn=completion_collate)


class DFaustWindowedSequential(ParametricCompletionDataset):
    NULL_SHAPE_SI = 0

    def __init__(self, data_dir_override, deformation, n_verts=6890):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)
        self._hits = []
        # print("yiftach now is :",self._full_dir)
        self._full_dir = self._vert_pick
        self.window_size=1
        self.stride=0

    def set_window_params(self,window_size,stride):
        self.window_size=window_size
        self.stride=stride
    def _get_datapoint_func(self, hit_index):
        _hit: HierarchicalIndexTree = self._hits[hit_index]

        def _datapoint_via_path_tup(path_tup):
            if path_tup[2] >= path_tup[-1]:
                # TODO: return zeros.
                gt_dict = {}
                tp_hi = _hit.random_path_from_partial_path([path_tup[0]])[:-1]  # Shorten hi by 1
                # tp_hi = gt_dict['gt_hi'][:-1]
                tp_dict = self._full_dict_by_hi(tp_hi)
                gt_dict['gt_f'] = tp_dict['gt_f']
                gt_dict['tp_hi'] = tp_dict['gt_hi']
                gt_dict['tp'] = tp_dict['gt']
                gt_dict['tp_f'] = tp_dict['gt_f']
                gt_dict['gt_hi'] = path_tup
                gt_dict['gt'] = np.zeros((6890, 3), dtype='float32')
                gt_dict['gt_mask'] = self._mask_by_hi(tp_hi + (path_tup[3],))
                gt_dict['gt_real_sample'] = False
                return gt_dict
            gt_dict = self._full_dict_by_hi(path_tup)
            tp_hi = _hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
            tp_dict = self._full_dict_by_hi(tp_hi)
            gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_f'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_f']
            gt_dict['gt_mask'] = self._mask_by_hi(path_tup)
            gt_dict['gt_real_sample'] = True

            return gt_dict

        return _datapoint_via_path_tup

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:>05}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>05}.npy'

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>05}_{hi[3]}.npz'

    def loaders(self, s_nums=None, s_shuffle=True, s_transform=None, split=(.8, .1, .1), s_dynamic=False,
                global_shuffle=False, batch_size=16, stride=6, window_size=2, device='cuda', method='f2p', n_channels=6,
                truncation_size=3):
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
        assert sum(split) == 1, "Split fracs must sum to 1"
        # TODO - Clean up this function, add in smarter defaults, simplify
        if (method == 'f2p' or method == 'p2p') and not self._hit_in_memory:
            method = 'rand_' + method
            warn(f'Tuple dataset index is too big for this dataset. Reverting to {method} instead')
        if (method == 'frand_f2p' or method == 'frand_p2p') and len(split) != 1:
            raise ValueError("Seeing the fully-rand methods have no connection to the partition, we cannot support "
                             "a split dataset here")
        # Logic:
        print(f"stride is {stride} window is {window_size}")
        s_nums = to_list(s_nums)
        subjects = list(self._hit.get_id_union_by_depth(depth=1))
        subjects = split_frac(subjects, split)
        n_parts = len(split)
        loaders = []
        for i in range(n_parts):
            set_subjects, req_set_size, do_shuffle, transforms, is_dynamic = subjects[i], s_nums[i], None, None, None

            partial_hit = self._hit.keep_ids_by_depth(keepers=list(set_subjects), depth=1)
            self._hits = self._hits + [partial_hit]
            if req_set_size is None:
                req_set_size = partial_hit.num_indexed()
            eff_set_size = min(partial_hit.num_indexed(), req_set_size)
            if eff_set_size != req_set_size:
                warn(f'At Loader {i + 1}/{n_parts}: Requested {req_set_size} objects while set has only {eff_set_size}.'
                     f' Reverting to latter')
            # if not is_dynamic:  # Truncate only if not dynamic
            #     set_ids = set_ids[:eff_set_size]  # Truncate
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
            ids = list(range(eff_set_size))
            ldr = self._loader(method=method, transforms=transforms, n_channels=n_channels, ids=None,
                               batch_size=batch_size, device=device, set_size=eff_set_size, partial_hit=partial_hit,
                               truncation_size=truncation_size, hit_index=i, stride=stride, window_size=window_size)
            ldr.init_recon_table(recon_stats)
            loaders.append(ldr)

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    def _reorder_identity_tuple(self, tup):
        return *tup[:-2], tup[-1], tup[-2] # first elements are subject,pose,
                                           # and index, tup[-1]=azimuth, tup[-2]=sequence length

    def _loader(self, method, transforms, n_channels, ids, batch_size, device, set_size, partial_hit,
                truncation_size=3, hit_index=0, stride=3, window_size=3):

        # Handle Device:

        device = str(device).split(':')[0]  # Compatible for both strings & pytorch devs
        assert device in ['cuda', 'cpu', 'cpu-single']
        pin_memory = (device == 'cuda')
        if device == 'cpu-single':
            n_workers = 0
        else:
            n_workers = determine_worker_num(10e7, batch_size)
        import itertools

        ids = partial_hit.get_path_union_by_depth(3)
        ids = [[ident + (i, ) for i in range(10)] for ident in ids]
        ids = list(itertools.chain(*ids))

        ids = [self._reorder_identity_tuple(tup) for tup in ids]
        # Compile Sampler:
        if set_size is None:
            set_size = len(ids)
        assert len(ids) > 0, "Found loader with no data samples inside"
        length = sum([dp[-1] for dp in ids])
        sampler_length = min(set_size, len(ids))  # Allows for dynamic partitions

        data_sampler = SubsetChoiceSampler(ids, set_size)
        # Compiler Transforms:
        transforms = self._transformation_finalizer_by_method(method, transforms, n_channels)
        #sampler = EncompassedBatchSampler(data_sampler,
#EncompassedBatchSampler(Sampler):
 #   def __init__(self, subsampler, batch_size=1, window_size=2, length=None):
        return self.LOADER_CLASS(
            FullPartWindowedSequentialTorchDataset(self, transforms, method, hit_index, self.window_size, self.stride), batch_size=batch_size,
            sampler=data_sampler, num_workers=n_workers, pin_memory=pin_memory,
            collate_fn=sequential_completion_collate)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust Scans
# ----------------------------------------------------------------------------------------------------------------------

class DFaustScans(CompletionDataset, ABC):  # TODO
    def __init__(self, data_dir_override, deformation, num_pose_per_sub=10):
        self.num_proj_per_pose = deformation.num_projections()
        self.num_pose_per_sub = num_pose_per_sub
        super().__init__(data_dir_override=data_dir_override, deformation=deformation,
                         cls='scan', suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(self.num_pose_per_sub):
                hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust Scans
# ----------------------------------------------------------------------------------------------------------------------
class DFaustAccessories(CompletionDataset):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _construct_hit(self):
        self.num_proj_per_pose = self._deformation.num_projections()
        return HierarchicalIndexTree({file.stem: self.num_proj_per_pose for file in self._full_dir.glob('*')},
                                     in_memory=True)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'{hi[0]}_{hi[1]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'{hi[0]}.obj'


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Mixamo
# ----------------------------------------------------------------------------------------------------------------------

class MixamoBase(ParametricSkinningDataset, ABC):
    def __init__(self, data_dir_override, deformation, suspected_corrupt):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=suspected_corrupt)

    def _hi2proj_path_default(self, hi):
        for i in range(10):  # Num Angles. Hacky - but works. TODO - Should we rename?
            fp = self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>03}_{hi[3]}_angi_{i}.npz'
            if fp.is_file():
                return fp
        else:
            raise AssertionError

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>03}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>03}.obj'

    def _hi2joint_path(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>03}.pkl'

    def _full_path2data_default(self, fp):
        return read_obj_verts(fp)


class MixamoSkinnedGIP(MixamoBase):  # Should be: MixamoPyProj_2k_10ang_1fr
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GIP
            data_dir_override = MIXAMO_PATH_GIP
        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)


class MixamoSkinnedGaon9(MixamoBase):  # Should be: MixamoPyProj_2k_10ang_1fr
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9
        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)


class MiniMixamoSkinned13(MixamoBase):  # Should be: MiniMixamoPyProj_2k_10ang_1fr
    def __init__(self, *args, **kwargs):
        super().__init__(suspected_corrupt=False, *args, **kwargs)


class MiniMixamoSkinnedSingleSubject(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):
        self.subjects = ['000', '010', '020', '030', '040', '050', '060', '070', '080', '090']
        self.desired_subject = ['000']
        self.ignored_subjects = list(set(self.subjects) - set(self.desired_subject))

        if data_dir_override is None:
            from cfg import MINI_MIXAMO_PATH_GAON9
            data_dir_override = MINI_MIXAMO_PATH_GAON9

        super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit: HierarchicalIndexTree = super()._construct_hit()
        hit = hit.remove_ids_by_depth(depth=1, goners=self.ignored_subjects)
        # we just toke Bartending
        hit = hit.remove_ids_by_depth(depth=2,
                                      goners=['Air Squat', 'Arm Stretching', 'Sprint Turn', 'Standing', 'Stomping',
                                              'Talking At Watercooler', 'Throwing', 'Turn 90 Left', 'Turn 90 Right',
                                              'Victory', 'Walking', 'Waving'])
        # hit = hit.remove_ids_by_depth(depth=3, goners=[str(num).zfill(3) for num in range(2, 450)])
        return hit


class MiniMixamoSkinned(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MINI_MIXAMO_PATH_GAON9
            data_dir_override = MINI_MIXAMO_PATH_GAON9
        super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit: HierarchicalIndexTree = super()._construct_hit()
        return hit


# class MixamoSkinned(MixamoBase):
#     def __init__(self, data_dir_override, *args, **kwargs):
#         if data_dir_override is None:
#             from cfg import MIXAMO_PATH_GAON9
#             data_dir_override = MIXAMO_PATH_GAON9
#         super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)
#
#     def _construct_hit(self):
#         hit : HierarchicalIndexTree = super()._construct_hit()
#         return hit


class MixamoSkinned(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):
        self.subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9

        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)

    # TODO - do it smarter
    def _construct_hit(self):
        hit: HierarchicalIndexTree = super()._construct_hit()
        from cfg import BROKEN_ANIMATION
        broken_dir_path = BROKEN_ANIMATION
        all_animations = set([])
        for sub in self.subjects:
            with open(os.path.join(broken_dir_path, sub + '.txt'), "r") as f:
                broken_animation = f.read()
            broken_animation = broken_animation.split("\n")
            all_animations = all_animations.union(set(broken_animation))
        hit = hit.remove_ids_by_depth(depth=2, goners=list(all_animations))
        return hit


class MixamoSkinnedSingleSubject(MixamoSkinned):
    def __init__(self, data_dir_override, *args, **kwargs):
        self.subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
        self.desired_subject = ['000']
        self.ignored_subjects = list(set(self.subjects) - set(self.desired_subject))

        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9

        super().__init__(data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit: HierarchicalIndexTree = super()._construct_hit()
        hit = hit.remove_ids_by_depth(depth=1, goners=self.ignored_subjects)
        return hit


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------

class DatasetMenu:
    # TODO - once we have the whole grid, this could be simplified
    _IMPLEMENTED = {
        'AmassTestProj': (AmassTest, AzimuthalProjection()),
        'AmassValdProj': (AmassVald, AzimuthalProjection()),
        'AmassTrainProj': (AmassTrain, AzimuthalProjection()),

        'SMALTestEleProj': (SMALTest, OrbitalProjection()),
        'SMALValdEleProj': (SMALVald, OrbitalProjection()),
        'SMALTrainEleProj': (SMALTrain, OrbitalProjection()),
        'FaustProj': (Faust, AzimuthalProjection()),
        'FaustEleProj': (Faust, OrbitalProjection()),
        'FaustSemCuts': (Faust, SemanticCut()),

        'DFaustProj': (DFaust, AzimuthalProjection()),
        'DFaustProjSequential': (DFaustSequential, AzimuthalProjection()),
        'DFaustProjRandomWindowed': (DFaustWindowedSequential, AzimuthalProjection()),
        'DFaustSemCuts': (DFaust, SemanticCut(n_expected_proj=4)),

        # 'DFaustAccessories' : (DFaustAccessories, AzimuthalProjection()),
        'DFaustAccessoriesEleProj': (DFaustAccessories, OrbitalProjection()),
        'DFaustAccessoriesSemCuts': (DFaustAccessories, SemanticCut()),

        'MiniMixamoSkinned13Proj': (MiniMixamoSkinned13, AzimuthalProjection(n_expected_proj=2)),

        # the following loaders works on Gaon9
        'MiniMixamoSkinnedSingleSubject': (MiniMixamoSkinnedSingleSubject, AzimuthalProjection(n_expected_proj=2)),
        'MiniMixamoSkinned': (MiniMixamoSkinned, AzimuthalProjection(n_expected_proj=2)),

        'MixamoSkinnedSingleSubject': (MixamoSkinnedSingleSubject, AzimuthalProjection(n_expected_proj=2)),
        'MixamoSkinned': (MixamoSkinned, AzimuthalProjection(n_expected_proj=2)),

        # 'MiniMixamoSkinned13EleProj': (MiniMixamoSkinned13, OrbitalProjection()),
        'MiniMixamoSkinned13SemCuts': (MiniMixamoSkinned13, SemanticCut(n_expected_proj=4)),

        # 'FaustTrainScansProj': (FaustTrainScans, AzimuthalProjection()),
        'FaustTrainScansEleProj': (FaustTrainScans, OrbitalProjection()),
        'FaustTrainScansSemCuts': (FaustTrainScans, SemanticCut()),

        # 'FaustTestScansProj': (FaustTestScans, AzimuthalProjection()),
        'FaustTestScansEleProj': (FaustTestScans, OrbitalProjection()),
        'FaustTestScansSemCuts': (FaustTestScans, SemanticCut()),
        #
        # 'DFaustScansProj': (DFaustScans, AzimuthalProjection()),
        # 'DFaustScansEleProj': (DFaustScans, OrbitalProjection()),
        # 'DFaustScansSemCuts': (DFaustScans, SemanticCut()),

        'MixamoSkinnedGIPProj': (MixamoSkinnedGIP, AzimuthalProjection(n_expected_proj=2)),
        'MixamoSkinnedGaon9Proj': (MixamoSkinnedGaon9, AzimuthalProjection(n_expected_proj=2)),
    }

    @classmethod
    def all_sets(cls):
        return tuple(cls._IMPLEMENTED.keys())

    @classmethod
    def order(cls, dataset_name, data_dir_override=None, full_dir_override=None):
        if dataset_name in cls._IMPLEMENTED:
            tup = cls._IMPLEMENTED[dataset_name]
            return tup[0](data_dir_override=data_dir_override, deformation=tup[1])
        else:
            raise ValueError(f'Could not find dataset {dataset_name} - check spelling')

    @classmethod
    def skinning_sets(cls):
        # TODO - create more filters?
        return tuple(k for k, v in cls._IMPLEMENTED.items() if issubclass(v[0], ParametricSkinningDataset))

    @classmethod
    def test_all_sets(cls, with_full_validation=False):
        from util.strings import green_banner, red_banner
        for ds_name in cls.all_sets():
            try:
                ds: CompletionDataset = cls.order(ds_name)
                ldr = ds.loaders(split=[1], s_shuffle=[False], s_nums=[1],
                                 batch_size=1, device='cpu-single', method='rand_f2p',
                                 n_channels=6, s_dynamic=[False])  # , s_transform=[RandomPartNormalDirections()]
                for _ in ldr:
                    # plot_mesh_montage(**prepare_completion_montage(d), strategy='mesh')
                    green_banner(f'SUCCESS - {ds_name}')
                    break
                if with_full_validation:
                    ds.validate_dataset()

            except:
                red_banner(f'FAILED: {ds_name}')
                the_type, the_value, the_traceback = sys.exc_info()
                print(f'Exception: {the_type.__name__} : {the_value}')
                # print(f'Stack Trace:')
                # traceback.print_tb(the_traceback)


# ----------------------------------------------------------------------------------------------------------------------
# TODO: Move to correct location!
# ----------------------------------------------------------------------------------------------------------------------
# noinspection DuplicatedCode
class SequentialAnimationBatchSampler(Sampler):
    def __init__(self, subset_sampler, indices, batch_size=1, length=None, set_size=None):
        self.indices = indices
        if length is None:
            length = len(subset_sampler)
        self.total_length = length
        self.length = length // batch_size + 1
        self.batch_size = batch_size
        self.i = 0
        self.subset_sampler = subset_sampler
        self.set_size = set_size

    def __iter__(self):
        # Inefficient, without replacement:
        loc = 0
        random.shuffle(self.indices)
        for index in self.subset_sampler:
            while True:
                if self.set_size is not None and loc > self.set_size:
                    break
                current_batch_size = min(self.batch_size, index[-1] - index[2])
                returned_indices = [index] * self.batch_size
                for i in range(len(returned_indices)):
                    returned_indices[i] = (
                        *returned_indices[i][:-3], returned_indices[i][-3] + i, *returned_indices[i][-2:])
                yield returned_indices
                if index[2] + self.batch_size >= index[-1]:
                    break
                index = (
                    *index[:-3], index[-3] + self.batch_size, *index[-2:])
                loc = loc + len(returned_indices)
            if self.set_size is not None and loc > self.set_size:
                break

    def __len__(self):
        return self.length


class EncompassedBatchSampler(Sampler):
    def __init__(self, subsampler, batch_size=1, window_size=2, length=None):
        self.subsampler = subsampler
        if length is None:
            length = len(self.subsampler)
        assert batch_size%window_size==0, f"window size {window_size} does not divide batch size {batch_size}"
        self.length = length // (batch_size*window_size) 
        self.batch_size = batch_size
        self.window_size = window_size

    def __iter__(self):
        # Inefficient, without replacement:
        # return iter(self.indices[:self.length])
        entries = []
        counter = 0
        for sample in self.subsampler:
            counter += 1
            entries += [sample]
            if counter == self.batch_size*self.window_size:
                yield entries
                entries = []
            
    def __len__(self):
        return self.length


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Helper Functions
# ----------------------------------------------------------------------------------------------------------------------

def generate_parts():
    from geom.mesh.io.base import write_ply
    ds: CompletionDataset = DatasetMenu.order('FaustProj')
    for i in range(10):
        v, f = ds.partial_shape_by_hi(('8', '7', i), verts_only=False)
        write_ply(f'{i}.ply', v, f)


def check_correspondence():
    from geom.mesh.vis.base import plot_mesh_montage, uniclr
    ds: ParametricCompletionDataset = DatasetMenu.order('DFaustProj')
    ldr = ds.loaders(method='full', batch_size=9, s_shuffle=True, n_channels=3)
    vi = np.random.choice(range(ds.num_verts()), size=10, replace=False)
    clr = uniclr(N=ds.num_faces(), inds=vi)
    for d in ldr:
        plot_mesh_montage(vb=d['gt'], fb=ds.faces(), clrb=clr, strategy='mesh', lighting=True, show_edges=True)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # check_correspondence()
    # DatasetMenu.test_all_sets(with_full_validation=True)
    # ds: CompletionDataset = DatasetMenu.order('FaustEleProj',data_dir_override='R:\Mano\data\MiniMixamoSkinned13')
    ds: CompletionDataset = DatasetMenu.order('FaustEleProj').validate_dataset()

    # print("f"
