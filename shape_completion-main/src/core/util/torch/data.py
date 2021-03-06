import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from copy import deepcopy


def determine_worker_num(num_examples, batch_size):
    import psutil
    num_batch_runs = int(num_examples / batch_size)
    if num_batch_runs < 10:  # Very small amount of runs
        return 0
    else:
        cpu_cnt = psutil.cpu_count(logical=False)
        if batch_size < cpu_cnt:
            return int(batch_size)
        else:
            return int(cpu_cnt)


# ----------------------------------------------------------------------------------------------------------------------
#                                               Argsparse Extension
# ----------------------------------------------------------------------------------------------------------------------
def none_or_str(value):
    if value == 'None':
        return None
    return value  # This is str by default


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class ReconstructableLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recon_table = None

    def set_name(self):
        return self.dataset._ds_inst.name()

    def init_recon_table(self, table):
        self._recon_table = deepcopy(table)

    def recon_table(self):
        return deepcopy(self._recon_table)

    def indices(self):
        return list(self.batch_sampler.sampler.indices)

    def num_indexed(self):
        return len(self.batch_sampler.sampler.indices)

    def num_in_iterable(self):
        return self.batch_sampler.sampler.length  # Presuming SubsetChoiceSampler

    def __repr__(self):
        return f'{self.__class__.__name__}({self.set_name()},batch_size={self._recon_table["batch_size"]},' \
               f'set_size={self._recon_table["set_size"]},transforms={self._recon_table["transforms"]})'


# 'dataset_name': self.name(),
# 'batch_size': batch_size,
# 'split': split,
# 'id_in_split': i,
# 'set_size': eff_set_size,
# 'transforms': str(transforms),
# 'global_shuffle': global_shuffle,
# 'partition_shuffle': do_shuffle,
# 'method': method,
# 'n_channels': n_channels,
# 'in_memory_index': self._hit_in_memory,
# 'is_dynamic': is_dynamic


class ParametricLoader(ReconstructableLoader):

    def faces(self):
        return self.dataset._ds_inst.faces()

    def num_verts(self):
        return self.dataset._ds_inst.num_verts()

    def num_faces(self):
        return self.dataset._ds_inst.num_faces()

    def null_shape(self, n_channels=None):
        if n_channels is None:
            n_channels = self._recon_table['n_channels']
        return self.dataset._ds_inst.null_shape(n_channels)

    def plot_null_shape(self, strategy='mesh', with_vnormals=False):
        return self.dataset._ds_inst.plot_null_shape(strategy=strategy, with_vnormals=with_vnormals)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class SubsetChoiceSampler(Sampler):
    def __init__(self, indices, length=None):
        self.indices = indices
        if length is None:
            length = len(self.indices)
        self.length = length

    def __iter__(self):
        # Inefficient, without replacement:
        # return iter(self.indices[:self.length])
        return (self.indices[i] for i in torch.randperm(len(self.indices))[:self.length])
        # Efficient, with replacement:
        # return (self.indices[i] for i in torch.randint(low=0,high=len(self.indices),size=(self.length,)))

    def __len__(self):
        return self.length


