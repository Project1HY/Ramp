import core.data.sets
from core.data.base import ParametricCompletionDataset, CompletionDataset
from core.data.index import HierarchicalIndexTree
import pickle as pk
import h5py

def read_pickle(pk_path):
    with open(pk_path, 'rb') as r:
        dict_read = pk.load(r)
    return dict_read

def read_masks(path):
    with h5py.File(path, 'r') as f:
        seq_frames = sorted(f.keys())
        n_frames = len(seq_frames)
    return(n_frames)

class DFaustHIT(CompletionDataset):  # TODO
    def __init__(self, data_dir_override, deformation, max_pose_per_sub, sequence_dict, path, num_proj_per_frame =10, num_pose_per_sub=None):
        self.num_proj_per_pose = deformation.num_projections()
        self.max_pose_per_sub = len(sequence_dict.keys())
        if num_pose_per_sub is None:
            num_pose_per_sub = max_pose_per_sub
        self.num_pose_per_sub = num_pose_per_sub
        frames_per_pose_and_sub = read_masks(path)
        self.frames_per_pose_and_sub = frames_per_pose_and_sub
        self.num_proj_per_frame = num_proj_per_frame
        super().__init__(data_dir_override=data_dir_override, deformation=deformation,
                         cls='scan', suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(self.num_pose_per_sub):
                hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)] = {}
                for frame_num in range(self.frames_per_pose_and_sub):
                    hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)][frame_num+(pose_id+sub_id*self.num_pose_per_sub)*self.frames_per_pose_and_sub]=self.num_proj_per_frame
        return HierarchicalIndexTree(hit, in_memory=True)