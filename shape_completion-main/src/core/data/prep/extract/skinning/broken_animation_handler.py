import os
from pathlib import Path
from tqdm import tqdm
import pickle
import sys
import json
import subprocess
from distutils.dir_util import copy_tree

FULL_ROOT = Path(r'R:\MixamoSkinned\full').resolve()
STATISTICS_ROOT = Path(r'R:\MixamoSkinned\statistics').resolve()
BROKEN_ROOT = Path(r'R:\MixamoSkinned\broken_animation').resolve()


class BrokenAnimationsPerSubject:
    def __init__(self, sub):
        self.sub = sub
        self.statistics_p = STATISTICS_ROOT / sub
        self.full_p = FULL_ROOT / sub
        self.broken_p = BROKEN_ROOT

        self.broken_dir_p = BROKEN_ROOT / sub
        self.broken_dir_p.mkdir(parents=True, exist_ok=True)

    def save_names(self):
        pkl_files = os.listdir(self.statistics_p)
        broken_animation = []
        for animation_pkl_file_name in tqdm(pkl_files, file=sys.stdout, total=len(pkl_files)):
            animation_pkl_file_path = os.path.join(self.statistics_p, animation_pkl_file_name)
            with open(animation_pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                if len(data['problematic_joints']) > 0:
                    broken_animation += [animation_pkl_file_name.split('.')[0]]
        with open(os.path.join(self.broken_p, self.sub + '.json'), "w") as fp:
            json.dump(broken_animation, fp)

    def move_animation(self):
        with open(os.path.join(self.broken_p, self.sub + '.txt'), "r") as f:
            broken_animation = f.read()
        broken_animation = broken_animation.split("\n")
        for animation_name in tqdm(broken_animation):
            source_path = os.path.join(self.full_p, animation_name)
            dest_path = os.path.join(self.broken_dir_p, animation_name)
            copy_tree(source_path, dest_path)


    def get_all_animation(self):
        all = set([])
        for sub in [str(sub).zfill(3) for sub in range(10, 100, 10)]:
            with open(os.path.join(self.broken_p, sub + '.txt'), "r") as f:
                broken_animation = f.read()
            broken_animation = broken_animation.split("\n")
            all = all.union(set(broken_animation))
        print(len(all))
        return all
if __name__ == '__main__':
    # subjects = [str(sub).zfill(3) for sub in range(10, 100, 10)]
    # for subject in subjects:
    #     BrokenAnimationsPerSubject(subject).move_animation()

    BrokenAnimationsPerSubject('000').get_all_animation()
