import os
from geom.mesh.io.collad import ColladaFile
from pathlib import Path
from tqdm import tqdm
from util.strings import banner, title
import multiprocessing
import pickle
import sys
import json

# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Globals

# ----------------------------------------------------------------------------------------------------------------------#

IN_ROOT = (Path(r'Z:\MixamoPyProj\collaterals\collada\MPI-FAUST')).resolve()
OBJ_ROOT = Path(r'Z:\MixamoPyProj\full').resolve()
METADATA_ROOT = Path(r'Z:\MixamoPyProj\full').resolve()
STATISTICS_ROOT = Path(r'R:\MixamoSkinned\statistics').resolve()
BROKEN_ROOT = Path(r'R:\MixamoSkinned\broken_animation').resolve()

# Monitor the processed animations: create an empty file for every proceesed animation
RECORD_ROOT = Path(r'Z:\MixamoPyProj\record').resolve()


class BrokenAnimationsPerSubject:
    def __init__(self, sub):
        self.sub = sub
        self.statistics_dp = STATISTICS_ROOT / sub
        self.broken_dp = BROKEN_ROOT
    def save_names(self):
        pkl_files = os.listdir(self.statistics_dp)
        broken_animation = []
        for animation_pkl_file_name in tqdm(pkl_files, file=sys.stdout, total=len(pkl_files)):
            animation_pkl_file_path = os.path.join(self.statistics_dp, animation_pkl_file_name)
            with open(animation_pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                if len(data['problematic_joints']) > 0:
                    broken_animation += [animation_pkl_file_name.split('.pkl')[0]]

        with open(os.path.join(self.broken_dp, self.sub + '.txt'), "w") as f:
            f.write("\n".join(broken_animation))



class DataExtarctionPerSubject:
    def __init__(self, sub):
        self.sub = sub
        self.in_dp = IN_ROOT / sub

        self.metadata_dp = METADATA_ROOT / sub
        self.metadata_dp.mkdir(parents=True, exist_ok=True)

        self.objects_dp = OBJ_ROOT / sub
        self.objects_dp.mkdir(parents=True, exist_ok=True)

        self.statics_dp = STATISTICS_ROOT / sub
        self.statics_dp.mkdir(parents=True, exist_ok=True)

        self.record_dp = RECORD_ROOT
        self.record_dp.mkdir(parents=True, exist_ok=True)

        self.todo_animations = self.get_todo_animations()

    def extract(self):
        banner(f'mixamo' + title(f' Dataset :: Subject {self.sub} '))

        a_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # print(self.todo_animations)
        result = a_pool.map(self._extract_subject, self.todo_animations)
        print(result)
        # for animation in self.todo_animations:
        #     print(animation)
        #     self._extract_subject(animation)

        # for animation in self.todo_animations:
        #     print(animation)
        #     self._extract_subject(animation)
        banner(f'Extraction of Subject {self.sub} - COMPLETED')

    def _extract_subject(self, animation):
        animation_name = animation + ".dae"
        animation_dp = os.path.join(self.in_dp, animation_name)

        animation_obj_dp = os.path.join(self.objects_dp, animation)
        Path(animation_obj_dp).mkdir(exist_ok=True)

        animation_metadata_dp = os.path.join(self.metadata_dp, animation)
        Path(animation_metadata_dp).mkdir(exist_ok=True)

        collada_animation = ColladaFile(animation_dp, is_converted_file=True)

        # collada_animation.save_animation_keyframes_as_obj_files(animation_obj_dp)
        collada_animation.save_animation_meta_data(animation_metadata_dp)
        collada_animation.save_statiscis(os.path.join(self.statics_dp))

        with open(self.record_dp / (animation + '.txt'), 'w') as writer:
            writer.writelines('\n')

        return True

    def get_todo_animations(self):
        all_animation = [os.path.splitext(animation)[0] for animation in self.animations_per_subject()]
        self.record_dp /= self.sub
        if self.record_dp.is_dir():
            animations_done = [os.path.splitext(animation)[0] for animation in os.listdir(self.record_dp)]
            # list substarction
            todo_animation = [animation for animation in all_animation if animation not in animations_done]
            return todo_animation
        else:
            self.record_dp.mkdir(parents=True, exist_ok=True)
            return all_animation

    def animations_per_subject(self):
        """
        :return: All validation group names given the subject
        """
        return os.listdir(self.in_dp)


if __name__ == '__main__':
    # subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
    # for sub in subjects:
    # DataExtarctionPerSubject(sub='000').extract()
    # DataExtarctionPerSubject(sub='010').extract()
    # DataExtarctionPerSubject(sub='020').extract()
    # DataExtarctionPerSubject(sub='030').extract()
    # DataExtarctionPerSubject(sub='040').extract()
    # DataExtarctionPerSubject(sub='050').extract()
    # DataExtarctionPerSubject(sub='060').extract()
    # DataExtarctionPerSubject(sub='070').extract()
    # DataExtarctionPerSubject(sub='080').extract()
    # DataExtarctionPerSubject(sub='090').extract()
    subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
    for subject in subjects:
        BrokenAnimationsPerSubject(subject).save_names()
