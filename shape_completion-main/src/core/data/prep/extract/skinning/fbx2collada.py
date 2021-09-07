import os
import subprocess
import json
from tqdm import tqdm
import sys
from pathlib import Path
import multiprocessing


# TODO - Correct Filepaths
class FbxDecoder:
    def __init__(self):

        self.fbx_path = os.path.join('C:', 'Program Files', 'Autodesk', 'FBX', 'FBX Converter', '2013.3', 'bin',
                                     'FbxConverter.exe')

        self.load_dir_base = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'spares', 'collaterals',
                                          'fbx_sequences', 'MPI-FAUST')
        self.save_dir_base = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'spares', 'collaterals',
                                          'collada_sequences', 'MPI-FAUST')

    def extract_subject(self, sub):
        load_dir_subject = os.path.join(self.load_dir_base, sub)
        save_dir_subject = os.path.join(self.save_dir_base, sub)

        if os.path.exists(save_dir_subject) == False:
            os.makedirs(save_dir_subject)

        animation_dict, animation_dict_fp = self.animations_dict(sub, load_dir_subject)
        animations_todo = [k for k, v in animation_dict.items() if v < 1]
        if animations_todo:

            for animation_index, animation_name in tqdm(enumerate(animations_todo), file=sys.stdout,
                                                        total=len(animations_todo), unit='animations'):
                fbx_file_path = os.path.join(load_dir_subject, animation_name + '.fbx')
                collada_file_path = os.path.join(save_dir_subject, animation_name + '.dae')

                subprocess.call([self.fbx_path, fbx_file_path, collada_file_path])
                animation_dict[animation_name] = 1
                with open(animation_dict_fp, 'w') as handle:
                    json.dump(animation_dict, handle, sort_keys=True, indent=4)

    def animations_dict(self, sub, load_dir_subject):
        animations_dict_fp = Path(f'{sub}_validation_dict.json').resolve()
        if animations_dict_fp.is_file():  # Animation Group Dict exists
            with open(animations_dict_fp, 'r') as handle:
                animations_dict = json.load(handle)
        else:  # Create it:
            animations_dict = {os.path.splitext(animation)[0]: 0 for animation in
                               self.animations_per_subject(load_dir_subject)}
            with open(animations_dict_fp, 'w') as handle:
                json.dump(animations_dict, handle, sort_keys=True, indent=4)  # Dump as JSON for readability
                print(f'Saved animation group cache for subject {sub} at {animations_dict_fp}')
        return animations_dict, animations_dict_fp

    def animations_per_subject(self, load_dir_subject):
        """
        :param sub: The subject name
        :return: All animation names given the subject
        """
        fp = Path(load_dir_subject).resolve()
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator


if __name__ == '__main__':
    sub = '040'
    decoder = FbxDecoder()
    # decoder.extract_subject(sub)

    a_pool = multiprocessing.Pool()

    subjets = ['040', '050', '060', '070', '080', '090']
    result = a_pool.map(decoder.extract_subject, subjets)
