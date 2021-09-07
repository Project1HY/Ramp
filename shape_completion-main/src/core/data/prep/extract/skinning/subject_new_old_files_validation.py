import os
from pathlib import Path
# from pywavefront import *
import multiprocessing
from geom.mesh.io.base import read_obj
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ---------------------------------------------------------------------------------------------------------------------#
# TODO - Correct Filepaths
BASE_FILES_ROOT = (Path(r'C:\Users\oshri.halimi\Documents\demo\full')).resolve()
NEW_FILES_ROOT = Path(r'C:\Users\oshri.halimi\Documents\demo\full_from_converted_fbx_to_collada').resolve()

# Monitor the processed animations: create an empty file for every proceesed animation
RECORD_ROOT = Path(r'C:\Users\oshri.halimi\Documents\demo\validation_record').resolve()


class SubjectValidator:
    def __init__(self, sub):
        self.sub = sub
        self.old_subject_animations = BASE_FILES_ROOT / sub
        self.new_subject_animations = NEW_FILES_ROOT / sub

        self.record_dp = RECORD_ROOT
        self.record_dp.mkdir(parents=True, exist_ok=True)

        self.todo_animations = self.get_todo_animations()

    def validate(self):
        a_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = a_pool.map(self.validate_animation, self.todo_animations)

        if np.sum(result) == len(self.todo_animations):
            print(f'subject {self.sub} validation finished successfully')

    def validate_animation(self, animation):
        old_animation_dir = os.path.join(self.old_subject_animations, animation)
        new_animation_dir = os.path.join(self.new_subject_animations, animation)

        result = True
        for file_name in os.listdir(old_animation_dir):
            if not file_name.endswith('.obj'):
                continue
            old_obj_path = os.path.join(old_animation_dir, file_name)
            new_obj_path = os.path.join(new_animation_dir, file_name)

            old_vertices, old_faces = read_obj(old_obj_path)
            new_vertices, new_faces = read_obj(new_obj_path)

            new_vertices /= 100

            a = np.all(np.isclose(old_vertices, new_vertices, rtol=0, atol=1e-05, equal_nan=False))
            b = np.all(old_faces == new_faces)

            if not a or not b:
                result = False

        with open(self.record_dp / (animation + '.txt'), 'w') as writer:
            writer.writelines(f'{int(result)}\n')

        return result

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
        return os.listdir(self.old_subject_animations)


if __name__ == '__main__':
    SubjectValidator(sub='020').validate()
    # base_vertices, base_faces = read_obj(r'\\gip-main\data\ShapeCompletion\Mixamo\full\000\2hand Idle\006.obj')
    # vertices, faces = read_obj(r'\\gip-main\data\ShapeCompletion\Mixamo\full_from_converted_fbx_to_collada\000\2hand Idle\006.obj')
    # vertices /= 100
    #
    # assert np.all(np.isclose(base_vertices, vertices, rtol=0, atol=1e-05, equal_nan=False))
    # assert np.all(base_faces == faces)
    # assert np.isclose(base_vertices, vertices, rtol=1e-05, atol=1e-08, equal_nan=False).all()
    # print(vertices.shape)
