import os
import json
from geom.mesh.io.collad import ColladaFile
from pathlib import Path
# from pywavefront import *
import multiprocessing
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ---------------------------------------------------------------------------------------------------------------------#

# TODO - Correct Filepaths
ROOT = Path(r'\\gip-main\data\ShapeCompletion\Mixamo\spares\collaterals\collada_sequences\MPI-FAUST').resolve()
REFERENCE_ANIMATION = 'Baseball Catcher.dae'
VALIDATION_OUT = Path(r'\\gip-main\data\ShapeCompletion\Mixamo\subject_animation_validation_dictionary').resolve()

# Monitor the processed animations: create an empty file for every proceesed animation
RECORD_ROOT = Path(r'\\gip-main\data\ShapeCompletion\Mixamo\subjectper_seq_validation_record').resolve()


class SubjectValidator:
    def __init__(self, sub):
        self.sub = sub
        self.subject_animations_dir = ROOT / sub
        self.reference_animation_path = self.subject_animations_dir / REFERENCE_ANIMATION
        self.validation_out_dir = VALIDATION_OUT
        self.validation_out_dir.mkdir(parents=True, exist_ok=True)

        reference_animation = ColladaFile(os.path.join(self.reference_animation_path), is_converted_file=True)

        self.reference_weights = reference_animation.weights
        self.reference_joint_name2id = reference_animation.joint_name2id
        self.reference_inv_bind_matrices = reference_animation.inv_joints_trans
        self.reference_bind_pose_matrix = reference_animation.bind_pose_matrix
        self.reference_vertices = reference_animation.vertices
        self.reference_faces = reference_animation.faces

        self.record_dp = RECORD_ROOT
        self.record_dp.mkdir(parents=True, exist_ok=True)

        self.todo_animations = self.get_todo_animations()

    def validate(self):
        a_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = a_pool.map(self.validate_animation, self.todo_animations)

        dict = self.get_validation_dict()
        for animation, res in zip(self.todo_animations, result):
            dict[animation] = str(res)
        print(dict)
        self.write_validation_dict(dict)
        print(
            f'{np.sum(result)} animation have identical metadata from {len(self.todo_animations)} for subject {self.sub}')

    def validate_animation(self, animation_name):
        animation_path = os.path.join(self.subject_animations_dir, animation_name + '.dae')

        animation = ColladaFile(animation_path, is_converted_file=True)

        a = np.all(self.reference_weights == animation.weights)
        b = np.all(self.reference_joint_name2id == animation.joint_name2id)
        c = np.all(self.reference_bind_pose_matrix == animation.bind_pose_matrix)
        d = np.all(self.reference_vertices == animation.vertices)
        e = np.all(self.reference_faces == animation.faces)

        res = (a == b == c == d == e)
        for joint_name, joint_inv_mat in self.reference_inv_bind_matrices.items():
            res = (res == np.all(joint_inv_mat == animation.inv_joints_trans[joint_name]))

        with open(self.record_dp / (animation_name + '.txt'), 'w') as writer:
            writer.writelines(f'{int(res)}\n')
        return res

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
        return os.listdir(self.subject_animations_dir)

    def get_validation_dict(self):
        validation_dict_fp = self.validation_out_dir / f'{self.sub}_validation_dict.json'
        if validation_dict_fp.is_file():  # Validation Group Dict exists
            with open(validation_dict_fp, 'r') as handle:
                validation_dict = json.load(handle)
                return validation_dict
        else:  # Create it:
            return {}

    def write_validation_dict(self, dict):
        validation_dict_fp = self.validation_out_dir / f'{self.sub}_validation_dict.json'
        with open(validation_dict_fp, 'w') as handle:
            json.dump(dict, handle, sort_keys=True, indent=4)  # Dump as JSON for readability


if __name__ == '__main__':
    subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
    for sub in subjects:
        SubjectValidator(sub=sub).validate()
    # base_vertices, base_faces = read_obj(r'\\gip-main\data\ShapeCompletion\Mixamo\full\000\2hand Idle\006.obj')
    # vertices, faces = read_obj(r'\\gip-main\data\ShapeCompletion\Mixamo\full_from_converted_fbx_to_collada\000\2hand Idle\006.obj')
    # vertices /= 100
    #
    # assert np.all(np.isclose(base_vertices, vertices, rtol=0, atol=1e-05, equal_nan=False))
    # assert np.all(base_faces == faces)
    # assert np.isclose(base_vertices, vertices, rtol=1e-05, atol=1e-08, equal_nan=False).all()
    # print(vertices.shape)
