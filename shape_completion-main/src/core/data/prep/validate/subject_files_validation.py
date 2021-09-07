import os
import numpy as np
from util.fs import pkl_load

if __name__ == '__main__':
    subject_metadata_dir = r'\\gip-main\data\ShapeCompletion\Mixamo\subject_meta_data'
    subjects_metadata_files = os.listdir(subject_metadata_dir)

    base_file_path = os.path.join(subject_metadata_dir, subjects_metadata_files[0])
    base_subject_metadata = pkl_load(fp=base_file_path)

    base_inv_bind_matrices = base_subject_metadata['inv_bind_matrices']
    base_joint_name2id = base_subject_metadata['joint_name2id']
    base_joints_tree_hierarchy = base_subject_metadata['joints_tree_hierarchy']
    base_rest_vertices = base_subject_metadata['rest_vertices']
    base_faces = base_subject_metadata['faces']
    base_weights = base_subject_metadata['weights']

    for file in subjects_metadata_files:
        file_path = os.path.join(subject_metadata_dir, file)
        subject_metadata = pkl_load(fp=file_path)

        inv_bind_matrices = subject_metadata['inv_bind_matrices']
        joint_name2id = subject_metadata['joint_name2id']
        joints_tree_hierarchy = subject_metadata['joints_tree_hierarchy']
        rest_vertices = subject_metadata['rest_vertices']
        faces = subject_metadata['faces']
        weights = subject_metadata['weights']

        # checks that all the labels are equal
        assert set(base_joint_name2id.keys()) == set(joint_name2id.keys())

        # size equality check
        assert len(base_inv_bind_matrices) == len(inv_bind_matrices)
        assert np.stack(base_weights.values()).shape == np.stack(weights.values()).shape
        assert base_faces.shape == faces.shape
        assert base_rest_vertices.shape == rest_vertices.shape

        for edge in base_joints_tree_hierarchy:
            assert edge in joints_tree_hierarchy
