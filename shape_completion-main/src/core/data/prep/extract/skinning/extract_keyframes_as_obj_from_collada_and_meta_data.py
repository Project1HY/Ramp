from geom.mesh.io.collad import ColladaFile
import numpy as np
import os

# TODO (Haitham) - Insert into prep main / metadata_extraction as a function
# TODO - Correct Filepaths
subject_animation_path = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'spares', 'collaterals', 'collada_sequences', 'MPI-FAUST', '000')
subject_obj_path = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'full_from_converted_fbx_to_collada', '000')
subject_keyframes_meta_data_path = os.path.join('\\\\gip-main', 'data', 'ShapeCompletion', 'Mixamo', 'full_meta_data_keyframes', '000')


# if os.path.exists(save_dir_subject) == False:
#     os.makedirs(save_dir_subject)

for animation_name in os.listdir(subject_animation_path):
    if animation_name.endswith(".dae"):

        animation_path = os.path.join(subject_animation_path, animation_name)

        animation_obj_dirc = os.path.join(subject_obj_path, os.path.splitext(animation_name)[0])
        os.makedirs(animation_obj_dirc, exist_ok=True)

        animation_keyframes_meta_data_dirc = os.path.join(subject_keyframes_meta_data_path, os.path.splitext(animation_name)[0])
        os.makedirs(animation_keyframes_meta_data_dirc, exist_ok=True)

        animation = ColladaFile(animation_path, is_converted_file=True, log_path='logger.txt', log_bind_shape_matrix_path='bind_logger.txt')

        animation.save_animation_keyframes_as_obj_files(animation_obj_dirc)
        animation.save_animation_meta_data(animation_keyframes_meta_data_dirc)
