import os
import json
from geometry.mesh.io.loader import Loader
from pathlib import Path
import tempfile

from util.strings import banner, print_yellow, print_red, title
from geometry.mesh.io.base import read_obj_verts, read_ply_verts, read_off, read_ply
from geometry.mesh.plot import plot_mesh_montage
from util.fs import assert_new_dir
import multiprocessing

# ----------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ----------------------------------------------------------------------------------------------------------------------#
# loacl disk paths
# IN_ROOT = (Path(r'C:\Users\oshri.halimi\Documents\demo\collada_seq\mpi')).resolve()
# OBJ_ROOT = Path(r'C:\Users\oshri.halimi\Documents\demo\full_seq\mpi').resolve()
# METADATA_ROOT = Path(r'C:\Users\oshri.halimi\Documents\demo\metadata').resolve()

# network paths
IN_ROOT = (Path(r'\\gip-main\data\ShapeCompletion\Mixamo\spares\collaterals\collada_sequences\MPI-FAUST')).resolve()
OBJ_ROOT = Path(r'\\gip-main\data\ShapeCompletion\Mixamo\full_from_converted_fbx_to_collada').resolve()
METADATA_ROOT = Path(r'\\gip-main\data\ShapeCompletion\Mixamo\full_meta_data_keyframes').resolve()

# Monitor the processed animations: create an empty file for every proceesed animation - record always local
G_RECORD_ROOT = Path(r'C:\Users\oshri.halimi\Projects\Deep-Shape-Completion\data_creation_monitoring\record').resolve()

LOGGER_PATH = 'logger.txt'
BIND_LOGGER = 'bind_logger.txt'


class DataCreator:
    TMP_OBJECT_ROOT = Path(tempfile._get_default_tempdir()).resolve() / 'objects'
    TMP_METADATA_ROOT = Path(tempfile._get_default_tempdir()).resolve() / 'metadata'

    TMP_OBJECT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_METADATA_ROOT.mkdir(parents=True, exist_ok=True)

    RECORD_ROOT = G_RECORD_ROOT

    OUT_IS_A_NETWORK_PATH = False

    def __init__(self):

        self.in_dp = IN_ROOT

        self.metadata_dp = METADATA_ROOT
        self.metadata_dp.mkdir(parents=True, exist_ok=True)

        self.objects_dp = OBJ_ROOT
        self.objects_dp.mkdir(parents=True, exist_ok=True)

        print(f'Target output directory: {self.objects_dp}')

        self.tmp_objects_dp = self.objects_dp  # The write is not temporary - it is final
        self.tmp_metadata_dp = self.metadata_dp

        # if self.OUT_IS_A_NETWORK_PATH:
        #     assert self.TMP_OBJECT_ROOT.is_dir(), f"Could not find directory {self.TMP_OBJECT_ROOT}"
        #     self.tmp_objects_dp = self.TMP_OBJECT_ROOT
        #     self.tmp_objects_dp.mkdir(parents=True, exist_ok=True)
        #
        #     assert self.TMP_METADATA_ROOT.is_dir(), f"Could not find directory {self.TMP_METADATA_ROOT}"
        #     self.tmp_metadata_dp = self.TMP_METADATA_ROOT
        #     self.tmp_metadata_dp.mkdir(parents=True, exist_ok=True)

        self.record_dp = self.RECORD_ROOT
        self.record_dp.mkdir(parents=True, exist_ok=True)
        print(f'Target validation-animation directory: {self.record_dp}')

    def extract_subject(self, sub):
        banner(f'mixamo' +
               title(f' Dataset :: Subject {sub} '))
        (self.tmp_objects_dp / sub).mkdir(exist_ok=True)  # TODO - Presuming this dir structure
        (self.tmp_metadata_dp / sub).mkdir(exist_ok=True)

        self.extract_subject_validated(sub)
        banner(f'Extraction of Subject {sub} - COMPLETED')

    def extract_subject_validated(self, sub):
        animations_todo = self.get_animations(sub)
        a_pool = multiprocessing.Pool(multiprocessing.cpu_count())

        result = a_pool.map(self._extract_subject, zip([sub] * len(animations_todo), animations_todo))
        print(result)

    def _extract_subject(self, tuple):

        # shape_fps = self.shape_fps_per_vgroup(sub, animation)
        # assert len(shape_fps) > 0, "Empty Validation group "
        sub, animation = tuple

        # Create all needed directories:
        animation_name = animation + ".dae"
        animation_dp = os.path.join(self.in_dp, sub, animation_name)  # TODO - Presuming this dir structure

        animation_obj_dp = os.path.join(self.tmp_objects_dp, sub, animation)  # TODO - Presuming this dir structure
        assert_new_dir(Path(animation_obj_dp).resolve(), parents=True)

        animation_metadata_dp = os.path.join(self.tmp_metadata_dp, sub,
                                             animation)  # TODO - Presuming this dir structure
        assert_new_dir(Path(animation_metadata_dp).resolve(), parents=True)

        collada_animation = Loader(animation_dp, is_converted_file=True, log_path=LOGGER_PATH,
                                   log_bind_shape_matrix_path=BIND_LOGGER)

        collada_animation.save_animation_keyframes_as_obj_files(animation_obj_dp)
        collada_animation.save_animation_meta_data(animation_metadata_dp)

        animation_validation_fp = self.record_dp / sub
        animation_validation_fp.mkdir(parents=True, exist_ok=True)
        animation_validation_fp /= (animation + '.txt')

        with open(animation_validation_fp, 'w') as writer:
            writer.writelines('\n')

        return True

    # def _transfer_local_animation_to_out_dir(self, sub, animation_name):
    #     animation_tmp_objects_dp = Path(os.path.join(self.tmp_objects_dp, sub, animation_name)).resolve()
    #     animation_objects_dp = Path(os.path.join(self.objects_dp, sub, animation_name)).resolve()
    #
    #     animation_tmp_metadata_dp = Path(os.path.join(self.tmp_metadata_dp, sub, animation_name)).resolve()
    #     animation_metadata_dp = Path(os.path.join(self.metadata_dp, sub, animation_name)).resolve()
    #
    #     # if animation_objects_dp.is_dir():
    #     #     shutil.rmtree(animation_objects_dp)
    #     #     time.sleep(2)  # TODO - find something smarter
    #     #
    #     # if animation_metadata_dp.is_dir():
    #     #     shutil.rmtree(animation_metadata_dp)
    #     #     time.sleep(2)  # TODO - find something smarter
    #
    #     shutil.copytree(src=animation_tmp_objects_dp, dst=animation_objects_dp)
    #     shutil.rmtree(animation_tmp_objects_dp, ignore_errors=True)  # Clean up
    #
    #     shutil.copytree(src=animation_tmp_metadata_dp, dst=animation_metadata_dp)
    #     shutil.rmtree(animation_tmp_metadata_dp, ignore_errors=True)  # Clean up

    def get_animations(self, sub):
        animation_validation_fp = self.record_dp / sub
        all_animation = [os.path.splitext(animation)[0] for animation in self.animations_per_subject(sub)]
        if animation_validation_fp.is_dir():
            animations_done = [os.path.splitext(animation)[0] for animation in os.listdir(animation_validation_fp)]
            # list substarction
            todo_animation = [animation for animation in all_animation if animation not in animations_done]
            return todo_animation
        else:
            return all_animation

    def animations_per_subject(self, sub):
        """
        :param sub: The subject name
        :return: All validation group names given the subject
        """
        # TODO check if all ends with .fpx
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator


if __name__ == '__main__':
    banner('MIXAMO extracting')
    m = DataCreator()
    m.extract_subject(sub='010')
    print('Ido')
    print('Ya maniak')

# ----------------------------------------------------------------------------------------------------------------------#
#                               Instructions on how to mount server for Linux Machines
# ----------------------------------------------------------------------------------------------------------------------#
# r"/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST"
# Mounting Instructions: [shutil does not support samba soft-links]
# sudo apt install samba
# sudo apt install cifs-utils
# sudo mkdir /usr/samba_mount
# sudo mount -t cifs -o auto,username=mano,uid=$(id -u),gid=$(id -g) //132.68.36.59/data /usr/samba_mount/
# To install CUDA runtime 10.2 on the Linux Machine, go to: https://developer.nvidia.com/cuda-downloads
# And choose the deb(local) version
