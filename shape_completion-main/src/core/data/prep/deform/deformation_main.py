import os
import sys
import shutil
from abc import ABC

import numpy as np
import pickle
import json
from tqdm import tqdm
from data.prep.deform.deformations import Projection, SemanticCuts
from pathlib import Path
import time
import tempfile

from util.strings import banner, print_yellow, print_red, title
from geom.mesh.op.cpu.remesh import clean_mesh
from geom.mesh.io.base import read_obj_verts, read_ply_verts, read_off, read_ply, read_mesh, read_obj
from geom.mesh.vis.base import plot_mesh_montage
from util.fs import assert_new_dir
from util.func import all_subclasses

# ---------------------------------------------------------------------------------------------------------------------#
#                                                      TODO
# ---------------------------------------------------------------------------------------------------------------------#
# [1] Add in AMASS projector
# [2] Replace Projection with Hidden point removal from Open3D
# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Globals
# ---------------------------------------------------------------------------------------------------------------------#
ROOT = (Path(__file__).parents[0]).resolve()  # Current dir
SYNTHETIC_DATA_ROOT = ROOT.parents[4] / 'data' / 'synthetic'
REALISTIC_DATA_ROOT = ROOT.parents[4] / 'data' / 'scan'
OUTPUT_ROOT = ROOT / 'deformation_outputs'
COLLATERALS_DIR = ROOT / 'collaterals'
DEBUG_PLOT = True


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def project_mixamo_main(n_azimutals=10, n_azimutal_subset=None, n_elevations=11, n_elevation_subset=None):
    if os.name == 'nt':
        in_dp = Path(r'Z:\ShapeCompletion\Mixamo\Blender\MPI-FAUST')
    else:  # Presuming Linux
        in_dp = Path(r"/usr/samba_mount/ShapeCompletion/Mixamo/Blender/MPI-FAUST")

    banner('MIXAMO Projection')
    deformer = Projection(n_azimutals=n_azimutals, n_azimutal_subset=n_azimutal_subset, n_elevations=n_elevations,
                          n_elevation_subset=n_elevation_subset)
    m = MixamoCreator(deformer, in_dp, shape_frac_from_vgroup=1)
    for i, sub in enumerate(m.subjects()):
        m.deform_subject(sub=sub)


def project_dataset_main(dataset_name='SMAL', n_azimutals=10, n_azimutal_subset=None, n_elevations=11,
                         n_elevation_subset=None, r=2.5):
    creators = [cls for cls in all_subclasses(DataCreator, non_inclusive=True) if
                cls.dataset_name().lower().startswith(dataset_name.lower())]
    assert creators, f"No relevant creators whose name starts with {dataset_name}"
    banner(f'{dataset_name.upper()} Projection')
    deformer = Projection(n_azimutals=n_azimutals, n_azimutal_subset=n_azimutal_subset,
                          n_elevations=n_elevations, n_elevation_subset=n_elevation_subset, r=r)
    for c in creators:
        m = c(deformer)
        for i, sub in enumerate(m.subjects()):
            m.deform_subject(sub=sub)


def semantic_cut_dataset_main(dataset_name='SMAL', src_vi='rand_str', cut_at=(0.13, 0.217), dist_map='graph',
                              num_cuts=10, remove_small_comps_override=None, direct_dist_mode=False):  # Todo
    creators = [cls for cls in all_subclasses(DataCreator, non_inclusive=True) if
                cls.__name__.lower().startswith(dataset_name.lower())]
    assert creators, f"No relevant creators whose name starts with {dataset_name}"
    banner(f'{dataset_name.upper()} Projection')
    deformer = SemanticCuts(src_vi=src_vi, cut_at=cut_at, dist_map=dist_map, return_only_mask=True, num_cuts=num_cuts,
                            direct_dist_mode=direct_dist_mode)
    for c in creators:
        if remove_small_comps_override is None:
            if 'scan' in c.dataset_name().lower():  # TODO - Quick hack for FaustScans
                deformer.remove_small_comps = False
            else:
                deformer.remove_small_comps = True
        else:
            deformer.remove_small_comps = remove_small_comps_override
        m = c(deformer)
        for i, sub in enumerate(m.subjects()):
            # if i in (0,1,2):
            #     continue
            m.deform_subject(sub=sub)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
class DataCreator:
    MIN_VGROUP_SUCCESS_FRAC = 0.5  # Under this, the validation group will be considered a failure
    MIN_VGROUP_SIZE = 10  # Under this, the validation group will be considered too small to take
    MAX_FAILED_VGROUPS_BEFORE_RESET = 7  # For PyRender deformation only - maximum amount of failures before reset

    TMP_ROOT = Path(tempfile._get_default_tempdir())
    OUT_ROOT = Path(OUTPUT_ROOT).resolve()  # May be overridden
    OUT_IS_A_NETWORK_PATH = False
    RECORD_ROOT = OUT_ROOT  # May be overridden

    COLLAT_DP = Path(COLLATERALS_DIR)

    def __init__(self, deformer, in_dp=None, shape_frac_from_vgroup=1):

        # Sanity:
        self.in_dp = self.expected_in_dp() if in_dp is None else Path(in_dp).resolve()
        assert self.in_dp.is_dir(), f"Could not find directory {self.in_dp}"

        # Set deformer:
        self.deformer = deformer
        self.shape_frac_from_vgroup = shape_frac_from_vgroup
        ds_id, deform_id = self.dataset_name(), self.deform_name()
        try:
            self.read_shape_func = getattr(self, f'read_shape_for_{deformer.name(False).lower()}')
        except:
            # TODO -  A bit hacky and breaks encapsulation, but meh
            self.read_shape_func = getattr(self, 'read_shape_for_projection')

        # Create dirs for the dataset + specific deformation:
        self.out_dp = self.OUT_ROOT / ds_id / deform_id / 'outputs'
        self.out_dp.mkdir(parents=True, exist_ok=True)
        print(f'Target output directory: {self.out_dp}')

        self.tmp_out_dp = self.out_dp  # The write is not temporary - it is final
        if self.OUT_IS_A_NETWORK_PATH:
            assert self.TMP_ROOT.is_dir(), f"Could not find directory {self.TMP_ROOT}"
            self.tmp_out_dp = self.TMP_ROOT / ds_id / deform_id
            self.tmp_out_dp.mkdir(parents=True, exist_ok=True)
        if deformer.needs_validation():
            self.record_dp = self.RECORD_ROOT / ds_id / deform_id / 'record'
            self.record_dp.mkdir(parents=True, exist_ok=True)
            print(f'Target validation-records directory: {self.record_dp}')

    @classmethod
    def dataset_name(cls):
        return cls.__name__[:-7]  # Without the Creator

    def deform_name(self):
        return f'{self.deformer.name()}_vg_frac_{self.shape_frac_from_vgroup}'.replace('.', '_')

    def deform_shape(self, fp):
        v, f = self.read_shape_func(fp)
        if os.name == 'nt' and self.deformer.name() == 'projection':  # TODO - Dirty Hack - should be removed to save complexity
            return [{'mask': v, 'angle_id': i} for i in range(self.deformer.num_expected_deformations())]
        deformed = self.deformer.deform(v, f)

        if DEBUG_PLOT:
            vs, labels = [], []
            for i, d in enumerate(deformed):
                if d is not None:
                    vs.append(v[d['mask'], :])
                    if self.deformer.name() == 'projection':
                        labels.append(f'[{i}]ele_{d["ele"][1] * 180 / np.pi:.0f}\nazi_{d["azi"][1] * 180 / np.pi:.0f}')
                    else:  # TODO - Generalize this
                        labels = None
            plot_mesh_montage(vb=vs, strategy='spheres', grid_on=True, labelb=labels)
        return deformed

    def deform_subject(self, sub):
        banner(f'{self.dataset_name().strip().upper()}' +
               title(f' Dataset :: Subject {sub} :: Deformation {self.deform_name()} Commencing'))
        (self.tmp_out_dp / sub).mkdir(exist_ok=True)  # TODO - Presuming this dir structure

        if self.deformer.needs_validation():
            self._deform_subject_validated(sub)
        else:
            self._deform_subject_unvalidated(sub)
        banner(f'Deformation of Subject {sub} - COMPLETED')

    def _deform_subject_validated(self, sub):
        vgd, vgd_fp = self._vgroup_dict(sub)
        vgs_todo = [k for k, v in vgd.items() if v < 1]
        if vgs_todo:
            total_failures = 0
            for vgi, vg in tqdm(enumerate(vgs_todo), file=sys.stdout, total=len(vgs_todo), unit='vgroup'):

                comp_frac = self._deform_and_locally_save_vgroup(sub, vg)
                if comp_frac >= self.MIN_VGROUP_SUCCESS_FRAC:
                    if self.OUT_IS_A_NETWORK_PATH:
                        self._transfer_local_vgroup_to_out_dir(sub, vg)

                    vgd[vg] = comp_frac  # Save the VGD to local area:
                    with open(vgd_fp, 'w') as handle:
                        json.dump(vgd, handle, sort_keys=True, indent=4)

                elif comp_frac >= 0:
                    print_red(f'\nWARNING - Deformation success rate for {vg} is below threshold - skipping')
                    total_failures += 1
                    if total_failures == self.MAX_FAILED_VGROUPS_BEFORE_RESET:
                        self.deformer.reset()
                else:  # -1 case
                    print_yellow(f'\nWARNING - Validation Group {vg} has too few shapes - skipping')
            self._print_vgd_statistics(vgd, print_vgd=False)

    def _deform_subject_unvalidated(self, sub):
        vgs_todo = self.vgroups_per_subject(sub)
        for vg in tqdm(vgs_todo, file=sys.stdout, total=len(vgs_todo), unit='vgroup'):
            vg_dp = self.out_dp / sub / vg  # TODO - Presuming this dir structure
            vg_dp.mkdir(exist_ok=True, parents=True)
            for fp in self.shape_fps_per_vgroup(sub, vg):
                deformed = self.deform_shape(fp)
                for i, d in enumerate(deformed):  # Expecting a list of outputs
                    file_name = self.deformer.shape_filename_given_shape_name(self.shape_name_from_fp(fp), i)
                    np.savez(vg_dp / file_name, **d)  # TODO - Presuming npz save

    def _deform_and_locally_save_vgroup(self, sub, vg):

        shape_fps = self.shape_fps_per_vgroup(sub, vg)
        assert len(shape_fps) > 0, "Empty Validation group "
        if len(shape_fps) < self.MIN_VGROUP_SIZE:
            return -1  # Empty

        if self.shape_frac_from_vgroup != 1:  # Decimate
            requested_number = int(self.shape_frac_from_vgroup * len(shape_fps))
            num_to_take = max(requested_number, self.MIN_VGROUP_SIZE)
            shape_fps = np.random.choice(shape_fps, size=num_to_take, replace=False)

        # Create all needed directories:
        # TODO - Presuming this dir structure
        vg_dp = self.tmp_out_dp / sub / vg.split('.')[0] / vg # For DFAUST SCANS 
        # vg_dp = self.tmp_out_dp / sub / vg # For all others
        assert_new_dir(vg_dp, parents=True)

        completed = 0
        total = len(shape_fps) * self.deformer.num_expected_deformations()
        # Project:
        for fp in shape_fps:
            deformed, i = self.deform_shape(fp), 0
            for d in deformed:
                if d is not None:
                    file_name = self.deformer.shape_filename_given_shape_name(self.shape_name_from_fp(fp), i)
                    np.savez(vg_dp / file_name, **d)  # TODO - Presuming npz save
                    i += 1
            completed += i
        return completed / total

    def _transfer_local_vgroup_to_out_dir(self, sub, vg):
        vg_tmp_dp = self.tmp_out_dp / sub / vg
        vg_out_dp = self.out_dp / sub / vg
        if vg_out_dp.is_dir():
            shutil.rmtree(vg_out_dp)
            time.sleep(2)  # TODO - find something smarter
        shutil.copytree(src=vg_tmp_dp, dst=vg_out_dp)
        shutil.rmtree(vg_tmp_dp, ignore_errors=True)  # Clean up

    def _vgroup_dict(self, sub):
        vgd_fp = self.record_dp / f'{sub}_{self.deform_name()}_validation_dict.json'
        if vgd_fp.is_file():  # Validation Group Dict exists
            with open(vgd_fp, 'r') as handle:
                vgd = json.load(handle)
            vgs_todo = [k for k, v in vgd.items() if v < 1]
            if vgs_todo:
                self._print_vgd_statistics(vgd)
        else:  # Create it:
            vgd = {vg: 0 for vg in self.vgroups_per_subject(sub)}
            with open(vgd_fp, 'w') as handle:
                json.dump(vgd, handle, sort_keys=True, indent=4)  # Dump as JSON for readability
                print(f'Saved validation group cache for subject {sub} at {vgd_fp}')
        return vgd, vgd_fp

    @staticmethod
    def _print_vgd_statistics(vgd, print_vgd=False):
        # Analysis:
        print('Cache Status:')
        empty, completed, partial = 0, 0, 0
        total = len(vgd)
        for comp_frac in vgd.values():
            empty += (comp_frac == 0)
            completed += (comp_frac == 1)
            partial += (comp_frac != 1 and comp_frac != 0)
        print(f'\t* Completed Validation Groups: {completed}/{total}')
        print(f'\t* Empty Validation Groups: {empty}/{total}')
        print(f'\t* Partial Validation Groups: {partial}/{total}')
        if print_vgd:
            print(json.dumps(vgd, indent=4, sort_keys=True))  # JSON->String

    def subjects(self):
        """
        :return: All subject names
        """
        raise NotImplementedError

    def vgroups_per_subject(self, sub):
        """
        :param sub: The subject name
        :return: All validation group names given the subject
        """
        raise NotImplementedError

    def shape_fps_per_vgroup(self, sub, vg):
        """
        :param sub: The subject name
        :param vg: The validation group name
        :return: All mesh file paths given shape name & validation group name
        """
        raise NotImplementedError

    @staticmethod
    def shape_name_from_fp(shape_fp):
        """
        :param shape_fp: Given the shape file path
        :return: The shape name
        """
        return shape_fp.with_suffix('').name  # Default parse

    @classmethod
    def expected_in_dp(cls):
        """
        :return: The expected input directory path
        """
        raise NotImplementedError
    # Remember to implement the read_shape_from_<YOUR DEFORMATION NAME>


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class MixamoCreator(DataCreator):
    RECORD_ROOT = Path(OUTPUT_ROOT)  # Override
    if os.name == 'nt':  # Override
        OUT_ROOT = Path(r'Z:\ShapeCompletion\Mixamo')
    else:  # Presuming Linux
        OUT_ROOT = Path(r"/usr/samba_mount/ShapeCompletion/Mixamo")
    OUT_IS_A_NETWORK_PATH = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.deformer, Projection):
            with open(self.COLLAT_DP / 'SMPL_faces.pkl', "rb") as f_file:
                self.f = pickle.load(f_file)  # Already int32
                self.f.flags.writeable = False  # Make this a read-only numpy array

    def subjects(self):

        return tuple(f'0{i}0' for i in range(10))

    def vgroups_per_subject(self, sub):

        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator

    def shape_fps_per_vgroup(self, sub, vg):

        fp = self.in_dp / sub / vg
        assert fp.is_dir(), f"Could not find path {fp}"
        return list(fp.glob('*.obj'))  # glob actually returns a generator

    def read_shape_for_projection(self, fp):
        # PyRender needs a multiplications of 100 to go smoothly. TODO - Assert this
        v = read_obj_verts(fp) * 100
        return v, self.f

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'{cls.dataset_name()}PyProj' / 'full'


class MiniMixamoCreator(MixamoCreator):
    OUT_IS_A_NETWORK_PATH = False
    OUT_ROOT = Path(OUTPUT_ROOT).resolve()  # May be overridden
    RECORD_ROOT = OUT_ROOT  # May be overridden

    def subjects(self):
        return tuple(f'0{i}0' for i in range(8))

    def read_shape_for_projection(self, fp):
        v = read_obj_verts(fp)
        return v, self.f

    def read_shape_for_semanticcuts(self, fp):
        return read_obj(fp)

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'MiniMixamoPyProj13' / 'full'


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

class SMALCreator(DataCreator, ABC):
    MIN_VGROUP_SUCCESS_FRAC = 1
    MIN_VGROUP_SIZE = 0

    def __init__(self, deformer, in_dp=None):
        super().__init__(deformer, in_dp, 1)

        if isinstance(self.deformer, Projection):
            with open(self.COLLAT_DP / 'SMAL_faces.pkl', "rb") as f_file:
                self.f = pickle.load(f_file)  # Already int32
                self.f.flags.writeable = False  # Make this a read-only numpy array

    def subjects(self):
        return 'cats', 'dogs', 'horses', 'cows', 'hippos'

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return [f.with_suffix('').name for f in fp.glob('*')]

    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / sub / f'{vg}.ply']

    def read_shape_for_projection(self, fp):
        return read_ply_verts(fp), self.f

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'{cls.dataset_name()}PyProj' / 'full'


class SMALTrainCreator(SMALCreator):
    pass  # Naming based change


class SMALValdCreator(SMALCreator):
    pass  # Naming based change


class SMALTestCreator(SMALCreator):
    pass  # Naming based change


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
class BaseFaustCreator(DataCreator, ABC):
    MIN_VGROUP_SUCCESS_FRAC = 1
    MIN_VGROUP_SIZE = 0

    def __init__(self, deformer, in_dp=None):
        super().__init__(deformer, in_dp, 1)

    def subjects(self):
        return [str(i) for i in range(10)]

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return [f.with_suffix('').name for f in fp.glob('*')]


class FaustCreator(BaseFaustCreator):
    def vgroups_per_subject(self, sub):
        return [str(i) for i in range(10)]

    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / f'tr_reg_0{sub}{vg}.off']

    @staticmethod
    def read_shape_for_projection(fp):
        return read_off(fp)

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'FaustPyProj' / 'full'


class FaustTestScanCreator(BaseFaustCreator):
    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / sub / f'{vg}.off']

    @classmethod
    def expected_in_dp(cls):
        return REALISTIC_DATA_ROOT / f'{cls.dataset_name()}PyProj' / 'full'

    @staticmethod
    def read_shape_for_projection(fp):
        return clean_mesh(*read_off(fp))


class FaustTrainScanCreator(BaseFaustCreator):
    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / sub / f'{vg}.ply']

    @classmethod
    def expected_in_dp(cls):
        return REALISTIC_DATA_ROOT / f'{cls.dataset_name()}PyProj' / 'full'

    @staticmethod
    def read_shape_for_projection(fp):
        v, f = clean_mesh(*read_ply(fp))
        return v, f


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
class DFaustCreator(DataCreator):
    MAX_FAILED_VGROUPS_BEFORE_RESET = 1

    def subjects(self):
        return os.listdir(self.in_dp)

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator

    def shape_fps_per_vgroup(self, sub, vg):
        fp = self.in_dp / sub / vg
        assert fp.is_dir(), f"Could not find path {fp}"
        return list(fp.glob('*.OFF'))  # glob actually returns a generator

    @staticmethod
    def read_shape_for_projection(fp):
        return read_off(fp)

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'DFaustPyProj' / 'full'


class DFaustAccessoriesCreator(DataCreator):
    def subjects(self):
        return ['1']

    def vgroups_per_subject(self, sub):
        fp = self.in_dp
        assert fp.is_dir(), f"Could not find path {fp}"
        return os.listdir(fp)  # glob actually returns a generator

    def shape_fps_per_vgroup(self, sub, vg):
        return [self.in_dp / vg]

    @staticmethod
    def read_shape_for_projection(fp):
        return read_mesh(fp)

    @classmethod
    def expected_in_dp(cls):
        return SYNTHETIC_DATA_ROOT / f'DFaustAccessories' / 'dressed'


class DFaustScansCreator(DataCreator):
    MIN_VGROUP_SUCCESS_FRAC = 1
    MIN_VGROUP_SIZE = 0
    MAX_FAILED_VGROUPS_BEFORE_RESET = 2

    def subjects(self):
        return ['50026', '50027']

    def vgroups_per_subject(self, sub):
        fp = self.in_dp / sub
        assert fp.is_dir(), f"Could not find path {fp}"
        # Due to the large size of the vertices, we validate on every mesh
        return [file.stem for file in fp.rglob('*.ply')]

    def shape_fps_per_vgroup(self, sub, vg):
        seq_name = vg.split('.')[0]
        return [self.in_dp / sub / seq_name / f'{vg}.ply']

    @staticmethod
    def read_shape_for_projection(fp):
        return clean_mesh(*read_ply(fp))

    @classmethod
    def expected_in_dp(cls):
        return REALISTIC_DATA_ROOT / f'DFaustScans'


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    DEBUG_PLOT = False
    project_dataset_main(dataset_name='DFaustScans', n_azimutals=10, n_elevations=11, r=2.5)
    # project_dataset_main(dataset_name='DFaustAccessories', n_azimutals=10, n_elevations=11, r=2.5)
    # project_dataset_main(dataset_name='SMAL', n_azimutals=10, n_elevations=11, r=2.5)
    # semantic_cut_dataset_main(dataset_name='DFaustAccessories',num_cuts=10,dist_map='euclidean_cloud',remove_small_comps_override=False,
    #                           cut_at=[0.12,0.25],src_vi='rand_vi',direct_dist_mode=True)
    # semantic_cut_dataset_main(dataset_name='DFaustScans', num_cuts=4)
# ---------------------------------------------------------------------------------------------------------------------#
#                               Instructions on how to mount server for Linux Machines
# ---------------------------------------------------------------------------------------------------------------------#
# r"/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST"
# Mounting Instructions: [shutil does not support samba soft-links]
# sudo apt install samba
# sudo apt install cifs-utils
# sudo mkdir /usr/samba_mount
# sudo mount -t cifs -o auto,username=oshri.halimi,uid=$(id -u),gid=$(id -g) //gip-main/data /usr/samba_mount/
# To install CUDA runtime 10.2 on the Linux Machine, go to: https://developer.nvidia.com/cuda-downloads
# On Right-Door - only version 18.04 deb(local) for x86_64 works - so it is better to stick to this.
