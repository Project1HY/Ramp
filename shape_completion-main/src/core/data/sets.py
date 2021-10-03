from data.base import *
from data.index import HierarchicalIndexTree
from geom.mesh.io.base import *
from geom.mesh.op.cpu.remesh import clean_mesh
import os

try:
    # This snippet is needed to solve the unpickling error. See shorturl.at/jktw2
    if __name__ != '__main__':
        # noinspection PyUnresolvedReferences
        import __main__

        __main__.HierarchicalIndexTree = HierarchicalIndexTree
except ImportError:
    pass


# TODO - Consider adding Surreal Dataset
# TODO - Consider migrating to lmdb or h5py
# TODO - Allow extraction of an orbital subset by angle - its the same ids every time...

# ----------------------------------------------------------------------------------------------------------------------
#                                                       FAUST
# ----------------------------------------------------------------------------------------------------------------------

class Faust(ParametricCompletionDataset):
    def __init__(self, data_dir_override, deformation):
        self.num_proj_per_pose = deformation.num_projections()
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation,
                         cls='synthetic', suspected_corrupt=False)

    def _construct_hit(self):  # Override
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(10):
                hit[str(sub_id)][str(pose_id)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.off'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'tr_reg_0{hi[0]}{hi[1]}_{hi[2]}.npz'

    def _hi2proj_path_azimuthal_projections(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_00{hi[2]}.npz'


# ----------------------------------------------------------------------------------------------------------------------
#                                                       FAUST SCANS
# ----------------------------------------------------------------------------------------------------------------------

class FaustScanBase(CompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation, num_pose_per_sub=10):
        self.num_proj_per_pose = deformation.num_projections()
        self.num_pose_per_sub = num_pose_per_sub
        super().__init__(data_dir_override=data_dir_override, deformation=deformation,
                         cls='scan', suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(self.num_pose_per_sub):
                hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)


class FaustTrainScans(FaustScanBase):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, num_pose_per_sub=10)

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'tr_scan_{hi[1]:>03}.ply'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'tr_scan_{hi[1]:>03}' / f'tr_scan_{hi[1]:>03}_{hi[2]}.npz'

    def _full_path2data_default(self, fp):
        return clean_mesh(*read_ply(fp))


class FaustTestScans(FaustScanBase):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, num_pose_per_sub=20)

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'test_scan_{hi[1]:>03}.off'

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'test_scan_{hi[1]:>03}' / f'test_scan_{hi[1]:>03}_{hi[2]}.npz'

    def _full_path2data_default(self, fp):
        return clean_mesh(*read_off(fp))


# ----------------------------------------------------------------------------------------------------------------------
#                                                       AMASS
# ----------------------------------------------------------------------------------------------------------------------

class AmassBase(ParametricCompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}_projectionID_{hi[2]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}.OFF'


class AmassTrain(AmassBase, ABC):
    pass


class AmassVald(AmassBase, ABC):
    pass


class AmassTest(AmassBase, ABC):
    pass


# ----------------------------------------------------------------------------------------------------------------------
#                                                       SMAL
# ----------------------------------------------------------------------------------------------------------------------

class SMALBase(ParametricCompletionDataset, ABC):
    def __init__(self, data_dir_override, deformation, num_poses_per_sub):
        self._num_poses_per_sub = num_poses_per_sub
        super().__init__(n_verts=3889, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in ['cats', 'dogs', 'horses', 'cows', 'hippos']:
            hit[sub_id] = {}
            for pose_id in range(self._num_poses_per_sub):
                hit[sub_id][str(pose_id)] = 110
        return HierarchicalIndexTree(hit, in_memory=True)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'{hi[1]}' / f'{hi[1]}_{hi[2]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / f'{hi[1]}.ply'


class SMALTrain(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=20, *args, **kwargs)


class SMALVald(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=10, *args, **kwargs)


class SMALTest(SMALBase):
    def __init__(self, *args, **kwargs):
        super().__init__(num_poses_per_sub=10, *args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust
# ----------------------------------------------------------------------------------------------------------------------

class DFaust(ParametricCompletionDataset):
    def __init__(self, data_dir_override, deformation):
        super().__init__(n_verts=6890, data_dir_override=r"R:\Mano\data\DFaust\DFaust", deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:>05}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>05}.OFF'

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>05}_{hi[3]}.npz'


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust Scans
# ----------------------------------------------------------------------------------------------------------------------

class DFaustScans(CompletionDataset, ABC):  # TODO
    def __init__(self, data_dir_override, deformation, num_pose_per_sub=10):
        self.num_proj_per_pose = deformation.num_projections()
        self.num_pose_per_sub = num_pose_per_sub
        super().__init__(data_dir_override=data_dir_override, deformation=deformation,
                         cls='scan', suspected_corrupt=False)

    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[str(sub_id)] = {}
            for pose_id in range(self.num_pose_per_sub):
                hit[str(sub_id)][str(pose_id + sub_id * self.num_pose_per_sub)] = self.num_proj_per_pose
        return HierarchicalIndexTree(hit, in_memory=True)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       DFaust Scans
# ----------------------------------------------------------------------------------------------------------------------
class DFaustAccessories(CompletionDataset):
    def __init__(self, data_dir_override, deformation):
        super().__init__(data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=False)

    def _construct_hit(self):
        self.num_proj_per_pose = self._deformation.num_projections()
        return HierarchicalIndexTree({file.stem: self.num_proj_per_pose for file in self._full_dir.glob('*')},
                                     in_memory=True)

    def _hi2proj_path_default(self, hi):
        return self._proj_dir / hi[0] / f'{hi[0]}_{hi[1]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / f'{hi[0]}.obj'


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Mixamo
# ----------------------------------------------------------------------------------------------------------------------

class MixamoBase(ParametricSkinningDataset, ABC):
    def __init__(self, data_dir_override, deformation, suspected_corrupt):
        super().__init__(n_verts=6890, data_dir_override=data_dir_override, deformation=deformation, cls='synthetic',
                         suspected_corrupt=suspected_corrupt)

    def _hi2proj_path_default(self, hi):
        for i in range(10):  # Num Angles. Hacky - but works. TODO - Should we rename?
            fp = self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>03}_{hi[3]}_angi_{i}.npz'
            if fp.is_file():
                return fp
        else:
            raise AssertionError

    def _hi2proj_path_semantic_cuts(self, hi):
        return self._proj_dir / hi[0] / hi[1] / f'{hi[2]:>03}_{hi[3]}.npz'

    def _hi2full_path_default(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>03}.obj'

    def _hi2joint_path(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:>03}.pkl'

    def _full_path2data_default(self, fp):
        return read_obj_verts(fp)

class MixamoSkinnedGIP(MixamoBase):  # Should be: MixamoPyProj_2k_10ang_1fr
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GIP
            data_dir_override = MIXAMO_PATH_GIP
        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)

class MixamoSkinnedGaon9(MixamoBase):  # Should be: MixamoPyProj_2k_10ang_1fr
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9
        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)


class MiniMixamoSkinned13(MixamoBase):  # Should be: MiniMixamoPyProj_2k_10ang_1fr
    def __init__(self, *args, **kwargs):
        super().__init__(suspected_corrupt=False, *args, **kwargs)

class MiniMixamoSkinnedSingleSubject(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):

        self.subjects = ['000', '010', '020', '030', '040', '050', '060', '070', '080', '090']
        self.desired_subject = ['000']
        self.ignored_subjects = list(set(self.subjects) - set(self.desired_subject))

        if data_dir_override is None:
            from cfg import MINI_MIXAMO_PATH_GAON9
            data_dir_override = MINI_MIXAMO_PATH_GAON9

        super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit : HierarchicalIndexTree = super()._construct_hit()
        hit = hit.remove_ids_by_depth(depth=1, goners=self.ignored_subjects)
        # we just toke Bartending
        hit = hit.remove_ids_by_depth(depth=2, goners=['Air Squat', 'Arm Stretching', 'Sprint Turn', 'Standing', 'Stomping', 'Talking At Watercooler', 'Throwing', 'Turn 90 Left', 'Turn 90 Right', 'Victory', 'Walking', 'Waving'])
        # hit = hit.remove_ids_by_depth(depth=3, goners=[str(num).zfill(3) for num in range(2, 450)])
        return hit


class MiniMixamoSkinned(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):
        if data_dir_override is None:
            from cfg import MINI_MIXAMO_PATH_GAON9
            data_dir_override = MINI_MIXAMO_PATH_GAON9
        super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit : HierarchicalIndexTree = super()._construct_hit()
        return hit



# class MixamoSkinned(MixamoBase):
#     def __init__(self, data_dir_override, *args, **kwargs):
#         if data_dir_override is None:
#             from cfg import MIXAMO_PATH_GAON9
#             data_dir_override = MIXAMO_PATH_GAON9
#         super().__init__(suspected_corrupt=False, data_dir_override=data_dir_override, *args, **kwargs)
#
#     def _construct_hit(self):
#         hit : HierarchicalIndexTree = super()._construct_hit()
#         return hit


class MixamoSkinned(MixamoBase):
    def __init__(self, data_dir_override, *args, **kwargs):
        self.subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9

        super().__init__(suspected_corrupt=True, data_dir_override=data_dir_override, *args, **kwargs)
    #TODO - do it smarter
    def _construct_hit(self):
        hit : HierarchicalIndexTree = super()._construct_hit()
        from cfg import BROKEN_ANIMATION
        broken_dir_path = BROKEN_ANIMATION
        all_animations = set([])
        for sub in self.subjects:
            with open(os.path.join(broken_dir_path, sub + '.txt'), "r") as f:
                broken_animation = f.read()
            broken_animation = broken_animation.split("\n")
            all_animations = all_animations.union(set(broken_animation))
        hit = hit.remove_ids_by_depth(depth=2, goners=list(all_animations))
        return hit


class MixamoSkinnedSingleSubject(MixamoSkinned):
    def __init__(self, data_dir_override, *args, **kwargs):
        self.subjects = [str(sub).zfill(3) for sub in range(0, 100, 10)]
        self.desired_subject = ['000']
        self.ignored_subjects = list(set(self.subjects) - set(self.desired_subject))

        if data_dir_override is None:
            from cfg import MIXAMO_PATH_GAON9
            data_dir_override = MIXAMO_PATH_GAON9

        super().__init__(data_dir_override, *args, **kwargs)

    def _construct_hit(self):
        hit : HierarchicalIndexTree = super()._construct_hit()
        hit = hit.remove_ids_by_depth(depth=1, goners=self.ignored_subjects)
        return hit


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------

class DatasetMenu:
    # TODO - once we have the whole grid, this could be simplified
    _IMPLEMENTED = {
        'AmassTestProj': (AmassTest, AzimuthalProjection()),
        'AmassValdProj': (AmassVald, AzimuthalProjection()),
        'AmassTrainProj': (AmassTrain, AzimuthalProjection()),

        'SMALTestEleProj': (SMALTest, OrbitalProjection()),
        'SMALValdEleProj': (SMALVald, OrbitalProjection()),
        'SMALTrainEleProj': (SMALTrain, OrbitalProjection()),
        'FaustProj': (Faust, AzimuthalProjection()),
        'FaustEleProj': (Faust, OrbitalProjection()),
        'FaustSemCuts': (Faust, SemanticCut()),

        'DFaustProj': (DFaust, AzimuthalProjection()),
        # 'DFaustEleProj': (Faust, OrbitalProjection()),
        'DFaustSemCuts': (DFaust, SemanticCut(n_expected_proj=4)),

        # 'DFaustAccessories' : (DFaustAccessories, AzimuthalProjection()),
        'DFaustAccessoriesEleProj': (DFaustAccessories, OrbitalProjection()),
        'DFaustAccessoriesSemCuts': (DFaustAccessories, SemanticCut()),

        'MiniMixamoSkinned13Proj': (MiniMixamoSkinned13, AzimuthalProjection(n_expected_proj=2)),

        # the following loaders works on Gaon9
        'MiniMixamoSkinnedSingleSubject': (MiniMixamoSkinnedSingleSubject, AzimuthalProjection(n_expected_proj=2)),
        'MiniMixamoSkinned': (MiniMixamoSkinned, AzimuthalProjection(n_expected_proj=2)),

        'MixamoSkinnedSingleSubject': (MixamoSkinnedSingleSubject, AzimuthalProjection(n_expected_proj=2)),
        'MixamoSkinned': (MixamoSkinned, AzimuthalProjection(n_expected_proj=2)),

        # 'MiniMixamoSkinned13EleProj': (MiniMixamoSkinned13, OrbitalProjection()),
        'MiniMixamoSkinned13SemCuts': (MiniMixamoSkinned13, SemanticCut(n_expected_proj=4)),

        # 'FaustTrainScansProj': (FaustTrainScans, AzimuthalProjection()),
        'FaustTrainScansEleProj': (FaustTrainScans, OrbitalProjection()),
        'FaustTrainScansSemCuts': (FaustTrainScans, SemanticCut()),

        # 'FaustTestScansProj': (FaustTestScans, AzimuthalProjection()),
        'FaustTestScansEleProj': (FaustTestScans, OrbitalProjection()),
        'FaustTestScansSemCuts': (FaustTestScans, SemanticCut()),
        #
        # 'DFaustScansProj': (DFaustScans, AzimuthalProjection()),
        # 'DFaustScansEleProj': (DFaustScans, OrbitalProjection()),
        # 'DFaustScansSemCuts': (DFaustScans, SemanticCut()),

        'MixamoSkinnedGIPProj': (MixamoSkinnedGIP, AzimuthalProjection(n_expected_proj=2)),
        'MixamoSkinnedGaon9Proj': (MixamoSkinnedGaon9, AzimuthalProjection(n_expected_proj=2)),
    }

    @classmethod
    def all_sets(cls):
        return tuple(cls._IMPLEMENTED.keys())

    @classmethod
    def order(cls, dataset_name, data_dir_override=None):
        if dataset_name in cls._IMPLEMENTED:
            tup = cls._IMPLEMENTED[dataset_name]
            return tup[0](data_dir_override=data_dir_override, deformation=tup[1])
        else:
            raise ValueError(f'Could not find dataset {dataset_name} - check spelling')

    @classmethod
    def skinning_sets(cls):
        # TODO - create more filters?
        return tuple(k for k, v in cls._IMPLEMENTED.items() if issubclass(v[0], ParametricSkinningDataset))

    @classmethod
    def test_all_sets(cls, with_full_validation=False):
        from util.strings import green_banner, red_banner
        for ds_name in cls.all_sets():
            try:
                ds: CompletionDataset = cls.order(ds_name)
                ldr = ds.loaders(split=[1], s_shuffle=[False], s_nums=[1],
                                 batch_size=1, device='cpu-single', method='rand_f2p',
                                 n_channels=6, s_dynamic=[False])  # , s_transform=[RandomPartNormalDirections()]
                for _ in ldr:
                    # plot_mesh_montage(**prepare_completion_montage(d), strategy='mesh')
                    green_banner(f'SUCCESS - {ds_name}')
                    break
                if with_full_validation:
                    ds.validate_dataset()

            except:
                red_banner(f'FAILED: {ds_name}')
                the_type, the_value, the_traceback = sys.exc_info()
                print(f'Exception: {the_type.__name__} : {the_value}')
                # print(f'Stack Trace:')
                # traceback.print_tb(the_traceback)


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Helper Functions
# ----------------------------------------------------------------------------------------------------------------------

def generate_parts():
    from geom.mesh.io.base import write_ply
    ds: CompletionDataset = DatasetMenu.order('FaustProj')
    for i in range(10):
        v, f = ds.partial_shape_by_hi(('8', '7', i), verts_only=False)
        write_ply(f'{i}.ply', v, f)


def check_correspondence():
    from geom.mesh.vis.base import plot_mesh_montage, uniclr
    ds: ParametricCompletionDataset = DatasetMenu.order('DFaustProj')
    ldr = ds.loaders(method='full', batch_size=9, s_shuffle=True, n_channels=3)
    vi = np.random.choice(range(ds.num_verts()),size=10,replace=False)
    clr = uniclr(N=ds.num_faces(), inds=vi)
    for d in ldr:
        plot_mesh_montage(vb=d['gt'], fb=ds.faces(), clrb=clr,strategy='mesh',lighting=True,show_edges=True)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # check_correspondence()
    # DatasetMenu.test_all_sets(with_full_validation=True)
    #ds: CompletionDataset = DatasetMenu.order('FaustEleProj',data_dir_override='R:\Mano\data\MiniMixamoSkinned13')
    ds: CompletionDataset = DatasetMenu.order('FaustEleProj').validate_dataset()

    #print("f"

