from data.sets import DatasetMenu,SkinningConfiguration,HomogenicMatCfg
from data.base import CompletionDataset,ParametricSkinningDataset
from data.transforms import *
from typing import Union, List, Tuple

# Config:
FULL_SPLIT = [0.8, 0.1, 0.1]
VALD_SPLIT = [0.5, 0.5]
LARGE_DATASET_THERSHOLD = 1000  # Maybe use in_memory from the hit?


# TODO - Insert multi-train sets - currently only one trainset is supported
# TODO - Allow training on scans
# TODO - allowing skipping of first sectors
# ----------------------------------------------------------------------------------------------------------------------
#                                                       Loader Sets
# ----------------------------------------------------------------------------------------------------------------------

#def Ff2p_completion_loaders(hp, train='FaustEleProj',
#                           vald_test='FaustTrainScansEleProj',
#                           test='FaustTrainScansEleProj'):
#def f2p_completion_loaders(hp, train='FaustEleProj',
#                           vald_test='FaustEleProj',
#                           test='FaustEleProj'):
def f2p_completion_loaders(hp, train=None,
                           vald_test=None,
                           test=None):
    """
    :param hp: The hyperparameters object
    :param (str or None) train:
        Train + Vald + Test - The name of the training set you which to train on. Usually we pick DFaust,Mixamo or Amass
        If None is chosen, no training will take place.
    :param (str, Sequence or None) vald_test:
        Vald + Test - The name of the validation and test sets you wish
        to  see generalization results on, besides the test and validation sets of your training dataset. If None is
        chosen, no validation sets are inferred on, and only tests sets stemming from "test" will be tested.
    :param (str, Sequence or None) test:
        Only Test - The name of the test sets you which to infer on, besides all the above sets. If None is chosen,
        no test sets are inferred on besides those supplied by "vald_test"
    :return: The loader sets
    """
    m1 = _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p')
    #m2 = _multiloaders(hp, train_names=None, vald_test_names=vald_test, test_names=test, method='rand_f2p')
    #return _join_loader_sets(m1, m2)
    return _join_loader_sets(m1)


def f2p_completion_augmented_loaders(hp, train='FaustProj',
                                     vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                                     test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[RandomRotate((-180, 180)), RandomTranslate((-5, 5)), RandomMaskFlip(prob=0.5)])


def f2p_completion_random_decimation_loaders(hp, train='DFaustProj',
                                             vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                                             test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[RandomMaskFlip(0.5), RandomMaskDecimation((0, 0.7))])


def f2p_hks_completion_loaders(hp, train='DFaustProj',
                               vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                               test=None, t=(5e-3, 1e1, 10), k=200):
    """
    :param hp:
    :param vald_test:
    :param train:
    :param test:
    :param t: The time parameter from the heat_kernel_signature function in geometry/mesh/op/cpu/descriptor.py
    :param k: The number of eigenvalues to use for computation
    """
    assert hp.in_channels == 3 + t[2] or hp.in_channels == 6 + t[2], "Remember to make room for the HKS descriptor"
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[AppendHKS(t=t, k=k, keys=('tp',))])


def random_normals_loaders(hp, train='DFaustProj',
                           vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                           test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[RandomizeNormalDirections(keys=('gt', 'tp', 'gt_part'))])


def random_normals_loaders_only_part(hp, train='DFaustProj',
                                     vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                                     test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[RandomizeNormalDirections(keys=('gt_part',))])


def approximated_normals_loaders(hp, train='DFaustProj',
                                 vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                                 test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[ReplaceNormalsWithApproximation(keys=('gt', 'tp', 'gt_part'))])


def approximated_normals_loaders_only_part(hp, train='DFaustProj',
                                           vald_test=('FaustProj', 'FaustSemCuts', 'MiniMixamoSkinned13Proj'),
                                           test='FaustTrainScansEleProj'):
    return _multiloaders(hp, train_names=train, vald_test_names=vald_test, test_names=test, method='rand_f2p',
                         transforms=[ReplaceNormalsWithApproximation(keys=('gt_part',))])


def denoising_loaders(hp, train='DFaustProj', noise_range=(0, 0.05)):
    assert hp.in_channels == 3  # Normals are not valid after noise
    assert hp.save_completions == 3  # Better save everything
    assert hp.plotter_class == 'DenoisingPlotter'
    return _loaders(hp, ds_name=train, cls='train', method='rand_f2f',
                    transforms=[RandomGaussianNoise(noise_range, okeys=('gt_noise',))], )


def joint_ablation_loaders(hp):
    return _loaders(hp, ds_name='MiniMixamoSkinned13Proj', cls='train', method='rand_f2j')


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Loader Sets for Skinning
# ----------------------------------------------------------------------------------------------------------------------
# if f2p need_part == true / f2f need_part == false
def skinnng_loaders(hp, ds_name, need_part):
    ds: ParametricSkinningDataset = DatasetMenu.order(ds_name)
    cfg = SkinningConfiguration(gt_local_joint_mats_cfg=HomogenicMatCfg.NONE,
                                gt_world_joint_mats_cfg=HomogenicMatCfg.ORIGINAL,
                                template_local_joint_mats_cfg=HomogenicMatCfg.ONLY_ROTATION,
                                template_world_joint_mats_cfg=HomogenicMatCfg.ONLY_TRANSLATION,
                                need_inverse_bind_mats=HomogenicMatCfg.TRUNC,
                                need_rest_pose_vertices=True,
                                need_skinning_weights=True,
                                need_part=need_part)

    ds.set_skinning_config(cfg)

    return ds.loaders(split=FULL_SPLIT, s_nums=hp.counts,
                      s_transform=[Center()],
                      batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
                      method='skinning', s_shuffle=[True] * 3,
                      s_dynamic=[True, False, False])



# ----------------------------------------------------------------------------------------------------------------------
#                                              Utility Functions
# ----------------------------------------------------------------------------------------------------------------------

def _loaders(hp, ds_name='DFaust', cls='train', transforms: Union[List, Tuple] = tuple(), method='rand_f2p'):
    if ds_name is None:
        return [None] * 3
    assert cls in ('train', 'vald_test', 'test')
    transforms = list(transforms)
    # transforms.append(L2BallNormalize())  # Always normalize to the L2 Ball
    transforms.append(Center())
    # Small Rule Corrections:
    batch_size = 1 if 'scan' in ds_name.lower() else hp.batch_size  # TODO - is this needed? Do we need CPU?
    ds: CompletionDataset = DatasetMenu.order(ds_name)
    if 'window' in ds_name.lower():
        ds.set_window_params(hp.window_size,hp.stride)
    if ds.num_full_shapes() > LARGE_DATASET_THERSHOLD:
        train_dynamic_partition = True  # TODO - Move inside loaders - doesn't belong here.
    else:
        train_dynamic_partition = False
        if method == 'rand_f2p':
            method = 'f2p'
    # assert False, f"hp window size: {hp.window_size}"
    if cls == 'train':
        return ds.loaders(split=FULL_SPLIT, s_nums=hp.counts,
                          s_transform=transforms,
                          batch_size=batch_size, device=hp.dev, n_channels=hp.in_channels,
                          method=method, s_shuffle=[True] * 3,
                          s_dynamic=[train_dynamic_partition, False, False])
    elif cls == 'vald_test':
        return [None] + ds.loaders(split=VALD_SPLIT, s_nums=hp.counts[1:3], s_transform=transforms,
                                   batch_size=batch_size, device=hp.dev, n_channels=hp.in_channels,
                                   method=method, s_shuffle=[True] * 2, s_dynamic=[False, False])
    elif cls == 'test':
        return [None, None, ds.loaders(split=[1], s_nums=hp.counts[2], s_transform=transforms,
                                       batch_size=batch_size, device=hp.dev, n_channels=hp.in_channels,
                                       method=method, s_shuffle=[True], s_dynamic=[False])]
    else:
        raise AssertionError



def _multiloaders(hp, train_names, vald_test_names, test_names, method='rand_f2p',
                  transforms: Union[List, Tuple] = tuple()):
    if not isinstance(train_names, (list, tuple)):
        train_names = [train_names]
    if not isinstance(vald_test_names, (list, tuple)):
        vald_test_names = [vald_test_names]
    if not isinstance(test_names, (list, tuple)):
        test_names = [test_names]

    ldrs = [_loaders(hp, ds_name=name, cls='train', method=method, transforms=transforms) for name in train_names] + \
           [_loaders(hp, ds_name=name, cls='vald_test', method=method, transforms=transforms) for name in
            vald_test_names] + \
           [_loaders(hp, ds_name=name, cls='test', method=method, transforms=transforms) for name in test_names]

    ldrs = list(zip(*ldrs))
    for i, ldr_set in enumerate(ldrs):
        ldr_set = list(filter(None, ldr_set))
        if len(ldr_set) == 0:
            ldr_set = None
        ldrs[i] = ldr_set
    return ldrs


def _join_loader_sets(*args):
    joined = [[], [], []]
    # Each arg is of the form:
    # [X , X , X ] where X is either None or a list of loaders.
    for ldr_set in args:
        for i in range(3):
            if ldr_set[i] is not None:
                joined[i] += ldr_set[i]

    # Transform empty lists to Nones:
    for i in range(3):
        if len(joined[i]) == 0:
            joined[i] = None
    return joined

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
#our new loader

def new_loaders(hp, ds_name='DFaustProj', transforms: Union[List, Tuple] = tuple(), method='rand_f2p',subject_keep = None, pose_keep = None, frame_keep = None):
    transforms = list(transforms)
    # transforms.append(L2BallNormalize())  # Always normalize to the L2 Ball
    transforms.append(Center())
    # Small Rule Corrections:
    batch_size = 1 if 'scan' in ds_name.lower() else hp.batch_size  # TODO - is this needed? Do we need CPU?
    ds: CompletionDataset = DatasetMenu.order(ds_name)
    # print("bdckcksfvkjdvdbkfsbldkf")

    """
    TODO: generalize this for generating"""
    if subject_keep is not None:
        ds._hit = ds._hit.keep_ids_by_depth([subject_keep], 1)
    if pose_keep is not None:
        ds._hit = ds._hit.keep_ids_by_depth([pose_keep], 2)
    if frame_keep is not None:
        ds._hit = ds._hit.keep_ids_by_depth(frame_keep, 3)
    # print(ds._hit)
    if ds.num_full_shapes() > LARGE_DATASET_THERSHOLD:
        train_dynamic_partition = True  # TODO - Move inside loaders - doesn't belong here.
    else:
        train_dynamic_partition = False
        if method == 'rand_f2p':
            method = 'f2p'

    return [None, None, ds.loaders(split=[1], s_nums=hp.counts[2], s_transform=transforms,
                                    batch_size=batch_size, device=hp.dev, n_channels=hp.in_channels,
                                    method=method, s_shuffle=[False], s_dynamic=[False])]


