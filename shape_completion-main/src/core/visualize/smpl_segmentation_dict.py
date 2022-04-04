import copy
from typing import Union
from save_load_obj import load_obj
from human_mesh_utils import get_defult_dir_values
from get_human_template import get_human_model
import scipy.sparse as sp

import numpy as np

import sys
sys.path.insert(0, '..')
from geom.mesh.vis.base import plot_mesh_montage
from geom.mesh.vis.base import plot_mesh

# ---------------------------------------------------------------------------------------------------------------------#
#                                           indicators
# ---------------------------------------------------------------------------------------------------------------------#

def indicator_to_index(indicator, flip=False):
    if flip:
        return np.where(indicator == 0)[0]  # Return index of all 0 elements
    else:
        return np.where(indicator != 0)[0]  # Return index of all non-zero elements


def flip_indicator(indicator):
    return ~indicator


def flip_index(n, index):
    return indicator_to_index(index_to_indicator(n, index, flip=True))


def index_to_indicator(n, index, flip=False, val: Union[bool, float, int] = True):
    assert val != 0
    if flip:
        indicator = np.full(n, val, dtype=type(val))
        indicator[index] = 0
    else:
        indicator = np.zeros(n, dtype=type(val))
        indicator[index] = val
    return indicator


def _indicator_test():
    print(index_to_indicator(n=10, index=[1, 2, 3]))
    print(index_to_indicator(n=10, index=[1, 2, 2]))
    print(index_to_indicator(n=10, index=[1, 2, 2], val=5))
    print(index_to_indicator(n=10, index=[1, 2, 3], flip=True))
    print(index_to_indicator(n=10, index=[1, 2, 2], flip=True))
    print(index_to_indicator(n=10, index=[1, 2, 2], val=5, flip=True))

    print('sep')

    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 3])))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2])))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], val=5)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 3], flip=True)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], flip=True)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], val=5, flip=True)))


# ---------------------------------------------------------------------------------------------------------------------#
#                                   segmentation_dict
# ---------------------------------------------------------------------------------------------------------------------#

def semantic_segmentation_to_color_vector(seg):
    n_vertices = max(v.max() for v in seg.values()) + 1
    color_seg = np.zeros((n_vertices,))
    for i, v in enumerate(seg.values(), 1):  # Allow for partial segmentations, with the val 0 as a collector
        color_seg[v] = i
    return color_seg


def segmentation_quantizer(seg, pairs):
    new_seg = copy.deepcopy(seg)
    for i, (tgt_name, src_names) in enumerate(pairs.items()):

        src_names = list(src_names)
        if len(src_names) == 1:  # Option 1: WildCard Prefix
            prefix = src_names[0].split('*')[0]
            src_names = [s for s in seg.keys() if s.startswith(prefix)]

        for k in src_names:  # Remove old names
            del new_seg[k]
        new_seg[tgt_name] = np.concatenate([seg[name] for name in src_names])
    return new_seg

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def smpl_segmentation_52_joints(f_name:str=None):
    if f_name==None:
        f_name=get_defult_dir_values()['smpl_segmentation_file']
    #from util.fileio import pickle_load
    return load_obj(f_name)


def simplified_smpl_segmentation_15_joints(f_name:str=None):
    pairs = {
        'Head': ('Head', 'Neck'),
        'Torso': ('Spine*',),
        'RightHand': ('RightHand*',),
        'LeftHand': ('LeftHand*',),
        'RightArm': ('RightShoulder', 'RightArm'),
        'LeftArm': ('LeftShoulder', 'LeftArm'),
        'RightFoot': ('RightFoot', 'RightToeBase'),
        'LeftFoot': ('LeftFoot', 'LeftToeBase')
    }
    return segmentation_quantizer(smpl_segmentation_52_joints(f_name=f_name), pairs)


def simplified_smpl_segmentation_6_joints(f_name:str=None):
    smpl_seg = simplified_smpl_segmentation_15_joints(f_name=f_name)
    pairs = {
        'Torso': ('Torso', 'Hips'),
        'LeftArm': ('LeftForeArm', 'LeftHand', 'LeftArm'),
        'RightArm': ('RightForeArm', 'RightHand', 'RightArm'),
        'LeftLeg': ('LeftFoot', 'LeftLeg', 'LeftUpLeg'),
        'RightLeg': ('RightFoot', 'RightLeg', 'RightUpLeg'),
    }
    return segmentation_quantizer(smpl_seg, pairs)

def highlight_segmentation(v, seg, f=None, split_screen=False, **plt_args):
    n_v, n_parts = len(v), len(seg)
    if split_screen:
        titles = list(seg.keys())
        colors = [index_to_indicator(n_v, v) for v in seg.values()]
        plot_mesh_montage(vs=[v] * n_parts, fs=f, colors=colors, titles=titles, **plt_args)
    else:
        # TODO - Add in legend
        plot_mesh(v,f, clr=semantic_segmentation_to_color_vector(seg),
                  clr_map='jet', **plt_args)

def _segmentation_tester():
    from geom.tool.vis import highlight_segmentation
    from cfg import Assets
    v, f = Assets.MAN.load()
    seg = simplified_smpl_segmentation_6_joints()
    print(len(seg))
    highlight_segmentation(v=v, f=f, seg=seg, split_screen=False, lighting=True, point_size=3)
    pass

def _segmentation_tester2():
    from error_metrics_test import get_mock_ds_and_ldr
    ds,ldr=get_mock_ds_and_ldr()
    for batch in ldr:
        v=batch['gt'][0]
        break
    f = ds._f
    seg = simplified_smpl_segmentation_6_joints()
    print(len(seg))
    highlight_segmentation(v=v, f=f, seg=seg, split_screen=False, lighting=True, point_size=3)
    pass


def flatten(t):
    #some code duplication
    return [item for sublist in t for item in sublist]

def get_valid_n_joints()->list:
    valid_n_joints=[6,15,52]
    return valid_n_joints


def get_segmentation(n_joints:int=6,include_full_segmentation:bool=False,seg_f_name:str=None,organs:list = None)->dict:
    assert(n_joints in get_valid_n_joints())
    seg=None
    model=get_human_model(gender='neutral')
    if n_joints==6:
        vertex_seg=simplified_smpl_segmentation_6_joints(f_name=seg_f_name)
    elif n_joints==15:
        vertex_seg=simplified_smpl_segmentation_15_joints(f_name=seg_f_name)
    else: #n_joints==52
        vertex_seg=smpl_segmentation_52_joints(f_name=seg_f_name)
    v,f=model.vertices,model.faces
    face_seg= {k : vertex_adjacent_faces(v,f,vi=vertex_index_list) for k,vertex_index_list in vertex_seg.items()}
    #repair this
    face_seg= {k:sorted(list(set(flatten(face_seg_list_of_lists.tolist())))) for k,face_seg_list_of_lists in face_seg.items()}

    res=dict()
    if include_full_segmentation and organs == None:
        res['Full']={'faces':list(range(len(f))),'vertices':list(range(len(v)))}
    if organs == None:
        organs = face_seg.keys()
    for k in organs:
        res[k]={'faces':face_seg[k],'vertices':vertex_seg[k]}

    return res



# ---------------------------------------------------------------------------------------------------------------------#
#                       Utils
# ---------------------------------------------------------------------------------------------------------------------#
def vertex_adjacent_faces(v, f, vi=None, clumped=False):
    if vi is None:
        vi = np.arange(len(v))
    else:
        vi = np.asanyarray(vi)
    A = vertex_face_adjacency(v, f)
    fi = A.tolil().rows[vi]
    if clumped and len(fi):
        fi = np.sort(np.unique(np.concatenate(fi).ravel())).astype(np.int)
    return fi

def vertex_face_adjacency(v, f, weight: Union[str, None, np.ndarray] = None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A weight vector can be passed which is then used instead of booleans - for example, the face areas
    weight vector format: [face0,face0,face0,face1,face1,face1,...]
    """
    row = f.ravel()  # Flatten indices
    col = np.repeat(np.arange(len(f)), 3)  # Data for vertices

    if weight is None:
        weight = np.ones(len(col), dtype=np.bool)
    # Otherwise, we suppose that 'weight' is a vector of the needed size.

    vf = sp.csr_matrix((weight, (row, col)), shape=(v.shape[0], len(f)), dtype=weight.dtype)
    return vf

def main():
    _segmentation_tester2()
    print('a')

if __name__=="__main__":
    main()
