from typing import Union

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import torch


def unique_rows(A, count_permutations_as_non_unique=False):
    a = np.sort(A, axis=1, kind='stable') if count_permutations_as_non_unique else A
    vals, indices, inv_indices, counts = np.unique(a, axis=0, return_index=True, return_inverse=True,
                                                   return_counts=True)
    # TODO - note, vals are not in their original ordering!
    if count_permutations_as_non_unique:
        vals = A[indices, :]
    return vals, indices, inv_indices, counts


def torch2numpy(*args):
    out = []
    for arg in args:
        arg = arg.numpy() if torch.is_tensor(arg) else arg
        out.append(arg)
    if len(args) == 1:
        return out[0]
    else:
        return tuple(out)


def l2_normalize(vectors,
                 return_valid=False,
                 threshold=1e-13):
    """
    Unitize a vector or an array or row-vectors.
    Parameters
    ------------
    vectors : (n,m) or (j) float
       Vector or vectors to be unitized
    return_valid :  bool
       If set, will return mask of nonzero vectors
    threshold : float
       Cutoff for a value to be considered zero.
    Returns
    ---------
    unit :  (n,m) or (j) float
       Input vectors but unitized
    valid : (n,) bool or bool
        Mask of nonzero vectors returned if `check_valid`
    """
    # make sure we have a numpy array
    vectors = np.asanyarray(vectors).squeeze()

    if len(vectors.shape) == 2:
        # for (m, d) arrays take the per-row unit vector
        # using sqrt and avoiding exponents is slightly faster
        # also dot with ones is faser than .sum(axis=1)
        norm = np.sqrt(np.dot(vectors * vectors,
                              [1.0] * vectors.shape[1]))
        # non-zero norms
        valid = norm > threshold
        # in-place reciprocal of nonzero norms
        norm[valid] **= -1
        # multiply by reciprocal of norm
        unit = vectors * norm.reshape((-1, 1))

    elif len(vectors.shape) == 1:
        # treat 1D arrays as a single vector
        norm = np.sqrt(np.dot(vectors, vectors))
        valid = norm > threshold
        if valid:
            unit = vectors / norm
        else:
            unit = vectors.copy()
    else:
        raise ValueError('vectors must be (n, ) or (n, d)!')

    if return_valid:
        return unit, valid
    return unit


def last_axis_norm(mat):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(np.power(mat, 2), axis=-1))


def edges(v, f, unique=True, return_index=False, return_inverse=False, return_counts=False):
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)
    return np.unique(e, axis=0, return_inverse=return_inverse, return_index=return_index,
                     return_counts=return_counts) if unique else e


def box_bounds(v, _=None):
    return np.array([v.min(axis=0), v.max(axis=0)])


def box_lengths(v, _=None):
    """
    The length, width, and height of the axis aligned
    bounding box of the mesh.
    Returns
    -----------
    extents : (3, ) float
      Array containing axis aligned [length, width, height]
    """
    return box_bounds(v).ptp(axis=0)


def max_diagonal_length(v, _=None):
    """
    A metric for the overall scale of the mesh, the length of the
    diagonal of the axis aligned bounding box of the mesh.
    Returns
    ----------
    scale : float
      The length of the meshes AABB diagonal
    """
    scale = float((box_lengths(v) ** 2).sum() ** .5)
    return scale


def point_cloud_radius(v, _=None):
    return max_diagonal_length(v) / 2


def edge_centers(v, f):
    e = edges(v, f, unique=True)
    return 0.5 * (v[e[:, 0], :] + v[e[:, 1], :])


def num_edges(v, f):
    return len(edges(v, f))


def face_barycenters(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: face_centers (numpy array or tensors), dim: [n_f x 3]
    """
    return (1 / 3) * (v[f[:, 0], :] + v[f[:, 1], :] + v[f[:, 2], :])


def is_sequence(obj):
    """
    Check if an object is a sequence or not.
    Parameters
    -------------
    obj : object
      Any object type to be checked
    Returns
    -------------
    is_sequence : bool
        True if object is sequence
    """
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))

    # check to make sure it is not a set, string, or dictionary
    seq = seq and all(not isinstance(obj, i) for i in (dict,
                                                       set,
                                                       str))

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()

    return seq


def hybrid_kwarg_index(i, expected_first_dim, **kwargs):
    for k, v in kwargs.items():
        if is_sequence(v):
            if len(v) == expected_first_dim:
                kwargs[k] = v[i]  # TODO - color falls here if v.shape[0] == n_meshes
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == expected_first_dim \
                    and k not in ['f', 'v', 'n']:
                # TODO - this is dangerous as well - seeing expected_first_dim could n_f or n_v for example
                # support for color tensors
                kwargs[k] = v[:, i]
    return kwargs


def numpy2pyvista(v, f=None):
    if f is None:
        return pv.PolyData(v)
    else:
        return pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))


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
    elif isinstance(weight, str):
        if weight == 'barycenter':
            weight = np.repeat(face_areas(v, f), 3)
        else:
            raise NotImplementedError(f'Unknown weight name: {weight}')
    # Otherwise, we suppose that 'weight' is a vector of the needed size.

    vf = sp.csr_matrix((weight, (row, col)), shape=(v.shape[0], len(f)), dtype=weight.dtype)
    return vf


def vertex_normals(v, f, normalized=True, weight_cls=None, return_weight=False):
    # NOTE - Vertex normals unreferenced by faces will be zero
    vf = vertex_face_adjacency(v, f, weight_cls)  # TODO switch to bincount
    vn = vf @ face_normals(v, f, normalized=True)
    if normalized:
        vn = l2_normalize(vn)
    if return_weight:
        return vn, vf.data.copy()
    return vn


def face_normals(v, f, normalized=True, return_valid=False):
    a = v[f[:, 0], :]  # [nf, 3] - Will hold vertices
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)
    if normalized:
        fn = l2_normalize(vectors=fn, return_valid=return_valid)
    return fn

def center_by_l2_mean(v):
    """
    This achieves translation invariance for the mesh
    """
    return v - np.mean(v, axis=0)  # TODO - keepdims?


def face_areas(v, f):
    if v.shape[1] == 3:
        return 0.5 * last_axis_norm(face_normals(v, f, normalized=False))
    else:
        # Handle 2D case triangle case
        assert v.shape[1] == 2
        return 0.5 * np.abs(face_normals(v, f, normalized=False))
