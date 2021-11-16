import numpy as np
import scipy
from scipy import sparse
from geom.matrix.cpu import row_normalize, last_axis_2norm
from scipy.sparse.csgraph import connected_components
import warnings


# ---------------------------------------------------------------------------------------------------------------------#
#                                                   Vertex properties
# ---------------------------------------------------------------------------------------------------------------------#


def vertex_normals(v, f, normalized=True):
    # NOTE - Vertex normals unreferenced by faces will be zero
    if f is None:
        return estimate_vertex_normals(v)
    else:
        # try:
        fn = face_normals(v, f, normalize=False)
        matrix = vertex_face_adjacency(v.shape[0], f)
        vn = matrix.dot(fn)
        if normalized:
            vn = row_normalize(vn)
        return vn
        # except:
            # return estimate_vertex_normals(v)


def vertex_moments(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)


def vertex_degree(v, f):
    return np.array(np.sum(vertex_adjacency(v, f), 1)).squeeze()


def estimate_vertex_normals(v, k=7, smoothing_iterations=0):
    """
    Estimate normals for a point cloud by locally fitting a plane to a small neighborhood of points
    :param v: ndarray [nv x 3] point cloud
    :param k: The number of nearest neighbors to use
    :param smoothing_iterations: Number of smoothing iterations to apply to the estimated normals
    :return: ndarray [nv x 3] estimate vertex normals
    """
    import point_cloud_utils as pcu
    return pcu.estimate_normals(v, k=k, smoothing_iterations=smoothing_iterations)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Edge Properties
# ---------------------------------------------------------------------------------------------------------------------#

def edges(f, unique=True, return_index=False, return_inverse=False, return_counts=False):
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)
    return np.unique(e, axis=0, return_inverse=return_inverse, return_index=return_index,
                     return_counts=return_counts) if unique else e


def num_edges(v, f):
    return vertex_adjacency(v, f).nnz >> 1  # number_non_zeros(vertex_adj_mat) / 2
    # Equivalent but faster than e1 = len(edges(f))


def unique_edges_by_classification(f, cls='boundary'):
    # Also add in extract_feature_edges
    assert cls in {'boundary', 'singular', '2manifold'}
    e, counts = edges(f, unique=True, return_counts=True)
    if cls == 'boundary':
        return e[counts == 1]
    elif cls == '2manifold':
        return e[counts == 2]
    else:
        return e[counts > 2]


def edge_distances(v, f):
    E = edges(f, unique=True)
    D = last_axis_2norm(v[E[:, 0], :] - v[E[:, 1], :])
    return D, E


# ---------------------------------------------------------------------------------------------------------------------#
#                                                   Face properties
# ---------------------------------------------------------------------------------------------------------------------#

def bary2data(v,f,bary,fi):
    """
    Returns the interpolated data using the given barycentric coordinates over the input geometry

    Args:
        v: [num_vert x 3] vertices array
        f: [num_faces x 3] faces array
        bary: [B x 3] barycentric coordinates
        fi:

    Returns:

    """
    def bary2data(P, T, B, t):
        """


        Parameters
        ----------
        P : Tensor
            the (N,D,) points set tensor
        T : LongTensor
            the (F,M) topology tensor
        B : Tensor
            the (X,F,) barycentric coordinates tensor
        t : LongTensor
            the (X,) faces indices tensor

        Returns
        -------
        Tensor
            the (X,D,) interpolated points set tensor
        """

        return P[T[:, t]] * B.unsqueeze(1)

def face_barycenters(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: face_centers (numpy array or tensors), dim: [n_f x 3]
    """
    return (1 / 3) * (v[f[:, 0], :] + v[f[:, 1], :] + v[f[:, 2], :])


def face_normals(v, f, normalize=True):
    try:
        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]
        fn = np.cross(b - a, c - a)
        if normalize:
            fn = row_normalize(fn)
        return fn
    except:
        from geom.mesh.vis.base import plot_mesh
        pass


def face_areas(v, f):
    return 0.5 * last_axis_2norm(face_normals(v, f, normalize=False))


def face_angles(v, f, return_edge_lengthes=False):
    """"
    Returns in radians the angles of each triangle via the law of cosines
    # TODO - make order clear
    First column:
    Second column:
    Third column:
    """
    v1 = v[f[:, 0], :]
    v2 = v[f[:, 1], :]
    v3 = v[f[:, 2], :]

    L1 = np.linalg.norm(v2 - v3, axis=1)
    L2 = np.linalg.norm(v1 - v3, axis=1)
    L3 = np.linalg.norm(v1 - v2, axis=1)

    cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
    cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
    cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)

    ang = np.arccos(np.column_stack((cos1, cos2, cos3)))  # Cosines of opposite edges for each triangle
    if return_edge_lengthes:
        l = np.column_stack((L1, L2, L3))  # Edges of each triangle
        return ang, l
    return ang


def self_intersecting_faces(v, f):
    import open3d
    # TODO
    open3d.geometry.TriangleMesh.get_self_intersecting_triangles()


# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Global Properties
# ---------------------------------------------------------------------------------------------------------------------#

def is_watertight(f):
    return len(unique_edges_by_classification(f, 'boundary')) == 0


def is_2manifold(f):
    return len(unique_edges_by_classification(f, '2manifold')) == 0


def is_orientable(v, f):
    import open3d
    # TODO
    open3d.geometry.TriangleMesh.is_orientable()


def is_connected(v, f):
    A = vertex_adjacency(v, f)
    n_comps, _ = connected_components(A, directed=False, connection='weak', return_labels=True)
    return n_comps == 1

def n_comps(v, f):
    A = vertex_adjacency(v, f)
    n_comps, _ = connected_components(A, directed=False, connection='weak', return_labels=True)
    return n_comps

def volume(v, f):
    # Note - Only works on watertight meshes
    # Source: https://n-e-r-v-o-u-s.com/blog/?p=4415
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.sum(np.cross(v1, v2) * v3) / 6


def surface_area(v, f):
    return np.sum(face_areas(v, f))


# ---------------------------------------------------------------------------------------------------------------------#
#                                                             Connectivity
# ---------------------------------------------------------------------------------------------------------------------#

def vertex_face_adjacency(nv, f, data=None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A data vector can be passed which is then used instead of booleans
    """
    # Input checks:
    f = np.asanyarray(f)  # Convert to an ndarray or pass if already is one
    nv = int(nv)

    # Computation
    row = f.reshape(-1)  # Flatten indices
    col = np.tile(np.arange(len(f)).reshape((-1, 1)), (1, f.shape[1])).reshape(-1)  # Data for vertices
    shape = (nv, len(f))

    if data is None:
        data = np.ones(len(col), dtype=np.bool)

    # assemble into sparse matrix
    return scipy.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)

    # TODO - Different Implementation - check if faster & equivalent
    # return sparse.coo_matrix((np.ones((3 * npoly,)),  # data
    #                           (np.hstack(self.polys.T),  # row
    #                            np.tile(range(npoly), (1, 3)).squeeze())),  # col
    #                          (npt, npoly)).tocsr()  # size


def vertex_adjacency(v, f, weight=None):
    nv = v.shape[0]
    if weight is None:
        # from util.time import timer
        # Doesn't really need more than nv
        vf = vertex_face_adjacency(nv, f)
        A = vf @ vf.transpose()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # WARNING: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
            A.setdiag(False)
            A.eliminate_zeros()
        # with timer():
        #     vf = vertex_face_adjacency(nv, f)
        #     A = vf @ vf.transpose()
        #     A = A.tolil()
        #     A.setdiag(False)
        #     # WARNING: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
        # with timer():
        #     e = edges(f)
        #     ii = np.concatenate((e[:, 0], e[:, 1]))
        #     jj = np.concatenate((e[:, 1], e[:, 0]))
        #     vals = np.ones_like(ii)
        #     A2 = sparse.csr_matrix((vals, (ii, jj)), shape=(nv, nv), dtype='float64')
        # assert np.array_equal(A,A2) # Fails on degenerate faces
        # TODO - check this one out:
        #     def adj(self):
        #         """Sparse vertex adjacency matrix.
        #         """
        #         npt = len(self.pts)
        #         npoly = len(self.polys)
        #         adj1 = sparse.coo_matrix((np.ones((npoly,)),
        #                                   (self.polys[:,0], self.polys[:,1])), (npt,npt))
        #         adj2 = sparse.coo_matrix((np.ones((npoly,)),
        #                                   (self.polys[:,0], self.polys[:,2])), (npt,npt))
        #         adj3 = sparse.coo_matrix((np.ones((npoly,)),
        #                                   (self.polys[:,1], self.polys[:,2])), (npt,npt))
        #         alladj = (adj1 + adj2 + adj3).tocsr()
        #         return alladj + alladj.T
    elif weight == 'euclidean':
        # Build the Euclidean Adj matrix:
        de, e = edge_distances(v, f)
        ii = np.concatenate((e[:, 0], e[:, 1]))
        jj = np.concatenate((e[:, 1], e[:, 0]))
        vals = np.concatenate((de, de))
        A = sparse.csr_matrix((vals, (ii, jj)), shape=(nv, nv), dtype='float64')
    else:
        raise NotImplementedError(f'Unknown weight {weight}')
    return A


def face_adjacency(nv, f):
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Vertex Masking
# ---------------------------------------------------------------------------------------------------------------------#


def padded_part_by_mask(vi, v):
    # Pad the mask to length:
    needed_padding_len = v.shape[0] - len(vi)  # Truncates ALL input channels
    mask_vi_padded = np.append(vi, np.random.choice(vi, needed_padding_len, replace=True))  # Copies
    return v[mask_vi_padded, :]

def flip_vertex_mask(nv, vi):
    indicator = vertex_mask_indicator(nv, vi)
    return np.where(indicator == 0)[0]


def vertex_mask_indicator(nv, vi, val=1):
    """
    :param nv: The number of vertices
    :param vi: The vertex ids which we want to indicate
    :param val: The value at vi
    :return: ndarray(nv) = val at vi or 0 otherwise
    """
    indicator = np.zeros((nv,), dtype=bool)
    indicator[vi] = val
    return indicator


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Normalization
# ---------------------------------------------------------------------------------------------------------------------#
def box_center(v):
    """
    :param v: [nv x 3] ndarry : A point cloud in 3D
    :return: A copy of v, whose center is now the Linf box center
    # To normalize by Linf, simply divide by np.max(np.abs(v)) <- The maximal coordinate
    """
    bbox_x = [np.min(v[:, 0]), np.max(v[:, 0])]
    bbox_y = [np.min(v[:, 1]), np.max(v[:, 1])]
    bbox_z = [np.min(v[:, 2]), np.max(v[:, 2])]
    center = 0.5 * np.array([bbox_x[0] + bbox_x[1], bbox_y[0] + bbox_y[1], bbox_z[0] + bbox_z[1]])
    return v - center[np.newaxis, :]


def mean_center(v):
    return v - v.mean(axis=0, keepdims=True)


def normalize_to_l2_ball(v, r=1):
    """
    :param v: [nv x 3] ndarry : A point cloud in 3D
    :param r: The radius of the L2 ball required
    :return: A copy of v, normalized to an L2 ball of radius r
    """
    v = mean_center(v)  # Remove L2-Center
    return v / (r * np.sqrt(np.max(np.sum(v ** 2, 1))))  # Divide by 2-Norm


def normalize_channels(v):
    """
    :param v: [nv x 3] ndarry : A point cloud in 3D
    """
    return v / np.max(v, 0)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                     TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _normal_estimate_grid_search():
    from geom.mesh.io.base import read_mesh, numpy2open3d_cloud
    from geom.mesh.vis.base import plot_mesh_montage
    from geom.mesh.op.cpu.remesh import clean_mesh
    from geom.matrix.cpu import total_l2_err
    from cfg import TEST_MESH_HUMAN_PATH, TEST_SCAN_PATH, TEST_MESH_HAND_PATH
    from util.time import timer
    from util.strings import banner
    import open3d as o3d
    import trimesh
    run_alg_control = [True,True] # Open3D, PntCloudUtil
    v1, f1 = read_mesh(TEST_SCAN_PATH)
    v,f = clean_mesh(v1,f1)
    # v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    true_nn = vertex_normals(v, f)
    P = numpy2open3d_cloud(v)

    if run_alg_control[0]:
        banner('Open 3D')
        optimal_signed_err, optimal_signed_params = np.inf, []
        optimal_unsigned_err, optimal_unsigned_params = np.inf, []
        for radii in [0.05, 0.1, 0.2, 0.3]:
            for nn in range(1, 50):
                with timer():
                    P.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radii, max_nn=nn))
                    nrml = np.asarray(P.normals)  # Already normalized
                    # plot_mesh_montage(vb=[v], fb=[f], nb=[nrml], strategy='mesh')

                err_signed = total_l2_err(true_nn, nrml)
                if err_signed < optimal_signed_err:
                    optimal_signed_err = min(err_signed, optimal_signed_err)
                    optimal_signed_params = [radii, nn]
                # print(f'Method: O3D radii={radii}, nn={nn} : Signed Error: {err_signed}')
                err_unsigned = total_l2_err(np.abs(true_nn), np.abs(nrml))
                if err_unsigned < optimal_unsigned_err:
                    optimal_unsigned_err = min(err_unsigned, optimal_unsigned_err)
                    optimal_unsigned_params = [radii, nn]
                # print(f'Method: O3D radii={radii}, nn={nn} : Unsigned Error: {err_unsigned}')

        print(f'Optimal Unsigned Params: {optimal_unsigned_params} with {optimal_unsigned_err}')
        print(f'Optimal Signed Params: {optimal_signed_params} with {optimal_signed_err}')

    # Notes:
    # [*] Faster than the 2nd method on all cases, better unsigned error, but not as good at orientation
    # Human:
    # [1] Best unsigned parameters: [radii,nn] = [0.05, 13] err~=271.211
    # [2] Best signed parameters: [radii,nn] = [0.05, 3], err~=12913
    # Hand:
    # [1] Best unsigned parameters: [radii,nn] = [0.05, 7] err~=31.504
    # [2] Best signed parameters: [radii,nn] = [0.2, 42], err~=8317.31
    # Scan (not that vertex normals are that good...):
    # [1] Best unsigned parameters: [radii,nn] = [0.05, 7]  err~=510.9826
    # [2] Best signed parameters: [radii,nn] = [0.05, 3], err~=381234.081

    if run_alg_control[1]:
        banner('Pnt Cloud Utils')
        optimal_signed_err, optimal_signed_params = np.inf, []
        optimal_unsigned_err, optimal_unsigned_params = np.inf, []
        for smoothing in [0, 1, 2, 3, 5, 10, 20]:
            for nn in range(1, 50):
                with timer():
                    nrml = estimate_vertex_normals(v, k=nn, smoothing_iterations=smoothing)
                    plot_mesh_montage(vb=[v], fb=[f], nb=[nrml],strategy='mesh')
                err_signed = total_l2_err(true_nn, nrml)
                if err_signed < optimal_signed_err:
                    optimal_signed_err = min(err_signed, optimal_signed_err)
                    optimal_signed_params = [smoothing, nn]
                # print(f'Method: PntUtil smoothing={smoothing}, nn={nn} : Signed Error: {err_signed}')
                err_unsigned = total_l2_err(np.abs(true_nn), np.abs(nrml))
                if err_unsigned < optimal_unsigned_err:
                    optimal_unsigned_err = min(err_unsigned, optimal_unsigned_err)
                    optimal_unsigned_params = [smoothing, nn]
                # print(f'Method: PntUtil smoothing={smoothing}, nn={nn} : Unsigned Error: {err_unsigned}')

        print(f'Optimal Unsigned Params: {optimal_unsigned_params} with {optimal_unsigned_err}')
        print(f'Optimal Signed Params: {optimal_signed_params} with {optimal_signed_err}')

    # Notes:
    # [*] Slower than open3d on all cases, and less accurate too. Better in orienting normals
    # Human:
    # [1] Best unsigned parameters: [smoothing,nn] = [0,9] err~=273
    # [2] Best signed parameters: [smoothing,nn] = [0,8], err~=931
    # Hand:
    # [1] Best unsigned parameters: [smoothing,nn] = [ , ] err~=
    # [2] Best signed parameters: [smoothing,nn] = [ , ], err~=
    # Scan (not that vertex normals are that good...):
    # [1] Best unsigned parameters: [smoothing,nn] = [ , ] err~=
    # [2] Best signed parameters: [smoothing,nn] = [ , ], err~=
    #Optimal Unsigned Params: [0, 7] with 510.98263055124494
    #Optimal Signed Params: [0, 7] with 740.6470458657952
    #
    pass


if __name__ == '__main__':
    _normal_estimate_grid_search()
