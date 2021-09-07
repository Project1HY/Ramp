from geom.matrix.cpu import last_axis_2norm, last_axis_normalize
from geom.mesh.op.cpu.base import vertex_adjacency, face_areas, face_angles, face_normals
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Collectors
# ---------------------------------------------------------------------------------------------------------------------#

def spectral_weights(v, f, cls='cotangent'):
    if cls in {'uniform', 'combinatorial', 'graph'}:
        return vertex_adjacency(v, f)
    elif cls in {'cotangent', 'conformal', 'meyer'}:
        return cotangent_weights(v, f)
    else:
        raise NotImplementedError(cls)


def vertex_mass_matrix(v, f, cls):
    """
    Computes the mass matrix of the mesh (v,f)
    Types:
    'full':         full mass matrix for p.w. linear finite element method
                    Reference: Finite Elements for Analysis and Design,  J. E. Akin
                    Diagonal = 0.5 * barycenter area
                    Off Diagonal = 0.25 * barycenter area
    'barycenter':  sparse diagonal lumped mass matrix obtained by summing 1/3 of each face
    'voronoi':     sparse diagonal lumped mass matrix obtained by computing voronoi areas,
                   without obtuse triangle handling
    'mixed':       sparse diagonal lumped mass matrix obtained by the mixed formulation in
                   Discrete Differential-Geometry Operators for Triangulated 2-Manifolds, Meyer et al
    None:          Returns None

    :param v:
    :param f:
    :param cls: One of 'full','barycenter','voronoi','mixed'
    """
    assert cls in {'full', 'barycenter', 'voronoi', 'mixed', None}
    if cls is None:
        return None
    return globals()[f'{cls}_vertex_mass_matrix'](v, f)  # Builds upon the naming convention


def laplacian(v, f, cls='cotangent'):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :param cls: The Laplacian type
    :return: The laplacian specified by type
    To compute the laplacian coordinates, use: L * v
    """

    nv = v.shape[0]
    W = spectral_weights(v, f, cls)
    D = sparse.spdiags(np.array(np.sum(W, 1)).squeeze(), 0, nv, nv)
    L = D - W

    if cls == 'meyer':  # TODO - find a better name / Remove this
        L *= -0.5  # Equivalent to: 0.5*(W -D)
    return L


def laplacian_spectrum(v, f, k, laplacian_cls='cotangent', mass_cls='barycenter'):
    L = laplacian(v, f, laplacian_cls)
    # assert is_symmetric(L.todense()) # Expensive check
    M = vertex_mass_matrix(v, f, mass_cls)
    eig_val, eig_vec = eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)
    return eig_val, eig_vec, L, M


def gradient(v, f):
    nv, nf = v.shape[0], f.shape[0]
    # Edge Vectors:
    e21 = v[f[:, 2]] - v[f[:, 1]]
    e02 = v[f[:, 0]] - v[f[:, 2]]
    e10 = v[f[:, 1]] - v[f[:, 0]]
    # Face Areas:
    fa = .5 * last_axis_2norm(np.cross(e21, e02))
    # Edge Normals:
    face_normals = last_axis_normalize(np.cross(last_axis_normalize(e10), last_axis_normalize(e21)))
    e1_normal = np.cross(face_normals, e21)
    e2_normal = np.cross(face_normals, e02)
    e3_normal = np.cross(face_normals, e10)

    i = np.repeat(np.arange(nf), 3)
    ii = np.concatenate((i, i + nf, i + 2 * nf))
    f_flat = f.flatten()
    jj = np.concatenate((f_flat, f_flat, f_flat))

    # Can be simplified - the Matlab flatten is != to numpy.flatten() - Row major vs col major
    vals = np.row_stack((e1_normal.T.flatten(), e2_normal.T.flatten(), e3_normal.T.flatten())).T.flatten()
    G = sparse.csr_matrix((vals, (ii, jj)), [3 * nf, nv])
    inv_face_area = 0.5 * np.tile(1 / fa, (1, 3))
    inv_face_mat = sparse.spdiags(inv_face_area, 0, 3 * nf, 3 * nf)
    G = inv_face_mat @ G
    return G


def divergence(v, f):
    nv, nf = v.shape[0], f.shape[0]
    G = gradient(v, f)
    GvInv = sparse.spdiags(1 / barycenter_vertex_areas(v, f), 0, nv, nv)  # Inverse barycenter matrix
    Gf = sparse.spdiags(np.tile(face_areas(v, f), (3,)), 0, 3 * nf, 3 * nf)
    D = - GvInv @ G.T @ Gf

    L = laplacian(v, f, 'meyer')
    L2 = barycenter_vertex_mass_matrix(v, f) * D * G  # Odd -
    print(np.sum(np.abs(L - L2)))  # 5.79 e-12

    return D, G


def tangent_projection(v, f, vector_field, normalize=True):
    # TODO - Is this the right place for this function?
    fn = face_normals(v, f, normalize=True)
    proj_vector_field = vector_field - np.sum(vector_field * fn, -1, keepdims=True) * fn

    norm_proj = last_axis_2norm(proj_vector_field)  # Scalar function for facets
    if normalize:
        proj_vector_field = last_axis_normalize(proj_vector_field)
    return proj_vector_field, norm_proj


# ---------------------------------------------------------------------------------------------------------------------#
#                                             Implementations - Weights
# ---------------------------------------------------------------------------------------------------------------------#

def cotangent_weights(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: The cotangent/conformal weight matrix
    [w_ij=  cot(alpha_ij)+cot(beta_ij) where alpha_ij and beta_ij are the adjacent angle to edge (i,j)]
    To compute the laplacian coordinates, use: L * v
    """
    # Compute triangle edge vectors
    l_01 = v[f[:, 0], :] - v[f[:, 1], :]
    l_02 = v[f[:, 0], :] - v[f[:, 2], :]
    l_12 = v[f[:, 1], :] - v[f[:, 2], :]
    # Compute triangle cotangent angles (dot product / mag cross product), seeing A.dot(B) = |A||B|cos(alpha) and
    # |A.cross(B)| = |A||B|sin(alpha)
    cot0 = (l_01 * l_02).sum(axis=1) / last_axis_2norm(np.cross(l_01, l_02))
    cot1 = (-l_12 * l_01).sum(axis=1) / last_axis_2norm(np.cross(-l_12, l_01))
    cot2 = (l_02 * l_12).sum(axis=1) / last_axis_2norm(np.cross(l_02, l_12))

    # TODO - Do we need to sanitize?
    #         cots[np.isinf(cots)] = 0
    #         cots[np.isnan(cots)] = 0
    #
    nv = v.shape[0]
    cot = np.concatenate((cot0, cot0, cot1, cot1, cot2, cot2))
    ii = np.concatenate([f[:, 1], f[:, 2], f[:, 2], f[:, 0], f[:, 0], f[:, 1]])
    jj = np.concatenate([f[:, 2], f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0]])
    return sparse.csr_matrix((cot, (ii, jj)), shape=(nv, nv), dtype='float64')


# ---------------------------------------------------------------------------------------------------------------------#
#                                             Implementations - Weights
# ---------------------------------------------------------------------------------------------------------------------#

def barycenter_vertex_areas(v, f):
    nv = v.shape[0]
    fa = (1 / 3) * face_areas(v, f)
    return np.bincount(f[:, 0], fa, nv) + np.bincount(f[:, 1], fa, nv) + np.bincount(f[:, 2], fa, nv)


def voronoi_vertex_areas(v, f, handle_obtuse=True):
    """
    Returns the Voronoi vertex areas as defined in:
    Discrete Differential-Geometry Operators for Triangulated 2-Manifolds, Meyer et al
    Note: This is a more accurate discretization technique, but at 5-6 times the computational cost of the
    barycenter_vertex_areas
    """
    nv = v.shape[0]
    ang, edge_len = face_angles(v, f, return_edge_lengthes=True)
    cot = np.cos(ang) / np.sin(ang)

    # Voronoi Areas for each angle in the face array, as supplied by equation (7), page 9
    fva = (1 / 8) * (edge_len[:, [1, 2, 0]] ** 2 * cot[:, [1, 2, 0]] + edge_len[:, [2, 0, 1]] ** 2 * cot[:, [2, 0, 1]])

    if handle_obtuse:
        # Barycenter Areas for faces
        fba = face_areas(v, f)
        # Fix obtuse triangles - TODO - this may be vectorized a bit better.
        locs = cot[:, 0] < 0
        fva[locs, 0], fva[locs, 1], fva[locs, 2] = fba[locs] / 2, fba[locs] / 4, fba[locs] / 4
        locs = cot[:, 1] < 0
        fva[locs, 0], fva[locs, 1], fva[locs, 2] = fba[locs] / 4, fba[locs] / 2, fba[locs] / 4
        locs = cot[:, 2] < 0
        fva[locs, 0], fva[locs, 1], fva[locs, 2] = fba[locs] / 4, fba[locs] / 4, fba[locs] / 2

    # Sum contribution for each vertex
    return np.bincount(f[:, 0], fva[:, 0], nv) + np.bincount(f[:, 1], fva[:, 1], nv) + np.bincount(f[:, 2], fva[:, 2],
                                                                                                   nv)


def voronoi_vertex_mass_matrix(v, f):
    return sparse.spdiags(voronoi_vertex_areas(v, f, handle_obtuse=False), 0, v.shape[0], v.shape[0])


def mixed_vertex_mass_matrix(v, f):
    return sparse.spdiags(voronoi_vertex_areas(v, f, handle_obtuse=True), 0, v.shape[0], v.shape[0])


def barycenter_vertex_mass_matrix(v, f):
    return sparse.spdiags(barycenter_vertex_areas(v, f), 0, v.shape[0], v.shape[0])


def full_vertex_mass_matrix(v, f):
    # Generate values
    fa = face_areas(v, f)
    fa3 = np.concatenate((fa, fa, fa))
    Mij = (1 / 12) * fa3
    Mii = (1 / 6) * fa3
    Mji = Mij
    vals = np.concatenate((Mij, Mji, Mii))

    # Generate indices:
    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]
    i = np.concatenate((f1, f2, f3))
    j = np.concatenate((f2, f3, f1))
    ii = np.concatenate((i, j, i))
    jj = np.concatenate((j, i, i))

    # Collect
    nv = v.shape[0]
    return sparse.csr_matrix((vals, (ii, jj)), [nv, nv])  # Sparse Cotangent Weight Matrix


# ---------------------------------------------------------------------------------------------------------------------#
#                                                         TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _tangent_projection_tester(normalize=False):
    from geom.mesh.synth.mesh_zoo_3d import icosphere
    from geom.mesh.vis.base import plot_projected_vectorfield
    v, f = icosphere(refinement_order=5)
    nv, nf = v.shape[0], f.shape[0]

    vfx = np.tile([1, 0, 0], (nf, 1))
    vfy = np.tile([0, 1, 0], (nf, 1))
    vfz = np.tile([0, 0, 1], (nf, 1))

    plot_projected_vectorfield(v, f, vfx, normalize, normal_scale=1, label='X', grid_on=True, clr_map='rainbow')
    plot_projected_vectorfield(v, f, vfy, normalize, normal_scale=1, label='Y', grid_on=True)
    plot_projected_vectorfield(v, f, vfz, normalize, normal_scale=1, label='Z', grid_on=True)


def _discrete_ops_tester():
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import plot_mesh, add_mesh, add_spheres, plotter, add_vectorfield
    from geom.mesh.op.cpu.dist import vertex_centroid_distance
    from geom.mesh.vis.base import plot_mesh, add_mesh, add_spheres, plotter
    from cfg import TEST_MESH_HAND_PATH

    v, f = read_mesh(TEST_MESH_HAND_PATH)
    func, c = vertex_centroid_distance(v)
    p = plotter()
    D, G = divergence(v, f)
    add_mesh(p, v, f, clr=func, strategy='mesh', lighting=True, smooth_shade_on=False)
    add_spheres(p, v=c, sphere_size=2, sphere_clr='w')
    gf = G @ func
    add_vectorfield(p, v, f, gf)
    p.show()

    p = plotter()
    add_mesh(p, v, f, clr=D @ G @ func, strategy='mesh', lighting=True, smooth_shade_on=False)
    add_spheres(p, v=c, sphere_size=2, sphere_clr='w')
    add_vectorfield(p, v, f, G @ func)
    p.show()


if __name__ == '__main__':
    _discrete_ops_tester()
