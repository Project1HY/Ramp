import numpy as np
from geom.mesh.op.cpu.base import face_areas

# TODO - for more sampling techniques see https://github.com/fwilliams/point-cloud-utils
# Poisson-Disk-Sampling of a mesh based on "Parallel Poisson Disk Sampling with Spectrum Analysis on Surface".
# Sampling a mesh with Lloyd's algorithm
# Monte-Carlo sampling on a mesh
# TODO - Clean this function up


def sample_by_area2(v,f):
    area = face_areas(v, f)
    faces = v[f, :]
    total_surface_area = np.sum(area)

    set_P = []
    # This looks like a random sampling by area:
    for i in range(faces.shape[0]):
        num_gen = area[i] / total_surface_area * 10000
        for j in range(int(num_gen) + 1):
            r1, r2 = np.random.rand(2)
            d = (1 - np.sqrt(r1)) * faces[i, 0] + np.sqrt(r1) * (1 - r2) * faces[i, 1] + np.sqrt(r1) * r2 * faces[i, 2]
            set_P.append(d)

    return np.array(set_P)

def sample_by_area(v, f, N):
    """
    Randomly sample points by area on a triangle mesh.  This function is
    extremely fast by using broadcasting/numpy operations in lieu of loops
    Parameters
    ----------
    v : ndarray (nv, 3)
        Array of points in 3D
    f : ndarray (nf, 3)
        Array of triangles connecting points, pointing to vertex indices
    N : int
        Number of points to sample
    colPoints : boolean (default True)
        Whether the points are along the columns or the rows

    Returns
    -------
    (Ps : NDArray (npoints, 3) array of sampled points,
     Ns : Ndarray (npoints, 3) of normals at those points   )
    """
    ###Step 1: Compute cross product of all face triangles and use to compute
    # areas and normals (very similar to code used to compute vertex normals)

    # Vectors spanning two triangle edges
    P0 = v[f[:, 0], :]
    P1 = v[f[:, 1], :]
    P2 = v[f[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    FNormals = np.cross(V1, V2)
    FAreas = np.sqrt(np.sum(FNormals ** 2, 1)).flatten()

    # Get rid of zero area faces and update points
    f = f[FAreas > 0, :]
    FNormals = FNormals[FAreas > 0, :]
    FAreas = FAreas[FAreas > 0]
    P0 = v[f[:, 0], :]
    P1 = v[f[:, 1], :]
    P2 = v[f[:, 2], :]

    # Compute normals
    NTris = f.shape[0]
    FNormals = FNormals / FAreas[:, None]
    FAreas = 0.5 * FAreas
    FNormals = FNormals
    VNormals = np.zeros_like(v)
    VAreas = np.zeros(v.shape[0])
    for k in range(3):
        VNormals[f[:, k], :] += FAreas[:, None] * FNormals
        VAreas[f[:, k]] += FAreas
    # Normalize normals
    VAreas[VAreas == 0] = 1
    VNormals = VNormals / VAreas[:, None]

    ###Step 2: Randomly sample points based on areas
    FAreas = FAreas / np.sum(FAreas)
    AreasC = np.cumsum(FAreas)
    samples = np.sort(np.random.rand(N))
    # Figure out how many samples there are for each face
    FSamples = np.zeros(NTris, dtype=np.int32)
    fidx = 0
    for s in samples:
        while s > AreasC[fidx]:
            fidx += 1
        FSamples[fidx] += 1
    # Now initialize an array that stores the triangle sample indices
    tidx = np.zeros(N, dtype=np.int64)
    idx = 0
    for i in range(len(FSamples)):
        tidx[idx:idx + FSamples[i]] = i
        idx += FSamples[i]
    N = np.zeros((N, 3))  # Allocate space for normals
    idx = 0

    # Vector used to determine if points need to be flipped across parallelogram
    V3 = P2 - P1
    V3 = V3 / np.sqrt(np.sum(V3 ** 2, 1))[:, None]  # Normalize

    # Randomly sample points on each face
    # Generate random points uniformly in parallelogram
    u = np.random.rand(N, 1)
    v = np.random.rand(N, 1)
    Ps = u * V1[tidx, :] + P0[tidx, :]
    Ps += v * V2[tidx, :]
    # Flip over points which are on the other side of the triangle
    dP = Ps - P1[tidx, :]
    proj = np.sum(dP * V3[tidx, :], 1)
    dPPar = V3[tidx, :] * proj[:, None]  # Parallel project onto edge
    dPPerp = dP - dPPar
    Qs = Ps - dPPerp
    dP0QSqr = np.sum((Qs - P0[tidx, :]) ** 2, 1)
    dP0PSqr = np.sum((Ps - P0[tidx, :]) ** 2, 1)
    idxreg = np.arange(N, dtype=np.int64)
    idxflip = idxreg[dP0QSqr < dP0PSqr]
    u[idxflip, :] = 1 - u[idxflip, :]
    v[idxflip, :] = 1 - v[idxflip, :]
    Ps[idxflip, :] = P0[tidx[idxflip], :] + u[idxflip, :] * V1[tidx[idxflip], :] + v[idxflip, :] * V2[tidx[idxflip], :]

    # Step 3: Compute normals of sampled points by barycentric interpolation
    Ns = u * VNormals[f[tidx, 1], :]
    Ns += v * VNormals[f[tidx, 2], :]
    Ns += (1 - u - v) * VNormals[f[tidx, 0], :]

    return (Ps, Ns)


# TODO - clean this function up
def random_mesh_samples(v, f, n_samples=10 ** 4):
    """
    Generate `n_samples` point samples on the mesh described by (v, f)
    :param v: A [n, 3] array of vertex positions
    :param f: A [n, 3] array of indices into v
    :param n_samples: The number of samples to generate
    :return: (P, face_ids) where P is an array of shape [n_samples, 3] and face_ids[i] is the face which P[i, :] lies on
    """
    vec_cross = np.cross(v[f[:, 0], :] - v[f[:, 2], :],
                         v[f[:, 1], :] - v[f[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Contributed by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = v[f[sample_face_idx, 0], :]
    B = v[f[sample_face_idx, 1], :]
    C = v[f[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    return P, sample_face_idx


# ---------------------------------------------------------------------------------------------------------------------#
#                                                         TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _sample_tester():
    from geom.mesh.io.base import read_mesh
    from cfg import TEST_MESH_HUMAN_PATH

    v, f = read_mesh(TEST_MESH_HUMAN_PATH)


if __name__ == '__main__':
    _sample_tester()
