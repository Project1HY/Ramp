from geom.mesh.op.cpu.spectra import laplacian_spectrum
from geom.mesh.op.cpu.base import vertex_mask_indicator
import numpy as np
from util.strings import warn
from util.container import is_integer


def heat_kernel_signature(v, f, t=(5e-3, 1e1, 10), k=200, scale_invariant=False, max_lambda_t=4 * np.log(10)):
    """
    Given a triangle mesh, approximate its curvature at some measurement scale by recording the amount of heat that
    remains at each vertex after a unit impulse of heat is applied.
    Mathematical relation to the Gaussian Curvature:
    HKS(v,t) = 1/(4*pi*t) + K(v)/(6*pi) + O(t)  Where K(v) is the Gaussian curvature at vertex v

    Properties:
    1) An approximation to both the heat kernel and the scalar/Gaussian curvature, which for 2D, characterize the
    surface completely (up to an isometry).
    2) Descriptor holds both local and global properties, with no information when t->0 or t->inf. Represents
    increasingly global properties of the shape with increasing time

    Hyper-parameters:
    # Original article used: k=300
    # Time domain: Log sampled from 4ln10/lambda_300 -> 4ln10/lambda_2

    :param v: ndarray (nv, 3) - Vertices
    :param f: ndarray (nf, 3) - Triangular Faces
    :param k: int - Number of eigenvalues to use
    :param t: Two options:
              (1) ndarray(T) - The time scales at which to compute the HKS
              (2) int - Sample t sample lograithmitcally from
              [t_min = max_lambda_t / max(lambda) -> t_max = max_lambda_t/ lambda2]
              (3) tuple (tmin,tmax,N
    :param max_lambda_t : if t is an ndarray - ignored. Else, as stated above
    :param scale_invariant: Whether to compute the SIHKS
    :return: hks: ndarray (nv, T) - A array of the heat kernel signatures at each of nv points at T time intervals

    """
    # NOTE - sometimes 2*barycenter is used, and sometimes no mass matrix is used.
    lambd, phi, _, M = laplacian_spectrum(v=v, f=f, k=k, laplacian_cls='cotangent', mass_cls='barycenter')

    # Sanity check
    if len(np.unique(np.round(lambd, 8))) != len(lambd):
        warn('Spectrum is not unique - HKS is invalid')

    if scale_invariant:  # Compute the scale invariant descriptor
        total_area = np.sum(M.diagonal())
        M /= total_area  # Areas now sum to 1
        lambd *= total_area
        phi *= np.sqrt(total_area)

    if is_integer(t):  # Number of samples
        lambd_max, lambd_min = np.abs(lambd[-1]), np.abs(lambd[0])
        assert lambd_max > lambd_min
        t_min, t_max = max_lambda_t / lambd_max, max_lambda_t / lambd_min
        t = np.exp(np.linspace(np.log(t_min), np.log(t_max), t))
    elif isinstance(t, tuple):
        t = np.exp(np.linspace(np.log(t[0]), np.log(t[1]), t[2]))

    res = (phi[:, :, np.newaxis] ** 2) * np.exp(-lambd[np.newaxis, :, np.newaxis] * t[np.newaxis, np.newaxis, :])
    return np.sum(res, 1)


def heat_equation_solution(v, f, vi, t, k=200, heat_val=100.0):
    """
    Simulate heat flow by projecting initial conditions
    onto the eigenvectors of the Laplacian matrix, and then sum up the heat
    flow of each eigenvector after it's decayed after an amount of time t
    Returns
    -------
    heat : ndarray (N) holding heat values at each vertex on the mesh
    """
    lambd, phi, _, M = laplacian_spectrum(v=v, f=f, k=k, laplacian_cls='cotangent', mass_cls='barycenter')
    initial_cond = vertex_mask_indicator(v.shape[0], vi, heat_val)

    # TODO - properly vectorize this piece of code
    ans = []
    if not isinstance(t, (tuple, list, np.ndarray)):
        t = [t]
    coeffs = (initial_cond[np.newaxis, :].dot(phi)).flatten()
    for ti in t:
        coeffs *= np.exp(-lambd * ti)
        heat = phi.dot(coeffs[:, np.newaxis])
        ans.append(heat)
    return np.column_stack(ans)


def wave_kernel_signature(Evec, Eval):
    # TODO - clean this up
    """
    Unlike HKS, the WKS can be intuited as a set of band-pass filters leading to better feature localization.
    However, the WKS does not represent large-scale features well (as they are filtered out)
    yielding poor performance at shape matching applications.
    :param Evec:
    :param Eval:
    :return:
    """
    # n = G.shape[0]
    # Evec, Eval = np.linal.eig(G)
    # Make sure the eigenvalues are sorted in ascending order
    # idx = np.argsort(Eval)
    # Eval, Evec = Eval[idx], Evec[:,idx]
    n = Evec.shape[0]

    E = abs(np.real(Eval))
    PHI = np.real(Evec)

    wks_variance = 6
    N = 100
    WKS = np.zeros(n * N).reshape((n, N))

    log_E = np.log(np.maximum(abs(E), 1e-6 * np.ones(n)))
    e = np.linspace(log_E[1], max(log_E) / 1.02, N)
    sigma = (e[1] - e[0]) * wks_variance

    C = np.zeros(N)  # weights used for the normalization of f_E

    # Could be vectorized
    for i in range(N):
        WKS[:, i] = np.sum(np.multiply(np.power(PHI, 2),
                                       np.tile(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)), (n, 1))),
                           axis=1).ravel()
        C[i] = np.sum(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)))

    # normalize WKS
    WKS = WKS / np.tile(C, (n, 1))
    return WKS


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def descriptors_normalization(desc, M):
    """
    Normalize the descriptors w.r.t. the mesh area
    :param desc: a set of descriptors defined on mesh S
    :param M: the area matrix of the mesh S
    :return: the normalized desc with the same size
    """
    tmp1 = np.sqrt(np.matmul(desc.transpose(), np.matmul(M.toarray(), desc)).diagonal())
    tmp2 = np.tile(tmp1, (desc.shape[0], 1))
    desc_normalized = np.divide(desc, tmp2)
    return desc_normalized


# ---------------------------------------------------------------------------------------------------------------------#
#                                                         TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _descriptor_tester():
    from geom.mesh.io.base import read_mesh
    from cfg import TEST_MESH_HUMAN_PATH
    from geom.mesh.vis.base import plot_mesh_montage

    v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    # HKS = heat_kernel_signature(v, f)
    # plot_mesh_montage([v] * 10, f, clrb=HKS.transpose())
    t = np.logspace(-5, 1)
    print(t)
    heat = heat_equation_solution(v, f, t=t, vi=[3])
    plot_mesh_montage([v] * len(t), f, clrb=heat.transpose())


if __name__ == '__main__':
    _descriptor_tester()
