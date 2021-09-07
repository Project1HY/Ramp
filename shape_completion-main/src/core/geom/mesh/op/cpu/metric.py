import numpy as np
from sklearn.neighbors import NearestNeighbors

from geom.matrix.cpu import stabilized_log_sum_exp, last_axis_2norm
from geom.mesh.op.cpu.base import face_areas,surface_area,vertex_adjacency,edge_distances
from scipy.sparse.csgraph import shortest_path
from scipy import sparse


def pdist(v1, v2, p=2):
    """
    Compute the pairwise distance matrix between a and b which both have size [m, n, d] or [n, d].
    The result is a tensor of size [m, n, n] (or [n, n]) whose entry [m, i, j] contains the distance_tensor between
    a[m, i, :] and b[m, j, :].
    :param v1: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param v2: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    squeezed = False
    if len(v1.shape) == 2 and len(v2.shape) == 2:
        v1 = v1[np.newaxis, :, :]
        v2 = v2[np.newaxis, :, :]
        squeezed = True

    if len(v1.shape) != 3:
        raise ValueError("Invalid shape for v1. Must be [m, n, d] or [n, d] but got", v1.shape)
    if len(v2.shape) != 3:
        raise ValueError("Invalid shape for v2. Must be [m, n, d] or [n, d] but got", v2.shape)

    ret = np.power(np.abs(v1[:, :, np.newaxis, :] - v2[:, np.newaxis, :, :]), p).sum(3)
    if squeezed:
        ret = np.squeeze(ret)

    return ret

# ---------------------------------------------------------------------------------------------------------------------#
#                                                  Metrics
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyArgumentList
def directional_chamfer(source, target, p=2):
    """
    Compute the chamfer distance between two point clouds a,b by the Lp norm
    :param source: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param target: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :param p: Norm to use for the distance_tensor
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    M = pdist(source, target, p)
    if len(M.shape) == 2:
        M = M[np.newaxis, :, :]

    # TODO - make sure the source and target and not reversed
    return M.min(1).sum(1), M.min(2).sum(1)  # sum the two for the full chamfer distance


# noinspection PyArgumentList
def sinkhorn(a, b, M, eps, max_iters=100, stop_thresh=1e-3):
    """
    Approximate the wasserstein distance between two point clouds
    Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
    :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
    :param M: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
             i.e. shape = [m, n, n] and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
    :param eps: The reciprocal of the sinkhorn regularization parameter
    :param max_iters: The maximum number of Sinkhorn iterations
    :param stop_thresh: Stop if the change in iterates is below this value
    :return:
    """
    # a and b are tensors of size [nb, m] and [nb, n]
    # M is a tensor of size [nb, m, n]

    M = np.squeeze(M)
    a = np.squeeze(a)
    b = np.squeeze(b)
    squeezed = False

    if len(M.shape) == 2 and len(a.shape) == 1 and len(b.shape) == 1:
        M = M[np.newaxis, :, :]
        a = a[np.newaxis, :]
        b = b[np.newaxis, :]
        squeezed = True
    elif len(M.shape) == 2 and len(a.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where m is the number of samples in a and "
                         "M has shape [m, n]" % str(a.shape))
    elif len(M.shape) == 2 and len(b.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where n is the number of samples in a and "
                         "M has shape [m, n]" % str(b.shape))

    if len(M.shape) != 3:
        raise ValueError("Got unexpected shape for M %s, should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(M.shape))
    elif len(M.shape) == 3 and len(a.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and M has shape [nb, m, n]" % str(a.shape))
    elif len(M.shape) == 3 and len(b.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and M has shape [nb, m, n]" % str(b.shape))

    nb = M.shape[0]
    m = M.shape[1]
    n = M.shape[2]

    if a.dtype != b.dtype or a.dtype != M.dtype:
        raise ValueError("Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                         % (str(a.dtype), str(b.dtype), str(M.dtype)))
    if a.shape != (nb, m):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                         str(a.shape))
    if b.shape != (nb, n):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                         str(b.shape))

    u = np.zeros_like(a)
    v = np.zeros_like(b)

    M_t = np.transpose(M, axes=(0, 2, 1))

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-M + np.expand_dims(v, 1)) / eps
        u = eps * (np.log(a) - stabilized_log_sum_exp(summand_u))

        summand_v = (-M_t + np.expand_dims(u, 1)) / eps
        v = eps * (np.log(b) - stabilized_log_sum_exp(summand_v))

        err_u = np.sum(np.abs(u_prev - u), axis=1).max()
        err_v = np.sum(np.abs(v_prev - v), axis=1).max()

        if err_u < stop_thresh and err_v < stop_thresh:
            break

    log_P = (-M + np.expand_dims(u, 2) + np.expand_dims(v, 1)) / eps

    P = np.exp(log_P)

    if squeezed:
        P = np.squeeze(P)

    return P


def euclidean_directional_hausdorff(source, target, return_index=True):
    """
    # TODO -Take a loot at scipy.spatial.distance
    The Hausdorff distance is the longest distance you can be forced to travel by an adversary who chooses
    a point in one of the two sets, from where you then must travel to the other set.
    In other words, it is the greatest of all the distances from a point in one set to
    the closest point in the other set.
    source : n by 3 array of representing a set of n points (each row is a point of dimension 3)
    target : m by 3 array of representing a set of m points (each row is a point of dimension 3)
    return_index : Optionally return the index pair `(i, j)` into source and target such that
               `source[i, :]` and `target[j, :]` are the two points with maximum shortest distance.
    # Note: Take a max of the one sided dist to get the two sided Hausdorff distance
      return max(hausdorff_a_to_b, hausdorff_b_to_a)
    :param source:
    :param target:
    :param return_index
    :return:
    """
    import point_cloud_utils as pcu
    # Compute each one sided *squared* Hausdorff distances
    return pcu.hausdorff(source, target, return_index=return_index)


def _wasserstein_example():
    # a and b are arrays where each row contains a point
    # Note that the point sets can have different sizes (e.g [100, 3], [111, 3])
    a = np.random.rand(100, 3)
    b = np.random.rand(100, 3)

    # M is a 100x100 array where each entry  (i, j) is the squared distance between point a[i, :] and b[j, :]
    M = pdist(a, b)

    # w_a and w_b are masses assigned to each point. In this case each point is weighted equally.
    w_a = np.ones(a.shape[0])
    w_b = np.ones(b.shape[0])

    # P is the transport matrix between a and b, eps is a regularization parameter, smaller epsilons lead to
    # better approximation of the true Wasserstein distance at the expense of slower convergence
    P = sinkhorn(w_a, w_b, M, eps=1e-3)

    # To get the distance as a number just compute the frobenius inner product <M, P>
    sinkhorn_dist = (M * P).sum()
    return sinkhorn_dist


def _hausdorff_example():
    import point_cloud_utils as pcu
    # Generate two random point sets
    a = np.random.rand(1000, 3)
    b = np.random.rand(500, 3)
    # dists_a_to_b is of shape (a.shape[0],) and contains the shortest squared distance
    # between each point in a and the points in b
    # corrs_a_to_b is of shape (a.shape[0],) and contains the index into b of the
    # closest point for each point in a
    dists_a_to_b, corrs_a_to_b = pcu.point_cloud_distance(a, b)

    # Compute each one sided squared Hausdorff distances
    hausdorff_a_to_b = pcu.hausdorff(a, b)
    hausdorff_b_to_a = pcu.hausdorff(b, a)

    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    hausdorff_dist = max(hausdorff_a_to_b, hausdorff_b_to_a)

    # Find the index pairs of the two points with maximum shortest distancce
    hausdorff_b_to_a, idx_b, idx_a = pcu.hausdorff(b, a, return_index=True)
    assert np.abs(np.sum((a[idx_a] - b[idx_b]) ** 2) - hausdorff_b_to_a) < 1e-5, "These values should be almost equal"


def nearest_neighbor(v1, v2, k=1):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor

    NOTE: If k is very large, it is better to use vertex_distance_matrix with knn_truncation in terms of time.
    Results are identical
    '''
    assert v1.shape == v2.shape

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(v2)
    distances, indices = neigh.kneighbors(v1, return_distance=True)
    if k==1:
        return distances.ravel(), indices.ravel()
    else:
        return distances,indices