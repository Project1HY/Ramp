import numpy as np
from sklearn.preprocessing import normalize


# TODO - Consider migrating to util.np

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def mse(A, B, ax=None):
    return ((A - B) ** 2).mean(axis=ax)


def mean_l2_err(A, B, ax=None):
    # Computes the RMSE
    # ax = 0 - Perform along rows, returning a value for each column
    return np.sqrt(((A - B) ** 2).mean(axis=ax))


def mean_l1_err(A, B, ax=None):
    return np.abs(A - B).mean(axis=ax)


def total_l1_err(A, B):
    return np.sqrt(np.sum(np.abs(A - B)))


def total_l2_err(A, B):
    return np.sum(((A - B) ** 2))


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def col_normalize(mat):
    return normalize(mat, norm='l2', axis=0)


def row_normalize(mat):
    return normalize(mat, norm='l2', axis=1)


def vec_normalize(vec):
    return normalize(vec, norm='l2')


def last_axis_normalize(mat):
    """ normalize array of vectors along the last axis """
    return mat / last_axis_2norm(mat)[..., np.newaxis]


def last_axis_2norm(mat):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(mat ** 2, axis=-1))


def forbenious_norm(mat):
    return np.sqrt(np.sum(mat ** 2))


def unique_rows(a):
    # The cleaner alternative `numpy.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    )
    a_unique, inv, cts = np.unique(b, return_inverse=True, return_counts=True)
    a_unique = a_unique.view(a.dtype).reshape(-1, a.shape[1])
    return a_unique, inv, cts


def row_intersect(A, B, assume_unique=True):
    # Remember that there is a problem if the array rows are not unique
    # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C, iA2B, _ = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True, assume_unique=assume_unique)

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C, np.sort(iA2B)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def random_sign_vector(N):
    return np.random.choice([-1, 1], size=N)


def stabilized_log_sum_exp(x):
    max_x = x.max(2)
    x = x - max_x[:, :, np.newaxis]
    ret = np.log(np.sum(np.exp(x), axis=2)) + max_x
    return ret


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def spherical2caresian(rthetaphi):
    xyz = np.zeros(rthetaphi.shape)
    r = rthetaphi[:, 0]
    theta = rthetaphi[:, 1]
    phi = rthetaphi[:, 2]
    xyz[:, 0] = r * np.sin(theta) * np.cos(phi)
    xyz[:, 1] = r * np.sin(theta) * np.sin(phi)
    xyz[:, 2] = r * np.cos(theta)
    return xyz


def cartesian2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def intrinsic_dimension(X, k1=6, k2=12, estimator='levina', trafo='var', mem_threshold=5000):
    """Calculate intrinsic dimension based on the MLE by Levina and Bickel [1].
    Parameters
    ----------
    X : ndarray
        - An ``m x n`` vector data matrix with ``n`` objects in an
          ``m`` dimensional feature space
        - An ``n x n`` distance matrix.

    k1 : int, optional (default: 6)
        Start of neighborhood range to search in.

    k2 : int, optional (default: 12)
        End of neighborhood range to search in.

    estimator : {'levina', 'mackay'}, optional (default: 'levina')
        Determine the summation strategy: see [2].
    trafo : {None, 'std', 'var'}, optional (default: 'var')
        Transform vector data.
        - None: no transformation
        - 'std': standardization
        - 'var': subtract mean, divide by variance (default behavior of
          Laurens van der Maaten's DR toolbox; most likely for other
          ID/DR techniques).
    mem_threshold : int, optional, default: 5000
        Controls speed-memory usage trade-off: If number of points is higher
        than the given value, don't calculate complete distance matrix at
        once (fast, high memory), but per row (slower, less memory).
    Returns
    -------
    d_mle : int
        Intrinsic dimension estimate (rounded to next integer)
    NOTE: the MLE was derived for euclidean distances. Using
    other dissimilarity measures may lead to undefined results.
    References
    ----------
        [1] Levina, E., Bickel, P. (2004)
        Maximum Likelihood Estimation of Intrinsic Dimension
        https://www.stat.berkeley.edu/~bickel/mldim.pdf

        [2] http://www.inference.phy.cam.ac.uk/mackay/dimension
    """
    n = X.shape[0]
    if estimator not in ['levina', 'mackay']:
        raise ValueError("Parameter 'estimator' must be 'levina' or 'mackay'.")
    if k1 < 1 or k2 < k1 or k2 >= n:
        raise ValueError(
            "Invalid neighborhood: Please make sure that 0 < k1 <= k2 < n. (Got k1={} and k2={}).".format(k1, k2))
    X = X.copy().astype(float)

    # New array with unique rows   (    % Remove duplicates from the dataset )
    X = X[np.lexsort(np.fliplr(X).T)]

    if trafo is None:
        pass
    elif trafo == 'var':
        X -= X.mean(axis=0)
        X /= X.var(axis=0) + 1e-7
    elif trafo == 'std':
        # Standardization
        X -= X.mean(axis=0)
        X /= X.std(axis=0) + 1e-7
    else:
        raise ValueError("Transformation must be None, 'std', or 'var'.")

    # Compute matrix of log nearest neighbor distances
    X2 = (X ** 2).sum(1)

    if n <= mem_threshold:  # speed-memory trade-off
        distance = X2.reshape(-1, 1) + X2 - 2 * np.dot(X, X.T)  # 2x br.cast
        distance.sort(1)
        # Replace invalid values with a small number
        distance[distance < 0] = 1e-7
        knnmatrix = .5 * np.log(distance[:, 1:k2 + 1])
    else:
        knnmatrix = np.zeros((n, k2))
        for i in range(n):
            distance = np.sort(X2[i] + X2 - 2 * np.dot(X, X[i, :]))
            # Replace invalid values with a small number
            distance[distance < 0] = 1e-7
            knnmatrix[i, :] = .5 * np.log(distance[1:k2 + 1])

    # Compute the ML estimate
    S = np.cumsum(knnmatrix, 1)
    indexk = np.arange(k1, k2 + 1)
    dhat = -(indexk - 2) / (S[:, k1 - 1:k2] - knnmatrix[:, k1 - 1:k2] * indexk)

    if estimator == 'levina':
        # Average over estimates and over values of k
        no_dims = dhat.mean()
    if estimator == 'mackay':
        # Average over inverses
        dhat **= -1
        dhat_k = dhat.mean(0)
        no_dims = (dhat_k ** -1).mean()

    return int(no_dims.round())
