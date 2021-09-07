import numpy as np


def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


# a = np.zeros((16,1024,3))
# print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def trim_voxels(a):
    """
    :param a: 3D Voxel Grid
    :return: A new version of a, where all the spacing around the data has been removed
    """
    bw = a > 0
    a_01 = bw.sum(axis=(0, 1))
    a_02 = bw.sum(axis=(0, 2))
    a_12 = bw.sum(axis=(1, 2))

    ind_z = np.where(a_01)[0]
    ind_y = np.where(a_02)[0]
    ind_x = np.where(a_12)[0]

    A = a[ind_x[0]:ind_x[-1]+1,
          ind_y[0]:ind_y[-1]+1,
          ind_z[0]:ind_z[-1]+1]
    return A


def smoothed_crossing_pipes(X, Y, Z, R_left, R_right):
    R_Z = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    R_Y = np.sqrt((X-0.5)**2 + (Z-0.5)**2)
    R_X = np.sqrt((Y-0.5)**2 + (Z-0.5)**2)

    R_max_Z = (R_left-R_right)*(1+np.cos(np.pi*Z))/2 + R_right
    R_max_Y = (R_left-R_right)*(1+np.cos(np.pi*Y))/2 + R_right
    R_max_X = (R_left-R_right)*(1+np.cos(np.pi*X))/2 + R_right

    S_Z = - R_Z + R_max_Z > 0
    S_Y = - R_Y + R_max_Y > 0
    S_X = - R_X + R_max_X > 0
    S = -(1-S_X)*(1-S_Y)*(1-S_Z) + 0.5

    for i in range(40):
        S = laplacian_filter(S, 0.05)

    return S


def refine_voxels(S, m=2):
    I, J, K = S.shape
    S_2 = np.zeros((m*I, m*J, m*K), dtype=bool)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                S_2[i::m, j::m, k::m] = S
    return S_2

def laplacian_filter(S, k, periodic=False):
    S_cp = np.zeros([i+2 for i in S.shape])
    S_cp[1:-1, 1:-1, 1:-1] = S
    if periodic:
        S_cp[0, :, :] = S_cp[-2, :, :]
        S_cp[:, 0, :] = S_cp[:, -2, :]
        S_cp[:, :, 0] = S_cp[:, :, -2]
        S_cp[-1, :, :] = S_cp[1, :, :]
        S_cp[:, -1, :] = S_cp[:, 1, :]
        S_cp[:, :, -1] = S_cp[:, :, 1]
    else:
        S_cp[0, :, :] = S_cp[1, :, :]
        S_cp[:, 0, :] = S_cp[:, 1, :]
        S_cp[:, :, 0] = S_cp[:, :, 1]
        S_cp[-1, :, :] = S_cp[-2, :, :]
        S_cp[:, -1, :] = S_cp[:, -2, :]
        S_cp[:, :, -1] = S_cp[:, :, -2]
    S_cp2 = np.zeros_like(S)
    S_cp2[:, :, :] = (1-6*k)*S[:, :, :]
    S_cp2[:, :, :] += k*(S_cp[:-2, 1:-1, 1:-1]
                         + S_cp[2:, 1:-1, 1:-1]
                         + S_cp[1:-1, :-2, 1:-1]
                         + S_cp[1:-1, 2:, 1:-1]
                         + S_cp[1:-1, 1:-1, :-2]
                         + S_cp[1:-1, 1:-1, 2:])
    return S_cp2
