import numpy as np
from geom.mesh.op.cpu.metric import nearest_neighbor
from geom.mesh.op.cpu.remesh import voxel_down_sample
from probreg import cpd, filterreg, gmmtree
import sklearn


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def _standardize_pose(self):
    """
    Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
    centroid and the principal axes are aligned with the vertical Z, Y, and X axes.
    # TODO - Clean this function up
    Returns:
    Nothing. Modified the mesh in place (for now)
    """
    self.mesh_.center_vertices_bb()
    vertex_array_cent = np.array(self.mesh_.vertices())

    # find principal axes
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(vertex_array_cent)

    # count num vertices on side of origin wrt principal axes
    comp_array = pca.components_
    norm_proj = vertex_array_cent.dot(comp_array.T)
    opposite_aligned = np.sum(norm_proj < 0, axis=0)
    same_aligned = np.sum(norm_proj >= 0, axis=0)
    pos_oriented = 1 * (same_aligned > opposite_aligned)  # trick to turn logical to int
    neg_oriented = 1 - pos_oriented

    # create rotation from principal axes to standard basis
    target_array = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # Z+, Y+, X+
    target_array = target_array * pos_oriented + -1 * target_array * neg_oriented
    R = np.linalg.solve(comp_array, target_array)
    R = R.T

    # rotate vertices, normals and reassign to the mesh
    vertex_array_rot = R.dot(vertex_array_cent.T)
    vertex_array_rot = vertex_array_rot.T
    self.mesh_.set_vertices(vertex_array_rot.tolist())
    self.mesh_.center_vertices_bb()


def align_pointcloud_to_z_axis(v):
    from geom.mesh.op.cpu.base import mean_center
    from numpy.linalg import svd
    m = mean_center(v)
    gram_matrix = m.T @ m
    principle_comps, _, _ = svd(gram_matrix)
    m = m @ -principle_comps  # Now primary axis is x,y,z
    m[:, [0, 2]] = m[:, [2, 0]]  # Now primary axis is z,y,x

    # Now move z to origin:
    m[:, 2] -= m[:, 2].min()

    return m


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def coherent_point_drift(source, target, tf_type_name='rigid', maxiter=50, tol=1e-5, voxel_size=0.08):
    """
    Coherent Point Drift method (2010)
    See: https://arxiv.org/pdf/0905.2635.pdf
    tf_type_name: Transformation name - one of  ('rigid', 'affine', 'nonrigid')
    :return:
    """
    ds_source = voxel_down_sample(source, voxel_size)
    ds_target = voxel_down_sample(target, voxel_size)
    tf_param, _, _ = cpd.registration_cpd(ds_source, ds_target, tf_type_name=tf_type_name,
                                          maxiter=maxiter, tol=tol)
    return tf_param.transform(source)


def filter_reg(source, target, target_normals=None, objective_type='pt2pt', maxitr=50, tol=1e-5):
    """
    FilterReg (CVPR2019)
    See: https://arxiv.org/pdf/1811.10136.pdf
    :param source: Source point cloud data.
    :param target: Target point cloud data.
    :param target_normals: (Optional) Normal vectors of target point cloud.
    :param objective_type: The type of objective function selected by 'pt2pt' or 'pt2pl'.
    :param maxitr: Maximum number of iterations to EM algorithm.
    :param tol: Tolerance for termination.
    """
    # target_normals = estimate_vertex_normals(target)
    tf_param, _, _ = filterreg.registration_filterreg(source, target, target_normals=target_normals,
                                                      objective_type=objective_type, maxiter=maxitr, tol=tol)
    return tf_param.transform(source)


def gmm_tree(source, target, maxiter=40, tol=1e-6, voxel_size=0.05):
    """"
    Usage Note:
    1) Pretty slow
    2) Especially for gmmreg and gmmtree, the calculation of Gaussian distribution is 0 for point groups that
    are too far apart, so the solution cannot be obtained well. Centers must be done in advance
    """
    ds_source = voxel_down_sample(source, voxel_size)
    ds_target = voxel_down_sample(target, voxel_size)
    tf_param, _ = gmmtree.registration_gmmtree(ds_source, ds_target, maxiter=maxiter, tol=tol)
    return tf_param.transform(source)


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp(source, target, init_pose=None, max_iterations=100, tolerance=1e-8):
    """
    # TODO - Works like shit. Similar to Open3D version
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """
    # T = o3.registration.registration_icp(numpy2open3D(src), numpy2open3D(tgt), 1,
    #                                        np.identity(4), o3.registration.TransformationEstimationPointToPoint(),
    #                                        o3.registration.ICPConvergenceCriteria(max_iteration=100)).transformation
    #     moved = src @ T[:3, :3] + T[:3, 3]
    assert source.shape == target.shape

    # get number of dimensions
    m = source.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, source.shape[0]))
    dst = np.ones((m + 1, target.shape[0]))
    src[:m, :] = np.copy(source.T)
    dst[:m, :] = np.copy(target.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(source, src[:m, :].T)
    print(f'Converged in {i} iterations with error {mean_error}')
    # Can also return the distance per vertex, the number of iterations, T or the final error
    return source @ T[:3, :3] + T[:3, 3]


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def _alignment_test():
    # IMPORTS:
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import plot_mesh, add_mesh
    from geom.matrix.transformations import rotation_matrix
    from cfg import TEST_MESH_HUMAN_PATH, TEST_SCAN_PATH

    # Prepare meshes
    src, f = read_mesh(TEST_SCAN_PATH)
    print(src.shape[0])
    src = voxel_down_sample(src, 0.01)
    print(src.shape[0])
    plot_mesh(src)
    R = rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
    # src +=0.5
    tgt = src @ R  # + np.array([0.3, 0.5, 2]).T

    # moved = icp(src, tgt)
    moved = coherent_point_drift(src, tgt)
    # moved = gmm_tree(src,tgt)
    # moved = filter_reg(src,tgt)

    print(f'ERROR is {np.linalg.norm(tgt - moved)}')
    print(f'Distance from source  {np.linalg.norm(src - moved)}')

    # PLOT:
    p, _ = plot_mesh(src, clr='red', show_on=False, strategy='points')
    add_mesh(p, tgt, f, clr='blue', strategy='wireframe')
    add_mesh(p, moved, clr='cyan', strategy='spheres')
    p.show()


def _align_to_z_plot_tester():
    from cfg import TEST_MESH_HUMAN_PATH
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import add_floor_grid, add_mesh, add_spheres, plotter
    v, f = read_mesh(TEST_MESH_HUMAN_PATH)

    p = plotter()
    # Add Floor Grid:
    add_floor_grid(p, camera_pos=None)
    # Add Z-Aligned Mesh
    v = align_pointcloud_to_z_axis(v)
    add_mesh(p, v, f, strategy='mesh', clr='cyan', lighting=True, camera_pos=None)
    # Add Origin Sphere
    add_spheres(p, v=np.zeros((1, 3)), sphere_clr='navy', camera_pos=None)

    # Play with camera position:
    cpos = p.camera_position  # [camera_position , focus_point_position, camera_up_direction]
    print(cpos)

    # Set required camera position:
    p.camera_position = ((-9, 0, 2), (0, 0, 0.8), (0.1, 0, 1))
    cpos = p.camera_position

    # Plot the camera position sphere in black:
    # camera_pos = np.zeros((1, 3))
    # camera_pos[:, :] = cpos[0]
    # add_spheres(p, v=camera_pos, sphere_clr='black', camera_pos=None)
    #
    # # Plot the focal point in pink:
    # focal_pos = np.zeros((1, 3))
    # focal_pos[:, :] = cpos[1]
    # add_spheres(p, v=focal_pos, sphere_clr='pink', camera_pos=None)

    # Auto-close is False so camera position can be returned
    p.show(auto_close=False)
    print(p.camera_position)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    _align_to_z_plot_tester()
