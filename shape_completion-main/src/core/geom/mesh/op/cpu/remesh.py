import numpy as np
from geom.mesh.op.cpu.base import flip_vertex_mask, is_connected,n_comps
from geom.mesh.io.base import numpy2open3d_mesh, open3d_mesh2numpy, numpy2open3d_cloud, open3d_cloud2numpy
# from geom.mesh.op.cpu.todo.open3d_tutorial import get_armadillo_mesh
import scipy
from util.time import time_me
import torch

# TODO - Meshfix
# [1] Collapse short edges / split long edges
# [2] Obtuse triangles are often not desirable due to their geometric nature
# (e.g. circumcenter of obtuse triangles are outside of the triangle).
# Each obtuse triangle can always be split into 2 or more right or sharp triangles.
# ---------------------------------------------------------------------------------------------------------------------#
#                                                           Mesh Decimation
# ---------------------------------------------------------------------------------------------------------------------#
def remove_faces(v, f, fi):
    # TODO - check this
    return open3d_mesh2numpy(numpy2open3d_mesh(v, f).remove_triangles_by_index(list(fi)).remove_unreferenced_vertices())


def remove_vertices(v, f, vi):
    return trunc_to_vertex_mask(v, f, flip_vertex_mask(v.shape[0], vi))

def centralize_mesh(v,com):
    com[:,3:]=0
    com=com.unsqueeze(1)
    # assert False, f"com shape {com.shape}"
    v=v-com
    return v

def trunc_to_vertex_mask(v, f, vi,clean=False):
    if f is None:
        return v[vi, :], None
    else:  # Also have to truncate faces
        nv = v.shape[0]
        # Compute map from old vertex indices to new vertex indices
        vlut = np.full((nv,), fill_value=-1)
        vlut[vi] = np.arange(len(vi))  # Bad vertices have no mapping, and stay -1.
        # Change vertex labels in face array. Bad vertices have no mapping, and stay -1.
        f2 = vlut[f]
        # Keep only faces with valid vertices:
        f2 = f2[np.sum(f2 == -1, axis=1) == 0, :]

        # TODO - Can we use this instead of clean_mesh?
        # faces = faces.reshape(-1)
        # unique_points_index = np.unique(faces)
        # unique_points = pts[unique_points_index]
        if clean:
            return clean_mesh(v[vi, :], f2, False, False)
        else:
            return v[vi, :], f2


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
@time_me
def mesh_fix(v, f):
    """
    MeshFix corrects topological errors in polygonal meshes.
    This software takes as input a polygon mesh and produces a copy of the input where all the occurrences of a specific
    set of "defects" are corrected. MeshFix has been designed to correct typical flaws present in RAW DIGITIZED mesh
    models, thus it might fail or produce coarse results if run on other sorts of input meshes
    (e.g. tessellated CAD models). When the software fails, it appends a textual description to the file "meshfix.log".
    The input is assumed to represent a single CLOSED SOLID OBJECT, thus the output will be a SINGLE WATERTIGHT TRIANGLE
    MESH bounding a polyhedron. All the singularities, self-intersections and degenerate elements are removed from the
    input, while regions of the surface without defects are left unmodified.
    - C++ source last updated 1 Jul 2020
    """
    from pymeshfix import _meshfix
    # Generate vertex and face arrays of cleaned mesh
    # where v and f are numpy arrays or python lists
    return _meshfix.clean_from_arrays(v, f)
    # # Create TMesh object
    # tin = _meshfix.PyTMesh()
    #
    # tin.LoadFile(infile)
    # # tin.load_array(v, f) # or read arrays from memory
    #
    # # Attempt to join nearby components
    # # tin.join_closest_components()
    #
    # # Fill holes
    # tin.fill_small_boundaries()
    # print('There are {:d} boundaries'.format(tin.boundaries())
    #
    # # Clean (removes self intersections)
    # tin.clean(max_iters=10, inner_loops=3)
    #
    # # Check mesh for holes again
    # print('There are {:d} boundaries'.format(tin.boundaries())
    #
    # # Clean again if necessary...
    #
    # # Output mesh
    # tin.save_file(outfile)
    #
    # # or return numpy arrays
    # vclean, fclean = tin.return_arrays()


def clean_mesh_vtk(v, f):
    pass
    # TODO - compare against open3d
    # merge duplicate points, and/or remove unused points and/or remove degenerate cells
    # import vtk
    # # TODO - just use PyVista directly...
    # # TODO - Check vtk capabilties
    # cleaner = vtk.vtkCleanPolyData()
    # cleaner.SetInputData(mesh)
    # cleaner.PointMergingOn()
    # # cleaner.SetTolerance(0.0)
    # cleaner.Update()
    # return cleaner.GetOutput()


def remove_non_manifold_edges(v, f):
    return open3d_mesh2numpy(numpy2open3d_mesh(v, f).remove_non_manifold_edges())


def clean_mesh(v, f, merge_eps=1e-5, remove_outlier=True, outlier_radius=0.05, nb_points=1):
    if remove_outlier:
        # Take a sphere of radius 0.05 around each point and expect at least nb_points in the sphere - or remove
        P, ind = numpy2open3d_cloud(v).remove_radius_outlier(nb_points=nb_points, radius=outlier_radius)
        # Compute average distance to neighbors and remove if further than statistics of std_ratio
        # P, ind = numpy2open3d_cloud(v).remove_statistical_outlier(nb_neighbors=5,std_ratio=7.0)

        v2, f2 = trunc_to_vertex_mask(v, f, ind)  # Does the cleaning inside

        # Small debug suite:
        # print('Outliers Removed', v.shape[0] - v2.shape[0], f.shape[0] - f2.shape[0])
        # if v2.shape[0] != v.shape[0]:
        #     from geom.mesh.vis.base import plot_mesh_montage
        #     plot_mesh_montage([v, v2], [f, f2], strategy='mesh')
        #     plot_mesh_montage([v, v2], [f, f2], strategy='spheres')
        return v2, f2
    else:  # Basic Cleaning:

        # Resolves a very odd bug where the faces reference non-existent vertices:
        M = numpy2open3d_mesh(v, f)
        bad_faces = list(np.where(np.sum(f > v.shape[0], axis=1) > 0)[0])
        if bad_faces:
            M = M.remove_triangles_by_index(bad_faces)
        # Merge points that sit atop each other
        M = M.merge_close_vertices(merge_eps)
        M = M.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        # Small code for remove_unreferenced_vertices
        # bad_vertices= list(set(f.ravel()) - set(np.unique(f.ravel())))
        # v,f = remove_vertices(v,f,bad_vertices)
        v2, f2 = open3d_mesh2numpy(M)

    assert np.max(f2) + 1 <= v2.shape[0], "Invalid mesh cleaning detected"
    return v2, f2


def print_mesh_status(v, f):
    M = numpy2open3d_mesh(v, f)
    print(f"[1] edge_manifold:          {M.is_edge_manifold(allow_boundary_edges=True)}")
    print(f"[2] edge_manifold_boundary: {M.is_edge_manifold(allow_boundary_edges=False)}")
    print(f"[3] vertex_manifold:        {M.is_vertex_manifold()}")  # TODO - what does this mean?
    # print(f"[4] self_intersecting:      {M.is_self_intersecting()}")  # TODO - expensive compute
    print(f"[5] watertight:             {M.is_watertight()}")  # TODO - why is this so expensive?
    print(f"[6] orientable:             {M.is_orientable()}")
    print(f"[7] connected               {is_connected(v, f)} [{n_comps(v,f)}]")  # No open3d alternative


def orient_triangles(v, f):
    # If the mesh is orientable this function orients all triangles such
    # that all normals point towards the same direction.
    return open3d_mesh2numpy(numpy2open3d_mesh(v, f).orient_triangles())  # TODO - check me


# ---------------------------------------------------------------------------------------------------------------------#
#                                           Downsample - TODO - move to sample
# ---------------------------------------------------------------------------------------------------------------------#
def voxel_down_sample(v, voxel_size):
    return open3d_cloud2numpy(numpy2open3d_cloud(v).voxel_down_sample(voxel_size=voxel_size))


# ---------------------------------------------------------------------------------------------------------------------#
#                                           Upsample - TODO - move to sample
# ---------------------------------------------------------------------------------------------------------------------#
def subdivide(v, f, fi=None, n=1):
    # May be implemented via https://github.com/PyMesh/PyMesh/blob/master/python/pymesh/meshutils/generate_box_mesh.py
    # face_index : faces to subdivide.
    #   if None: all faces of mesh will be subdivided
    #   if (n,) int array of indices: only specified faces
    import trimesh.remesh
    for i in range(n):
        v, f = trimesh.remesh.subdivide(v, f, face_index=fi)
    return v, f


def subdivide_to_size(v, f, max_edge, max_iter=10):
    """
    Subdivide a mesh until every edge is shorter than a specified length.
    Will return a triangle soup, not a nicely structured mesh.
    """
    import trimesh.remesh
    return trimesh.remesh.subdivide_to_size(v, f, max_edge, max_iter=max_iter)


# ---------------------------------------------------------------------------------------------------------------------#
#                                               Point-cloud Triangulation
# ---------------------------------------------------------------------------------------------------------------------#

def triangulate_by_convex_hull(v):
    """
    Triangulate a convex set of vertices
    """
    hull = scipy.spatial.ConvexHull(v)
    return v, hull.simplices  # .fix_orientation().fix_curvature()
    # TODO - see also create_from_point_cloud_alpha_shape(alpha, alpha, tetra_mesh, pt_map)


def triangulate_by_ball_pivot(pcd, radii):
    # TODO
    """
    Function that computes a triangle mesh from a oriented PointCloud.
    This implements the Ball Pivoting algorithm proposed in F. Bernardini et al.,
    “The ball-pivoting algorithm for surface reconstruction”, 1999.
    The implementation is also based on the algorithms outlined in Digne,
    “An Analysis and Implementation of a Parallel Ball Pivoting Algorithm”, 2014.
    The surface reconstruction is done by rolling a ball with a given radius over the point
    cloud, whenever the ball touches three points a triangle is created.

    Q.A:
    [1] How is the ball radius chosen? The radius, is obtained empirically based on the size and scale of the input
    point cloud. In theory, the diameter of the ball should be slightly larger than the average distance between points.

    [2] What if the points are too far apart at some locations and the ball falls through?
    When the ball pivots along an edge, it may miss the appropriate point on the surface and
    instead hit another point on the object or even exactly its three old points.
    In this case, we check that the normal of the new triangle Facet is consistently
    oriented with the point's Vertex normals. If it is not, then we reject that triangle and create a hole.

    [3] What if the surface has a crease or valley, such that the distance between the surface and
    itself is less than the size of the ball? In this case, the ball would just roll over the crease
    and ignore the points within the crease. But, this is not ideal behavior as the
    reconstructed mesh is not accurate to the object.

    [4] What if the surface is spaced into regions of points such that the ball cannot successfully roll
    between the regions? The virtual ball is dropped onto the surface multiple times at
    varying locations. This ensures that the ball captures the entire mesh, even when the points are
    inconsistently spaced out.

    :param pcd: pcd (open3d.geometry.PointCloud) – PointCloud from which the TriangleMesh surface is reconstructed.
    Has to contain normals.
    :param radii: radii (open3d.utility.DoubleVector) – The radii of the ball
    that are used for the surface reconstruction.
    :return:
    """
    pass
    # # https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    # import open3d
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3 * avg_dist
    # open3d.utility.DoubleVector([radius, radius * 2])
    # # Before exporting the mesh, we can downsample the result to an acceptable number of triangles,
    # # for example, 100k triangles:
    # dec_mesh = mesh.simplify_quadric_decimation(100000)
    # return open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    # # TODO - see also poisson reconstruction on the same page


def triangulate_by_poisson_reconstruction(v):
    """
    The Poisson Reconstruction is a bit more technical/mathematical. Its approach is known as an implicit meshing
    method, which I would describe as trying to “envelop” the data in a smooth cloth.
    Without going into too many details, we try to fit a watertight surface from the original point set by
    creating an entirely new point set representing an isosurface linked to the normals.
    There are several parameters available that affect the result of the meshing:

    Q.A:
    Which depth? a tree-depth is used for the reconstruction. The higher the more detailed the mesh (Default: 8).
    With noisy data you keep vertices in the generated mesh that are outliers but the algorithm doesn’t detect
    them as such. So a low value (maybe between 5 and 7) provides a smoothing effect, but you will lose detail.
    The higher the depth-value the higher is the resulting amount of vertices of the generated mesh.

    Which width? This specifies the target width of the finest level of the tree structure, which is called an octree.
    Don’t worry, I will cover this and best data structures for 3D in another article as it extends the scope of this
    one. Anyway, this parameter is ignored if the depth is specified.

    Which scale? It describes the ratio between the diameter of the cube used for reconstruction and the diameter of the
    samples’ bounding cube. Very abstract, the default parameter usually works well (1.1).

    Which fit? the linear_fit parameter if set to true, let the reconstructor use linear interpolation to estimate
    the positions of iso-vertices.
    """
    pass
    # # https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8,
    #                                                                          width=0, scale=1.1, linear_fit=False)[0]
    # # For this, we compute the initial bounding-box containing the raw point cloud,
    # # and we use it to filter all surfaces from the mesh outside the bounding-box:
    # bbox = pcd.get_axis_aligned_bounding_box()
    # p_mesh_crop = poisson_mesh.crop(bbox)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                         TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _remesh_tester():
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import plot_mesh
    from cfg import TEST_MESH_HUMAN_PATH
    from util.time import timer
    import trimesh
    import open3d as o3d
    # pcd = get_armadillo_mesh().sample_points_poisson_disk(7000)
    # v,f = open3d_mesh2numpy(get_armadillo_mesh())
    v,f = read_mesh(TEST_MESH_HUMAN_PATH)
    pcd = numpy2open3d_cloud(v)
    # pcd = get_armadillo_mesh()
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100

    print("Get all points that are visible from given view point")
    # from util.time
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    v,f = remove_vertices(v,f,pt_map)
    plot_mesh(v,f)
    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])
    # t = trimesh.load_mesh(TEST_MESH_HUMAN_PATH)
    # v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    # with timer():
    #     print_mesh_status(v, f)
    # # o3d.visualization.draw_geometries([numpy2open3d_cloud(v)])
    # mesh1 = numpy2open3d_mesh(v, f)
    # mesh2 = numpy2open3d_mesh(v, f)
    # mesh3 = numpy2open3d_mesh(v, f)
    # o3d.visualization.draw_geometries([mesh1, numpy2open3d_cloud(v)])
    # v, f = fix_mesh(v, f)
    import open3d
    # M = open3d.geometry.TriangleMesh(v, f)
    # plot_mesh(v, f)


if __name__ == '__main__':
    from geom.mesh.io.base import read_mesh
    v,f=read_mesh(r"C:\Users\idoim\Desktop\camera_remeshed.ply")
    print_mesh_status(v,f)
    # _remesh_tester()
