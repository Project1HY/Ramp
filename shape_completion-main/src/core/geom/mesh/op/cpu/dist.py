import numpy as np
from geom.matrix.cpu import last_axis_2norm
from geom.mesh.op.cpu.base import surface_area, vertex_adjacency
from scipy.sparse.csgraph import shortest_path
from geom.mesh.op.cpu.metric import pdist


# ---------------------------------------------------------------------------------------------------------------------#
#                                           Distance Matrices
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyTypeChecker
def vertex_distance_matrix(v, f=None, cls='euclidean_graph', max_dist=None, knn_truncation=None):
    if cls == 'euclidean_graph':
        # Fast approximation to the true geodesic distances
        D = euclidean_graph_distance_matrix(v, f, max_dist)
    elif cls == 'euclidean_cloud':
        D = euclidean_distance_matrix(v, max_dist)
    elif cls == 'graph':
        D = graph_distance_matrix(v, f, max_dist)
    elif cls == 'exact_geodesic':
        D = exact_geodesic_distance_matrix(v, f, max_dist, normalize_by_area=False)
    elif cls == 'area_weighted_exact_geodesic':
        D = exact_geodesic_distance_matrix(v, f, max_dist, normalize_by_area=True)
    elif cls == 'fast_marching':
        D = fast_marching_distance_matrix(v, f, max_dist=None)
    else:
        raise NotImplementedError(f'Unknown class {cls}')

    if knn_truncation is not None:
        # https://github.com/lmcinnes/umap/issues/114
        indices = np.argsort(D)
        indices_row = np.multiply(np.ones(D.shape[1], dtype=np.int32), np.arange(D.shape[0], dtype=np.int32)[:, None])
        D = D[indices_row, indices]  # Sorted D
        indices = indices[:, : knn_truncation]
        D = D[:, : knn_truncation]
        return D, indices
    else:
        return D


def fast_marching_distance_matrix(v, f, max_dist=None):
    raise NotImplementedError
    # TODO - Didn't find a binding for this


def euclidean_distance_matrix(v, max_dist=None):
    # Faster than pdist(x,x,2), and equivalent up to 1e-9
    r = np.sum(v * v, 1)
    r = r.reshape(-1, 1)
    D = r - 2 * np.dot(v, v.T) + r.T
    D[D < 0] = 0  # Remove negative zeros
    D = np.sqrt(D)  # TODO - This may be removed for faster computes
    if max_dist is not None:
        D[D > max_dist] = 0  # Remove all these entries
    return D


# noinspection PyTupleItemAssignment
def graph_distance_matrix(v, f, max_dist=None):
    G = vertex_adjacency(v, f, weight=None)
    D = shortest_path(G, directed=False, return_predecessors=False, unweighted=True, indices=None)
    if max_dist is not None:
        D[D > max_dist] = 0  # Remove all these entries
    return D


# noinspection PyTupleItemAssignment
def euclidean_graph_distance_matrix(v, f, max_dist=None):
    G = vertex_adjacency(v, f, weight='euclidean')
    D = shortest_path(G, directed=False, return_predecessors=False, unweighted=False, indices=None)
    if max_dist is not None:
        D[D > max_dist] = 0  # Remove all these entries
    return D


def exact_geodesic_distance_matrix(v, f, max_dist=1.0, normalize_by_area=False):
    """
    WARNING: This functions takes extremely long to compute for the full distance matrix
    (6 minutes on a medium sized mesh)

    Compute the exact (O(h^2)) geodesic distances from every vertex on the surface to all
    those with a distance 'max_distance' of then.
    :param v: nd.array [nv x 3] mesh vertices
    :param f: nd.array [nf x 3] mesh faces
    :param max_dist: float, as described above
    :param normalize_by_area : Normalizes by sqrt(mesh_surface_area)
    :return: matrix D of scipy.sparse.csc_matrix((nv,nv), dtype=numpy.float64) where Dij is the distance
    from node i to j if under the max_distance or 0 otherwise
    """
    import gdist
    area_normalization = np.sqrt(surface_area(v, f)) if normalize_by_area else 1
    if max_dist is None:
        max_dist = float('inf')
    else:
        max_dist *= area_normalization
    return gdist.local_gdist_matrix(v, f, max_dist) / area_normalization

def spherical_distance(pt0, pt1):
    """
    spherical_distance(a, b) yields the angular distance between points a and b, both of which
      should be expressed in spherical coordinates as (longitude, latitude).
    If a and/or b are (2 x n) matrices, then the calculation is performed over all columns.
    The spherical_distance function uses the Haversine formula; accordingly it may suffer from
    rounding errors in the case of nearly antipodal points.
    """
    dtheta = pt1[0] - pt0[0]
    dphi = pt1[1] - pt0[1]
    a = np.sin(dphi / 2) ** 2 + np.cos(pt0[1]) * np.cos(pt1[1]) * np.sin(dtheta / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a))

def vtp_geodesic():
    # Fast and Exact Discrete Geodesic Computation Based on Triangle-Oriented Wavefront Propagation
    import os
    import subprocess
    import tempfile
    """Compute geodesic distance using VTP method
            VTP Code
            --------
            - uses external authors' implementation of [Qin el al 2016]
            - https://github.com/YipengQin/VTP_source_code
            - vtp code must be compiled separately to produce VTP executable
            - once compiled, place path to VTP executable in pycortex config
            - i.e. in config put:
                [geodesic]
                vtp_path = /path/to/compiled/VTP
            Parameters
            ----------
            - vertex : int
                index of vertex to compute geodesic distance from
            """

    if config.has_option('geodesic', 'vtp_path'):
        vtp_path = config.get('geodesic', 'vtp_path')
    else:
        raise AssertionError('must set config["geodesic"]["vtp_path"]')

    if not os.path.exists(vtp_path):
        raise AssertionError('vtp_path does not exist: ' + str(vtp_path))

    # initialize temporary files
    f_obj, tmp_obj_path = tempfile.mkstemp()
    f_output, tmp_output_path = tempfile.mkstemp()

    # create object file
    formats.write_obj(tmp_obj_path, self.pts, self.polys)

    # run algorithm
    cmd = [vtp_path, '-m', tmp_obj_path, '-s', str(vertex), '-o', tmp_output_path]
    subprocess.call(cmd)

    # read output
    with open(tmp_output_path) as f:
        output = f.read()
        distances = np.array(output.split('\n')[:-2], dtype=float)

    if distances.shape[0] == 0:
        raise AssertionError('VTP error')

    os.close(f_obj)
    os.close(f_output)

    return distances
    # https://github.com/gallantlab/pycortex/blob/13054a2dc1a773b436b0cbfdd562a8f25154f20b/cortex/polyutils/exact_geodesic.py

# ---------------------------------------------------------------------------------------------------------------------#
#                                                 Distance Vectors
# ---------------------------------------------------------------------------------------------------------------------#

def vertex_centroid_distance(v):
    c = np.mean(v, axis=0, keepdims=True)
    cd = last_axis_2norm(v - c)
    return cd, c


def vertex_dist(v, f, src_vi, cls='euclidean_graph'):
    src_vi = np.asanyarray(src_vi)
    if cls == 'euclidean_graph':
        # Fast approximation to the true geodesic distances
        D = euclidean_graph_vertex_distance(v, f, src_vi)
    elif cls == 'euclidean_cloud':
        D = pdist(np.atleast_2d(v[src_vi, :]), v, 2)  # Handles singleton case
    elif cls == 'graph':
        D = graph_vertex_distance(v, f, src_vi)
    elif cls == 'exact_geodesic':
        raise NotImplementedError
    elif cls == 'area_weighted_exact_geodesic':
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown class {cls}')
    return D


def euclidean_graph_vertex_distance(v, f, src_vi):
    G = vertex_adjacency(v, f, weight='euclidean')
    return shortest_path(G, directed=False, return_predecessors=False, unweighted=False, indices=src_vi)


def graph_vertex_distance(v, f, src_vi):
    G = vertex_adjacency(v, f, weight=None)
    return shortest_path(G, directed=False, return_predecessors=False, unweighted=True, indices=src_vi)


# ---------------------------------------------------------------------------------------------------------------------#
#                                              FPS - TODO - Move me
# ---------------------------------------------------------------------------------------------------------------------#

def farthest_point_sampling(v, f, n, dist_cls='euclidean_graph', src=None, skip_first=True,return_dist=False):
    """
    Note: This method may actually be used to sample more than just meshe surfaces
    :param ndarray v: mesh vertices
    :param ndarray f: mesh faces
    :param int n: The number of points to sample wth FPS
    :param str dist_cls: One of the relevant matrix distance classes as can be found in vertex_distance_matrix
    :param Union[None,int] src:  The source point index or None to allow choosing a random point as initial
    :param bool skip_first: Whether to include or not include the first (possibly random) points into the returned arr
    :return: ndarray samples: of size [nx3]
    """
    # TODO - Add in incremental methods for the geodesic computes to save complexity
    nv = v.shape[0]
    src = np.random.randint(nv) if src is None else src

    # Init:
    D = vertex_distance_matrix(v, f, dist_cls)
    samples, joint_dist_vec = [], D[:, src]

    for i in range(n):
        samples.append(v[src])
        joint_dist_vec = np.minimum(joint_dist_vec, D[:, src])
        src = np.argmax(joint_dist_vec)

    if skip_first:
        samples = samples[1:]
        samples.append(v[src])
        # Append the last compute

    if return_dist:
        return np.array(samples),joint_dist_vec
    else:
        return np.array(samples)

# ---------------------------------------------------------------------------------------------------------------------#
#                                                           Paths
# ---------------------------------------------------------------------------------------------------------------------#

def paths():
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------#
#                                                         TEST SUITE
# ---------------------------------------------------------------------------------------------------------------------#
def _centroid_tester():
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import add_mesh, add_spheres, plotter
    from cfg import TEST_MESH_HAND_PATH

    v, f = read_mesh(TEST_MESH_HAND_PATH)
    cd, c = vertex_centroid_distance(v)
    p = plotter()
    add_mesh(p, v, f, clr=cd, strategy='mesh', lighting=True, smooth_shade_on=False)
    add_spheres(p, v=c, sphere_size=2, sphere_clr='w')
    p.show()
    pass

def _geodesic_world_tester():
    import pyvista as pv
    from pyvista import examples
    from geom.mesh.vis.base import plotter
    from geom.mesh.io.base import pyvista_mesh2numpy
    # Actually uses Dijkstra...
    # Load a global topography surface and decimate it
    land = examples.download_topo_land().triangulate().decimate(0.98)
    cape_town = land.find_closest_point((0.790801, 0.264598, -0.551942))
    dubai = land.find_closest_point((0.512642, 0.745898, 0.425255))
    bangkok = land.find_closest_point((-0.177077, 0.955419, 0.236273))
    rome = land.find_closest_point((0.718047, 0.163038, 0.676684))
    # pyvista_mesh2numpy(P)
    a = land.geodesic(cape_town, dubai)
    b = land.geodesic(cape_town, bangkok)
    c = land.geodesic(cape_town, rome)
    p = plotter()
    p.add_mesh(a + b + c, line_width=10, color="cyan", label="Geodesic Path")
    p.add_mesh(land, show_edges=True,color='tan')
    p.add_legend()
    p.camera_position = [(3.5839785524183934, 2.3915238111304924, 1.3993738227478327),
                         (-0.06842917033182638, 0.15467201157962263, -0.07331693636555875),
                         (-0.34851770951584765, -0.04724188391065845, 0.9361108965066047)]

    p.show()
    # distance = land.geodesic_distance(cape_town, rome)
    # distance

def _fps_tester():
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import add_mesh, add_spheres, plotter
    from geom.mesh.op.cpu.remesh import remove_vertices
    from cfg import TEST_MESH_HAND_PATH,TEST_MESH_HUMAN_PATH

    v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    v_new,dist = farthest_point_sampling(v, f, n=500, skip_first=True,return_dist=True)
    p = plotter()
    v,f = remove_vertices(v,f,dist < np.mean(dist))
    # Approximation to the Voronoi Cells - but it is looks mostly equivalent to a uniform sampling over
    # the vertices..
    add_mesh(p, v, f,lighting=True,strategy='mesh')
    # add_spheres(p, v_new)
    p.show()
    pass


def _nn_tester():
    from geom.mesh.io.base import read_mesh
    from cfg import TEST_MESH_HAND_PATH, TEST_MESH_HUMAN_PATH
    from util.time import timer
    from geom.mesh.op.cpu.metric import nearest_neighbor
    v1, f1 = read_mesh(TEST_MESH_HAND_PATH)
    v2, f2 = read_mesh(TEST_MESH_HUMAN_PATH)
    for s in {'euclidean_graph', 'euclidean_cloud'}:
        D = vertex_dist(v1, f1, [5, 10], cls=s)
    # with timer():
    #     d, i = nearest_neighbor(v1, v1, k=10)
    # with timer():
    #     D = vertex_distance_matrix(v1, f1, cls='euclidean_cloud', knn_truncation=10)
    # pass


if __name__ == '__main__':
    _geodesic_world_tester()
