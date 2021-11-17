import numpy as np
from util.fs import align_file_extension, file_extension
from plyfile import PlyData
import pyvista as pv
import meshio
from itertools import chain
import pickle as pk


# ---------------------------------------------------------------------------------------------------------------------#
#                                                Collectors
# ---------------------------------------------------------------------------------------------------------------------#
def supported_mesh_formats():
    return '*.off', '*.obj', '*.stl', '*.off'


def full_mesh_read(fp):
    """
    :param pathlib.Path or str fp: The file path
    :return: A meshio Mesh object with all needed fields
    """
    return meshio.read(fp)


def read_mesh(fp, verts_only=False):
    ext = file_extension(fp)
    # These are faster than meshio, and achieve the same task
    if ext == 'off':
        return read_off_verts(fp) if verts_only else read_off(fp)
    elif ext == 'ply':
        return read_ply_verts(fp) if verts_only else read_ply(fp)
    elif ext == 'obj':
        return read_obj_verts(fp) if verts_only else read_obj(fp)
    elif ext == 'npy':
        return read_npy_verts(fp) if verts_only else read_npy(fp)
    else:
        mesh = meshio.read(fp)
        return mesh.points if verts_only else mesh.points, mesh.cells


def full_mesh_write(fp, v, f, **kwargs):
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    # cell_data=cell_data,
    # field_data=field_data
    meshio.Mesh(points=v, cells=[("triangle", f)], **kwargs).write(fp)


def mesh_paths_from_dir(dp):
    patterns = supported_mesh_formats()
    return list(chain.from_iterable(dp.glob(pattern) for pattern in patterns))


# ---------------------------------------------------------------------------------------------------------------------#
#                                                     Quick READ
# ---------------------------------------------------------------------------------------------------------------------#
def read_obj_verts(fp):
    v = []
    try:
        with open(fp, 'r') as obj:
            for line in obj:
                elements = line.split()
                if elements[0] == 'v':  # Assuming vertices are first in the file
                    v.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif elements[0] == 'f':
                    continue  # Instead of break - Sometimes multiple meshes are appended on the file...
        return np.array(v)
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_obj(fp):
    v, f = [], []
    try:
        with open(fp, 'r') as obj:
            for line in obj:
                elements = line.split()
                if elements[0] == 'v':
                    v.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif elements[0] == 'f':
                    f.append(
                        [int(elements[1].split('/')[0]), int(elements[2].split('/')[0]),
                         int(elements[3].split('/')[0])])
        return np.array(v), np.array(f) - 1
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_off_verts(fp):
    v = []
    try:
        # TODO: yiftach change this back
        with open(fp,
                  "r") as fh:
            first = fh.readline().strip()
            if first != "OFF" and first != "COFF":
                raise (Exception(f"Could not find OFF header for file: {fp}"))

            # fast forward to the next significant line
            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            # <number of vertices> <number of faces> <number of edges>
            params = line.split()
            if len(params) < 2:
                raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            for i in range(int(params[0])):
                line = line.split()
                v.append([float(line[0]), float(line[1]), float(line[2])])
                line = fh.readline()

        return np.array(v)
    except Exception as e:

        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_off(fp):
    v, f = [], []
    try:
        with open(fp, "r") as fh:
            first = fh.readline().strip()
            if first != "OFF" and first != "COFF":
                raise (Exception(f"Could not find OFF header for file: {fp}"))

            # fast forward to the next significant line
            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            # <number of vertices> <number of faces> <number of edges>
            parameters = line.split()
            if len(parameters) < 2:
                raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            for i in range(int(parameters[0])):
                line = line.split()
                v.append([float(line[0]), float(line[1]), float(line[2])])
                line = fh.readline()

            for i in range(int(parameters[1])):
                line = line.split()
                f.append([int(line[1]), int(line[2]), int(line[3])])
                line = fh.readline()

        return np.array(v), np.array(f)
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


# TODO: Add PKL Reader for meshes
# def read_pkl(pickle_file):
#   with open(pickle_file, 'rb') as p:


def read_ply_verts(fp):
    # TODO - consider remove PlyData from dependencies
    try:
        with open(fp, 'rb') as f:
            plydata = PlyData.read(f)
        return np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']))
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_ply(fp):
    # TODO - consider remove PlyData from dependencies
    try:
        with open(fp, 'rb') as f:
            plydata = PlyData.read(f)
        v = np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']))
        f = np.stack(plydata['face']['vertex_indices'])
        return v, f
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_npy_verts(fp):
    return np.load(fp)


def read_npy(fp):
    pass


# ---------------------------------------------------------------------------------------------------------------------#
#                                                     Quick WRITE
# ---------------------------------------------------------------------------------------------------------------------#

def write_off(fp, v, f=None):
    fp = align_file_extension(fp, 'off')
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'OFF\n{len(str_v)} {len(str_f)} 0\n{"".join(str_v)}{"".join(str_f)}')


def write_obj(fp, v, f=None):
    fp = align_file_extension(fp, 'obj')
    str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        # Faces are 1-based, not 0-based in obj files
        str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f + 1]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


def write_ply(fp, v, f=None):
    fp = align_file_extension(fp, 'ply')
    with open(fp, 'w') as meshfile:
        meshfile.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n")
        if f is not None:
            meshfile.write(f"element face {len(f)}\nproperty list uchar int vertex_index\n")
        meshfile.write("end_header\n")

        str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
        if f is not None:
            str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
        else:
            str_f = []
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


def write_collada(fp, v, f, mesh_name="exported_mesh"):
    # TODO - Makes sure this works
    # TODO - Insert to collad
    import collada

    mesh = collada.Collada()

    vert_src = collada.source.FloatSource("verts-array", v, ('X', 'Y', 'Z'))
    geom = collada.geometry.Geometry(mesh, "geometry0", mesh_name, [vert_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#verts-array")

    triset = geom.createTriangleSet(np.copy(f), input_list, "")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    geomnode = collada.scene.GeometryNode(geom, [])
    node = collada.scene.Node(mesh_name, children=[geomnode])

    myscene = collada.scene.Scene("mcubes_scene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene

    mesh.write(fp)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Oddities
# ---------------------------------------------------------------------------------------------------------------------#

def read_npz_mask(fp):
    try:
        # TODO: yiftach change this back
        return np.load(
            fp)[
            "mask"]
    except Exception as e:
        raise OSError(f"Could not read or open npz file {fp}") from e


def numpy2open3d_cloud(v):
    import open3d as o3d
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))


def numpy2open3d_mesh(v, f):
    import open3d as o3d
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                     o3d.utility.Vector3iVector(f))


def open3d_cloud2numpy(P):
    return np.asarray(P.points)


def open3d_mesh2numpy(M):
    return np.asarray(M.vertices), np.asarray(M.triangles)


def pyvista_cloud2numpy(P):
    return P.points


def numpy2pyvista_cloud(v):
    return pv.PolyData(v)


def numpy2pyvista_mesh(v, f):
    return pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))


def pyvista_mesh2numpy(M):
    return M.points, M.faces.reshape(-1, 4)[:, 1:]


def trimesh2numpy(M):
    pass


def numpy2trimesh(v, f):
    import trimesh
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                   Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

def _fbx_tester():
    p = r'C:\Users\idoim\Desktop\681.fbx'
    v, f = read_mesh(p)


def _io_tester():
    from cfg import TEST_MESH_HUMAN_PATH
    from geom.mesh.vis.base import plot_mesh
    v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    plot_mesh(v, f)
    full_mesh_write('a.off', v, f)
    full_mesh_write('a.ply', v, f)
    full_mesh_write('a.obj', v, f)

    v2, f2 = read_mesh('a.off')
    assert np.array_equal(v2, v)
    assert np.array_equal(f2, f)

    v2, f2 = read_mesh(TEST_MESH_HUMAN_PATH)
    assert np.array_equal(v2, v)
    assert np.array_equal(f2, f)
    v2, f2 = read_mesh('a.ply')
    assert np.array_equal(v2, v)
    assert np.array_equal(f2, f)
    v2, f2 = read_mesh('a.obj')
    assert np.array_equal(v2, v)
    assert np.array_equal(f2, f)

    v2 = read_off_verts('a.off')
    assert np.array_equal(v2, v)

    v2 = read_mesh(TEST_MESH_HUMAN_PATH, verts_only=True)
    assert np.array_equal(v2, v)
    v2 = read_ply_verts('a.ply')
    assert np.array_equal(v2, v)
    v2 = read_obj_verts('a.obj')
    assert np.array_equal(v2, v)


if __name__ == '__main__':
    _fbx_tester()
