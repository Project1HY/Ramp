from itertools import chain
from pathlib import Path

import numpy as np
import pyvista as pv

from geom.matrix.transformations import rotation_matrix
from geom.mesh.io.base import read_mesh, supported_mesh_formats, mesh_paths_from_dir
from geom.mesh.vis.base import plot_mesh_montage, add_mesh
from util.strings import warn
from util.time import progress, busy_wait


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def multianimate_from_path(dp: Path, n: int, prefetch=True, **kwargs):
    """
    This function animations a multitude of sequences, in a manner corresponding to the default parameters of
    util.mesh.plots.add_mesh with any additional changes supplied in the **kwargs in this function's arguments.
    The code presumes that only vertices are changed from frame to frame.

    :param dp: A filepath to a directory of sequences. Each "sequence" is defined as a directory with a name relating
    to the actual mesh sequence (e.g. 'chicken_wings') with multiple frames inside it,
    named in alphabetical order, such as 000001.OFF,000002.OFF and so forth. Supported mesh exchange are detailed in
    util.mesh.exchange.read_mesh
    :param n: The number of sequences to take from fp. The first n sequences by alphabetical order will be shown on
    screen. If n>number of sequences in fp, a warning will be shown, and the minimum of the two taken as the effective n
    :param prefetch: Whether to load all frames into memory. For very large sequences, this is not feasible, memory wise
    Prefetching substantially increases the animation framerate.
    :param kwargs: Any visualization changes required. See util.mesh.plots.add_mesh for details
    :return: None
    """
    seqs_dps = list(dp.glob('*'))
    if len(seqs_dps) < n:
        warn(f'Requested a montage of N={n} windows, but supplied directory has only {len(seqs_dps)} sequences\n')
    else:
        seqs_dps = seqs_dps[:n]

    patterns = supported_mesh_formats()
    seqs_fps = [list(chain.from_iterable(dp.glob(pattern) for pattern in patterns)) for dp in seqs_dps]

    vs, fs, labels = [], [], []
    for seq_fps in seqs_fps:
        v, f = read_mesh(seq_fps[0])
        vs.append(v)
        fs.append(f)
        labels.append(seq_fps[0].parents[0].stem)
    p, ms = plot_mesh_montage(vs, fs, labelb=labels, auto_close=False, **kwargs)

    if prefetch:
        vs = [[read_mesh(mesh_fp, verts_only=True) for mesh_fp in seq_fps] for seq_fps in
              progress(seqs_fps, desc='Prefetching frames from disk - hold on')]

    longest_sequence = max(map(len, seqs_fps))
    ended = [False] * len(ms)
    for i in range(longest_sequence):  # Iterate over all frames
        for mi, m in enumerate(ms):
            if i < len(seqs_fps[mi]):
                v = vs[mi][i] if prefetch else read_mesh(seqs_fps[mi][i])
                p.update_coordinates(points=v, mesh=m, render=False)
            elif not ended[mi]:  # TODO - we can simply remove m from the list
                p.update_scalars(scalars=np.ones_like(vs[mi][0]), mesh=m, render=False)  # Color in white
                ended[mi] = True
        p.update()

    p.close()


def animate_from_path(dp, gif_name=None, **plt_args):
    mesh_paths = list(mesh_paths_from_dir(dp))
    assert len(mesh_paths) > 0, f"Did not find mesh files in {dp}"
    vs = []
    first_v, f = read_mesh(mesh_paths[0])
    vs.append(first_v)
    mesh_paths = mesh_paths[1:]
    for p in mesh_paths:
        vs.append(read_mesh(p, verts_only=True))
    animate(vs, f, gif_name, **plt_args)


def animate(vs, f=None, gif_name=None, pause=0.025, rotation_direction=(0, 1, 0), n_rotations=2,
            **plt_args):
    """
     This function animations a single sequence supplied by the list of vertex arrays vs, in a manner corresponding
     to the default parameters of util.mesh.plots.add_mesh with any additional changes supplied in the **kwargs
     in this function's arguments. The code presumes that only vertices are changed from frame to frame.

    :param vs: A list of numpy.arrays of [num_verticesx3]
    :param f: A single numpy.array of faces of [num_faces x3]
    :param gif_name: The name of the target gif. If gif_name is None, a gif will not be generated.
    :param kwargs: Any other relevant arguments to control visualization. See util.mesh.plots.add_mesh
    """
    p = pv.Plotter()
    add_mesh(p, vs[0], f, **plt_args) # Normals are not supported
    print('Orient the view, then press "q" to close window and produce movie')
    p.show(auto_close=False)
    # p.show(auto_close=False, interactive_update=True, interactive=False)

    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    if n_rotations > 0:
        center = vs[0].mean(axis=0).tolist()
    for i, v in enumerate(vs):
        if n_rotations >0:
            R = rotation_matrix(angle=i * 2 * np.pi * n_rotations / len(vs), direction=rotation_direction,
                                point=center)[:3, :3]
            v = v @ R
        p.update_coordinates(v)
        p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program
        if gif_name is not None:
            p.write_frame()

    p.close()  # Close movie and delete object


def animate_c3d(fp, remove_last=True, line_method=None):
    # TODO - Rewrite this function - looks like shit.
    # https://medium.com/@yvanscher/explanation-of-the-c3d-file-format-c8e065300510
    # https://alinen.github.io/MotionScriptTools/notes/MotionBuilderC3D.html
    # http://mocap.cs.cmu.edu/search.php?subjectnumber=%&motion=%  # Download more c3d here.
    import c3d
    reader = c3d.Reader(open(fp, 'rb'))
    vs, vertex_max = [], -1
    for i, v, analog in reader.read_frames():
        v = v[:, :3]
        if remove_last:
            v = v[:-1, :]  # Remove Origin - TODO - is this always like this?
        vertex_max = max(np.max(np.abs(v)), vertex_max)
        vs.append(v)

    # Normalize:
    vs = [v / vertex_max for v in vs]
    if line_method is not None:  # TODO - Fix this
        lines = line_method(vs[0]).lines
    else:
        lines = None

    animate(vs, strategy='spheres', lines=lines)  # TODO - Orient camera?


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

def _dataset_animation_tester(ds_name='DFaustProj', subject='50027', sequence='jumping_jacks', proj_id=2, dark=True):
    from data.sets import DatasetMenu, CompletionDataset
    from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask

    # Get the dataset
    ds: CompletionDataset = DatasetMenu.order(ds_name)
    print('Subject Names:', ds.subject_names())  # Subject Names
    print('Sequence Names:', tuple(ds._hit._hit[subject].keys()))  # Sequence Names
    seq_index = ds._hit._hit[subject][sequence]
    n_frames = len(seq_index)

    # Get representative faces for stability:
    mask = ds.deformation_mask_by_hi((subject, sequence, 0, proj_id))
    v, f = ds.full_shape_by_hi((subject, sequence, 0), verts_only=False)
    _, f = trunc_to_vertex_mask(v, f, mask)
    strategy = 'mesh'

    # Circular mask option - still doesn't work, needs a full update of mask
    # masks = [ds.deformation_mask_by_hi((subject, sequence, i, proj_id)) for i in range(10)]
    # f,strategy = None,'spheres'

    # Load all meshes and trucate them to mask
    vs = [ds.full_shape_by_hi((subject, sequence, i), verts_only=True)[mask, :] for i in progress(range(n_frames))]

    # Alternative - too jittery + animate() doesn't support face swap yet.
    # vs = [ds.partial_shape_by_hi((subject, sequence, i, proj_id), verts_only=True) for i in progress(range(n_frames))]
    if dark:
        pv.set_plot_theme('dark')
    animate(vs, f, strategy='mesh', clr='cyan', eye_dome=False, lighting=dark if dark else None,n_rotations=0)


def _multianimate_tester():
    from pathlib import Path
    # fp = Path(r'C:\Users\idoim\OneDrive - Technion\Ongoing\Research\DeepShape\data\synthetic\DFaustPyProj\full\50004')
    multianimate_from_path(Path(r'M:\datasets\dynamic_isometry\robert_sumner_animals'), n=8)


def _animate_tester():
    from pathlib import Path
    from geom.mesh.io.base import read_off
    fp = Path(
        r'C:\Users\idoim\OneDrive - Technion\Ongoing\Research\DeepShape\data\synthetic\DFaustPyProj\full\50002\running_on_spot')
    mesh_fps = list(fp.glob('*'))
    vs, f = [], None
    for mesh_fp in mesh_fps:
        v, f = read_off(mesh_fp)
        vs.append(v)

    animate(vs, f, strategy='spheres')


def _align_to_z_plot_tester():
    from cfg import TEST_MESH_HUMAN_PATH
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import add_floor_grid, add_mesh, add_spheres, plotter
    from geom.mesh.op.cpu.alignment import align_pointcloud_to_z_axis
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
    _dataset_animation_tester()
