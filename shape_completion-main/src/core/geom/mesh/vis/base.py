import math

import numpy as np
import pyvista as pv
import torch
from pyvista.plotting.theme import parse_color

from geom.matrix.cpu import last_axis_2norm
from geom.mesh.io.base import read_mesh
from geom.mesh.op.cpu.base import face_barycenters
from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask
from geom.mesh.op.cpu.spectra import tangent_projection
from geom.mesh.synth.mesh_zoo_3d import uniform_grid

pv.set_plot_theme("document")


# ---------------------------------------------------------------------------------------------------------------------#
#                                           General Utility Functions
# ---------------------------------------------------------------------------------------------------------------------#


def plotter():
    return pv.Plotter()


def uniclr(N, clr='w', inds=None, clr2='r'):
    rgb = np.tile(parse_color(clr), (N, 1))
    if inds is not None:
        rgb[inds, :] = np.tile(parse_color(clr2), (len(inds), 1))
    return rgb


def prepare_completion_montage(sample, keys=('gt', 'tp', 'gt_part'), with_normals=False):
    vb, fb, nb, labelb = [], [], [], []
    with_normals = with_normals and (sample['gt'].shape[2] > 3)  # normal exists
    faces_exist = 'gt_f' in sample.keys() or 'tp_f' in sample.keys()
    for k in keys:
        b = sample[k]
        for i in range(b.shape[0]):  # num batches
            labelb.append(f'{k}_{i}')
            vb.append(b[i, :, :3])
            if with_normals:
                nb.append(b[i, :, 3:6])
            if faces_exist:
                if k == 'gt_part':
                    fb.append(trunc_to_vertex_mask(b[i, :, :3], sample[f'gt_f'][i], sample['gt_mask'][i])[1])
                else:
                    fb.append(sample[f'{k}_f'][i])

    if not with_normals:
        nb = None
    if not faces_exist:
        fb = None
    return {'vb': vb, 'nb': nb, 'fb': fb, 'labelb': labelb}


# ---------------------------------------------------------------------------------------------------------------------#
#                                            Visualization Functions
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyIncorrectDocstring
def plot_mesh(v, f=None, n=None, strategy='spheres', grid_on=False, clr='lightcoral', normal_clr='lightblue',
              label=None, smooth_shade_on=True, show_edges=False, clr_map='rainbow', normal_scale=1, point_size=None,
              lighting=None, camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0)), opacity=1.0):
    """
    :param v: tensor - A numpy or torch [nv x 3] vertex tensor
    :param f: tensor |  None - (optional) A numpy or torch [nf x 3] vertex tensor OR None
    :param n: tensor |  None - (optional) A numpy or torch [nf x 3] or [nv x3] vertex or face normals. Must input f
    when inputting a face-normal tensor
    :param strategy: One of ['spheres','cloud','mesh']
    :param grid_on: bool - Plots an xyz grid with the mesh. Default is False
    :param clr: str or [R,G,B] float list or tensor - Plots  mesh with color clr. clr = v is cool
    :param normal_clr: str or [R,G,B] float list or tensor - Plots  mesh normals with color normal_clr
    :param label: str - (optional) - When inputted, displays a legend with the title label
    :param smooth_shade_on: bool - Plot a smooth version of the facets - just like 3D-Viewer
    :param show_edges: bool - Show edges in black. Only applicable for strategy == 'mesh'
    For color list, see pyvista.plotting.colors
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    # White background
    p = pv.Plotter()
    p, m = add_mesh(p, v=v, f=f, n=n, grid_on=grid_on, strategy=strategy, clr=clr, normal_clr=normal_clr,
                    label=label, smooth_shade_on=smooth_shade_on, show_edges=show_edges, cmap=clr_map,
                    normal_scale=normal_scale, point_size=point_size, lighting=lighting, camera_pos=camera_pos,
                    opacity=opacity)
    p.show()
    return p, m


# noinspection PyIncorrectDocstring
def plot_mesh_montage(vb, fb=None, nb=None, strategy='spheres', labelb=None, grid_on=False, clrb='lightcoral',
                      normal_clr='lightblue', smooth_shade_on=True, show_edges=False, normal_scale=1, auto_close=True,
                      camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1, 0)), lighting=None,link_plots=True,ext_func=None,
                      opacity=1):
    """
    :param vb: tensor | list - [b x nv x 3] batch of meshes or list of length b with tensors [nvx3]
    :param fb: tensor | list | None - (optional) [b x nf x 3]
    batch of face indices OR a list of length b with tensors [nfx3]
    OR a [nf x 3] in the case of a uniform face array for all meshes
    :param nb: tensor | list | None - (optional) [b x nf|nv x 3]  batch of normals. See above
    :param clrb: list of color options or a single color option
    :param labelb: list of titles for each mesh, or None
    * For other arguments, see plot_mesh
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    if hasattr(vb, 'shape'):  # Torch Tensor
        n_meshes = tuple(vb.shape)[0]
        vb = vb[:, :, :3]  # Truncate possible normals
    else:
        n_meshes = len(vb)
    assert n_meshes > 0
    n_rows = math.floor(math.sqrt(n_meshes))
    n_cols = math.ceil(n_meshes / n_rows)

    shape = (n_rows, n_cols)
    p = pv.Plotter(shape=shape)
    r, c = np.unravel_index(range(n_meshes), shape)
    ms = []
    for i in range(n_meshes):
        f = fb if fb is None or (hasattr(fb, 'shape') and fb.ndim == 2) else fb[i]
        if isinstance(clrb, list):
            clr = clrb[i]
        elif isinstance(clrb, np.ndarray):
            if clrb.ndim == 3:
                clr = clrb[:, :, i]
            else:
                clr = clrb
        else:
            clr = clrb
            # Uniform faces support. fb[i] is equiv to fb[i,:,:]
        n = nb if nb is None else nb[i]
        label = labelb if labelb is None else labelb[i]
        p.subplot(r[i], c[i])
        _, m = add_mesh(p, v=vb[i], f=f, n=n, strategy=strategy, label=label, grid_on=grid_on,
                        normal_scale=normal_scale, camera_pos=camera_pos,
                        clr=clr, normal_clr=normal_clr, smooth_shade_on=smooth_shade_on, show_edges=show_edges,
                        lighting=lighting,opacity=opacity)
        if ext_func is not None:
            ext_func(p,m,i)
        # add_spheres(p,vb[i][1,:][None,:])
        ms.append(m)

    if link_plots:
        p.link_views()
    p.show(auto_close=auto_close, interactive_update=not auto_close, interactive=auto_close, full_screen=True)
    return p, ms


def plot_skeleton(v, edges, tube_radius=0.1, point_size=1,
                  sphere_color=np.array([0, 1, 0]), tube_color=np.array([0, 80, 250]) / 255, transformations=None):
    p = pv.Plotter()
    p, m = add_skeleton(p=p, v=v, edges=edges, tube_radius=tube_radius, point_size=point_size,
                        sphere_color=sphere_color,
                        tube_color=tube_color, transformations=transformations)
    p.show()
    return p, m


def plot_projected_vectorfield(v, f, vf, normalize=True, **kwargs):
    p = pv.Plotter()
    add_vectorfield_tangent_projection(p=p, v=v, f=f, vf=vf, normalize=normalize, **kwargs)
    p.show()


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Helper Functions
# ---------------------------------------------------------------------------------------------------------------------#
def to_pyvista_edges(edges):
    return np.concatenate((np.full((edges.shape[0], 1), 2), edges), 1)

def add_vectorfield_tangent_projection(p, v, f, vf, normalize=True, **kwargs):
    if vf.shape[0] == 3 * f.shape[0]:
        vf = np.reshape(vf, (3, f.shape[0])).T
    vfp, clr = tangent_projection(v, f, vf, normalize)
    M = add_mesh(p, v=v, f=f, n=vfp, clr=clr, strategy='mesh', **kwargs)
    return p, M


def add_skeleton(p, v, edges: np.ndarray, transformations=None, scale=1, plot_points=True, tube_radius=0.01, point_size=1,
                 sphere_color='w', tube_color=np.array([0, 80, 250]) / 255, **kwargs):
    """
    :param p:
    :param v:
    :param plot_points:
    :param transformations:
    :param scale: determines the size of the coordinate system
    :param edges:
    :param float tube_radius: r <0 renders only spheres. r==0 renders lines, and r>0 renders as tubes of radius r
    :param point_size: The sphere size, by units of VTK
    :param [str,nd.arrray] sphere_color: The color of the spheres - either in string or rgb format
    :param [str,nd.arrray] tube_color: The color of the tubes - either in string or rgb format
    :return: The plotter object
    """
    # Construct a PolyData object with vertices + lines filled
    if edges.shape[1] == 2: # Doesn't have the 2-cell qualifier
        edges = to_pyvista_edges(edges)

    M = pv.PolyData(v)
    M.lines = edges

    if tube_radius > 0:
        M = M.tube(radius=tube_radius)
    if tube_radius >= 0:
        p.add_mesh(M, smooth_shading=True, color=tube_color, lighting=True)

    if plot_points:
        p, _ = add_spheres(p, v, sphere_clr=sphere_color, sphere_size=point_size, **kwargs)
    if transformations is not None:
        p = add_joint_coordinate_system(p, transformations, scale=scale)
    return p, M


def add_spheres(p, v, sphere_clr='black', sphere_size=1, **kwargs):
    return add_mesh(p, v=v, smooth_shade_on=True, clr=sphere_clr, strategy='spheres', point_size=sphere_size * 20,
                    lighting=True, **kwargs)


def add_integer_vertex_labels(p, v, font_size=10):
    M = pv.PolyData(v)
    # TODO - show_points doesn't seem to work
    p.add_point_labels(M, [str(i) for i in range(v.shape[0])], font_size=font_size, show_points=False,
                       shape=None, render_points_as_spheres=False, point_size=-1)
    p.add_floor()
    return p


def add_isocontours(p, v, f, scalar_func, isosurfaces=15, color='black', line_width=3, opacity=0.5):
    pnt_cloud = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    pnt_cloud._add_point_array(scalars=scalar_func, name='Func')
    contours = pnt_cloud.contour(isosurfaces=isosurfaces, scalars='Func')
    m = p.add_mesh(contours, color=color, line_width=line_width, smooth_shading=True, opacity=opacity,
                   render_lines_as_tubes=True)
    return p, m


def add_joint_coordinate_system(p, trans_mats, scale=1):
    # if not isinstance(trans_mats,(list,tuple)):
    #     trans_mats = [trans_mats]
    for joint_transformation in trans_mats:
        vec_start = joint_transformation[0:3, 3]  # Translation Vector
        x_direction = joint_transformation[0:3, 0:3] @ [1, 0.0, 0.0]
        y_direction = joint_transformation[0:3, 0:3] @ [0.0, 1, 0.0]
        z_direction = joint_transformation[0:3, 0:3] @ [0.0, 0.0, 1]
        x_vec = pv.Arrow(start=vec_start, direction=x_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)
        y_vec = pv.Arrow(start=vec_start, direction=y_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)
        z_vec = pv.Arrow(start=vec_start, direction=z_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)

        p.add_mesh(x_vec, color='red')  # R
        p.add_mesh(y_vec, color='green')  # G
        p.add_mesh(z_vec, color='blue')  # B
    return p


def add_floor_grid(p, nx=30, ny=30, x_bounds=(-3, 3), y_bounds=(-3, 3), z_val=0, plot_points=False,
                   grid_color='wheat', with_axis=False, **kwargs):
    v, edges = uniform_grid(nx=nx, ny=ny, x_bounds=x_bounds, y_bounds=y_bounds, z_val=z_val, cell_type='edges')

    if with_axis:
        axis = [np.array([[1, 0, 0, x_bounds[0]], [0, 1, 0, y_bounds[0]], [0, 0, 1, z_val + 0.1], [0, 0, 0, 1]])]
    else:
        axis = None
    return add_skeleton(p, v=v, edges=edges, plot_points=plot_points, tube_color=grid_color, transformations=axis,
                        tube_radius=0.02, **kwargs)


def add_vectorfield(p, v, f, vf, clr='lightblue', normal_scale=1, colormap='rainbow'):
    # TODO - Support explicit supply of color per arrow - need to extend clr vector on to arrow mesh(with np.tile)

    # Align flat vector fields:
    if f is not None and vf.shape[0] == 3 * f.shape[0]:
        vf = np.reshape(vf, (3, f.shape[0])).T
    elif vf.shape[0] == 3 * v.shape[0]:
        vf = np.reshape(vf, (3, v.shape[0])).T

    # Prepare the PolyData object:
    if f is not None and vf.shape[0] == f.shape[0]:
        pnt_cloud = pv.PolyData(face_barycenters(v, f))
    else:
        assert vf.shape[0] == v.shape[0]
        pnt_cloud = pv.PolyData(v)

    pnt_cloud['glyph_scale'] = last_axis_2norm(vf)
    pnt_cloud['vectors'] = vf
    arrows = pnt_cloud.glyph(orient="vectors", scale="glyph_scale", factor=0.05 * normal_scale)
    p.add_mesh(arrows, color=clr, colormap=colormap)
    return p, arrows


def add_mesh(p, v, f=None, n=None, strategy='spheres', grid_on=False, clr='lightcoral',
             normal_clr='lightblue', label=None, smooth_shade_on=True, show_edges=False, cmap='rainbow',
             normal_scale=1, camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0)), lines=None, opacity=1.0,
             point_size=None, lighting=None, eye_dome=False):
    # TODO - Clean this shit function up
    # Align arrays:
    v = v.numpy() if torch.is_tensor(v) else v
    f = f.numpy() if torch.is_tensor(f) else f
    n = n.numpy() if torch.is_tensor(n) else n
    clr = clr.numpy() if torch.is_tensor(clr) else clr
    normal_clr = normal_clr.numpy() if torch.is_tensor(normal_clr) else normal_clr

    # Align strategy
    style = 'surface'
    if strategy == 'mesh' or strategy == 'wireframe':
        assert f is not None, "Must supply faces for mesh strategy"
        if strategy == 'wireframe':
            style = strategy
    else:
        f = None  # Destroy the face information
    spheres_on = (strategy == 'spheres')

    # Create Data object:
    if f is not None:
        # Adjust f to the needed format
        pnt_cloud = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    else:
        pnt_cloud = pv.PolyData(v)

    if lines is not None:
        pnt_cloud.lines = lines

    # Default size for spheres & pnt clouds
    if point_size is None:
        point_size = 12.0 if spheres_on else 2.0  # TODO - Dynamic computation of this, based on mesh volume

    # Handle difference between color and scalars, to support RGB tensor
    if isinstance(clr, str) or len(clr) == 3:
        scalars = np.tile(parse_color(clr), (v.shape[0], 1))
        clr_str = clr
        rgb = True
    else:
        clr_str = 'w'
        scalars = clr
        rgb = isinstance(clr, (np.ndarray, np.generic)) and clr.squeeze().ndim == 2  # RGB Vector
    # TODO - use a kwargs approach to solve messiness
    if lighting is None:
        # Default Pyvista Light
        d_light = {'ambient': 0.0, 'diffuse': 1.0, 'specular': 0.0, 'specular_power': 100.0}
    else:
        # Our Default Lighting
        d_light = {'ambient': 0.3, 'diffuse': 0.6, 'specular': 1, 'specular_power': 20}
        # Add the meshes to the plotter:
    p.add_mesh(pnt_cloud, style=style, smooth_shading=smooth_shade_on, scalars=scalars, cmap=cmap,
               show_edges=show_edges,  # For full mesh visuals - ignored on point cloud plots
               render_points_as_spheres=spheres_on, point_size=point_size,
               rgb=rgb, opacity=opacity, lighting=lighting, **d_light)  # For sphere visuals - ignored on full mesh
    # ambient/diffuse/specular strength, specular exponent, and specular color
    #     'Shiny',	0.3,	0.6,	0.9,	20,		1.0
    #     'Dull',		0.3,	0.8,	0.0,	10,		1.0
    #     'Metal',	0.3,	0.3,	1.0,	25,		.5
    if camera_pos is not None:
        p.camera_position = camera_pos
    if n is not None:  # Face normals or vertex normals
        add_vectorfield(p, v, f, n, clr=normal_clr, normal_scale=normal_scale)

    # Book-keeping:
    if label is not None and label:
        siz = 0.25
        p.add_legend(labels=[(label, clr_str)], size=[siz, siz / 2])
    if grid_on:
        p.show_grid()

    if eye_dome:
        p.enable_eye_dome_lighting()
    return p, pnt_cloud


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

def _visuals_tester():
    from data.sets import DatasetMenu
    from data.transforms import Center

    ds = DatasetMenu.order('MixamoSkinnedGaon9Proj')
    samp = ds.sample(2, transforms=[Center()], method='rand_f2p')
    plot_mesh_montage(**prepare_completion_montage(samp, with_normals=True),
                      strategy='spheres', clrb='lightblue', grid_on=True)


def _spheres_tester():
    from geom.matrix.cpu import spherical2caresian
    pv.set_plot_theme('dark')

    n = 50000
    mesh = pv.PolyData(spherical2caresian(np.random.random((n, 3)) * 1000))
    mesh["radius"] = 10 * np.random.rand(n)

    # Low resolution geometry
    geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    # geom = pv.Disc()
    # Progress bar is a new feature on master branch
    glyphed = mesh.glyph(scale="radius", geom=geom, )  # progress_bar=True)

    p = pv.Plotter(notebook=False)
    p.add_mesh(glyphed, color='white')
    p.show()


def _denoising_plot_tester():
    noisy_gt = read_mesh(r'C:\Users\idoim\Desktop\1\gt_50025_hips_483_tp_50025_punching_179_gtnoise.obj')
    tp = read_mesh(r'C:\Users\idoim\Desktop\1\gt_50025_hips_483_tp_50025_punching_179_tp.ply')
    res = read_mesh(r'C:\Users\idoim\Desktop\1\gt_50025_hips_483_tp_50025_punching_179_res.obj')
    gt = read_mesh(r'C:\Users\idoim\Desktop\1\gt_50025_hips_483_tp_50025_punching_179_gt.obj')
    p = pv.Plotter()

    add_mesh(p, *res, clr=(178, 254, 255), strategy='spheres', point_size=10)
    add_mesh(p, *gt, clr='cyan', opacity=0.2, strategy='mesh', smooth_shade_on=True)
    add_mesh(p, *res, clr='pink', opacity=0.2, strategy='mesh', smooth_shade_on=True)
    add_mesh(p, *gt, clr='pink', opacity=0.5, strategy='spheres', point_size=10)
    p.show()


def _simple_plot_tester():
    from cfg import TEST_MESH_HUMAN_PATH
    pv.set_plot_theme('dark')
    p = plotter()
    v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    add_mesh(p, v, f, strategy='mesh', clr='aqua', eye_dome=True)
    # add_integer_vertex_labels(p, v)
    p.show()


"""
 Plot Menu Controls:
 
    q                               Close the rendering window
    v                               Isometric camera view
    w                               Switch all datasets to a wireframe representation
    r                               Reset the camera to view all datasets
    s                               Switch all datasets to a surface representation
    shift+click or middle-click     Pan the rendering scene
    left-click                      Rotate the rendering scene in 3D
    ctrl+click                      Rotate the rendering scene in 2D (view-plane)
    mouse-wheel or right-click      Continuously zoom the rendering scene
    shift+s                         Save a screenhsot (only on BackgroundPlotter)
    shift+c                         Enable interactive cell selection/picking
    up/down                         Zoom in and out
    +/-                             Increase/decrease the point size and line widths
"""
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    _simple_plot_tester()
