import pyvista as pv
import numpy as np
import torch
import time

_DEF_PYVISTA_LIGHT_PARAMS = {'ambient': 0.0, 'diffuse': 1.0, 'specular': 0.0, 'specular_power': 100.0}
_DEF_LIGHT_PARAMS = {'ambient': 0.3, 'diffuse': 0.6, 'specular': 1, 'specular_power': 20}
_STYLE_RENAMINGS = {'sphered_wireframe': 'wireframe', 'spheres': 'points'}

_N_VERTICES_TO_CAM_POS = {  # TODO - find some better idea then this...
    6890: ((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0)),
    3978: ((0.276, 0.192, -1.72), (0.023, -0.005, 0.003), (-0.245, -0.958, -0.146)),
    988: [(-0.01627141988736187, 0.5966247704974398, 1.397480694725684),
          (0.0673985233265075, -0.1198104746242973, 0.0654804238676567),
          (-0.009245843701938337, 0.880410561042025, -0.4741220922715016)]
}

def stringify(lst):
    return [str(ele) for ele in lst]

def torch2numpy(*args):
    out = []
    for arg in args:
        arg = arg.numpy() if torch.is_tensor(arg) else arg
        out.append(arg)
    if len(args) == 1:
        return out[0]
    else:
        return tuple(out)


def busy_wait(dt):
    current_time = time.time()
    while time.time() < current_time + dt:
        pass

def concat_cell_qualifier(arr):
    # arr = np.expand_dims(arr,axis=1)
    return np.concatenate((np.full((arr.shape[0], 1), arr.shape[1]), arr), 1)

def plotter(theme='document', **kwargs):
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
        shift+s                         Save a screenshot (only on BackgroundPlotter)
        shift+c                         Enable interactive cell selection/picking
        up/down                         Zoom in and out
        +/-                             Increase/decrease the point size and line widths
    """
    pv.set_plot_theme(theme)
    p = pv.Plotter(off_screen=True, **kwargs)
    return p

def is_sequence(obj):
    """
    Check if an object is a sequence or not.
    Parameters
    -------------
    obj : object
      Any object type to be checked
    Returns
    -------------
    is_sequence : bool
        True if object is sequence
    """
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))

    # check to make sure it is not a set, string, or dictionary
    seq = seq and all(not isinstance(obj, i) for i in (dict,
                                                       set,
                                                       str))

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()

    return seq

def hybrid_kwarg_index(i, expected_first_dim, **kwargs):
    for k, v in kwargs.items():
        if is_sequence(v):
            if len(v) == expected_first_dim:
                kwargs[k] = v[i]  # TODO - color falls here if v.shape[0] == n_meshes
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == expected_first_dim \
                    and k not in ['f', 'v', 'n']:
                # TODO - this is dangerous as well - seeing expected_first_dim could n_f or n_v for example
                # support for color tensors
                kwargs[k] = v[:, i]
    return kwargs


def add_lines(p, v, lines, line_width=1, line_color='darkblue', **plt_args):
    # TODO - Merge add_lines & add_general_lines
    mesh = pv.PolyData(v)
    lines = concat_cell_qualifier(lines)
    mesh.lines = lines
    tubes = mesh.tube(radius=line_width / 1000)  # TODO - Handle automatic scaling
    p.add_mesh(tubes, smooth_shading=True, **plt_args)
    return p


def add_mesh(p, v, f=None, n=None, lines=None,  # Input
             style='surface', smooth_shading=True, eye_dome=False, depth_peeling=False, lighting=None,  # Global arg
             camera_pos=None,
             edge_color='darkblue', line_color='darkblue', cmap='rainbow',
             show_edges=False, clim=None, show_scalar_bar=False,  # Color options
             point_size=6, line_width=1,  # Scales
             grid_on=False, opacity=1.0, title=None, axes_on=False, as_a_single_mesh=False,  # Misc
             ):
    # Input Argument corrections
    assert style in ['surface', 'wireframe', 'points', 'spheres', 'sphered_wireframe', 'glyphed_spheres']
    f = None if style in ['points', 'spheres', 'glyphed_spheres'] else f
    if style in ['wireframe', 'sphered_wireframe', 'surface'] and f is None:
        style = 'spheres'  # Default fallback if we forgot the faces
    title = title if title is None else str(title)  # Allow for numeric titles
    render_points_as_spheres = True if style == 'spheres' else False
    v, f, n, lines = torch2numpy(v, f, n, lines)
    light_params = _DEF_PYVISTA_LIGHT_PARAMS if lighting is None else _DEF_LIGHT_PARAMS

    mesh = pv.PolyData(v) if f is None else pv.PolyData(v, concat_cell_qualifier(f))
    if lines is not None:  # TODO - support general lines, and not just vertex indices. Rename "lines"
        if as_a_single_mesh:  # Animate skeleton support
            style = 'surface'  # Pyvista bug - lines only plot with style == surface
            render_points_as_spheres = True
            mesh.lines = concat_cell_qualifier(lines)
        else:
            add_lines(p, v, lines=lines, line_width=line_width, line_color=line_color, cmap=cmap, opacity=opacity,
                      lighting=lighting, **light_params)

    style = _STYLE_RENAMINGS.get(style, style)  # Translation of styles for insertion to Pyvista
    # Point Size = Diameter of a point (VTK Manual), expressed in Screen Units - automatically scaled by camera
    p.add_mesh(mesh, style=style, smooth_shading=smooth_shading, cmap=cmap, show_edges=show_edges,
               point_size=point_size, render_points_as_spheres=render_points_as_spheres,
               edge_color=edge_color,
               opacity=opacity, lighting=lighting, clim=clim, line_width=line_width, render_lines_as_tubes=True,
               show_scalar_bar=show_scalar_bar, stitle='',
               **light_params)

    if camera_pos is None:
        camera_pos = _N_VERTICES_TO_CAM_POS.get(len(v), None)

    if camera_pos is not None:
        p.camera_position = camera_pos

    if title:
        p.add_text(title, font_size=12, position='upper_edge')
    if grid_on:
        p.show_bounds(**{'grid': 'back', 'location': 'all', 'ticks': 'both'})
    if eye_dome:
        p.enable_eye_dome_lighting()
    if depth_peeling:
        p.enable_depth_peeling()
    if axes_on:
        p.add_axes()
    # TODO - enable texture:
    return p
