import math

import numpy as np

from base_plot import plotter, add_mesh, hybrid_kwarg_index
from util.container import is_number


def plot_mesh_montage(vs, fs=None, ns=None, lines=None, colors='w',  # Input
                      style='surface', smooth_shading=True, eye_dome=False, depth_peeling=False, lighting=None,
                      # Global arg
                      camera_pos=None,
                      normal_color='lightblue', edge_color='darkblue', line_color='darkblue', cmap='rainbow',
                      show_edges=False, clim=None, show_scalar_bar=False,  # Color options
                      normal_scale=1, point_size=6, line_width=1,  # Scales
                      grid_on=False, opacity=1.0, titles=None, axes_on=False,  # Misc
                      auto_close=True, link_plots=True, ext_func=None, max_plots_per_screen=16, ret_meshes=False,
                      unit_sphere_on=False
                      # Singletons
                      ):
    total_meshes = len(vs)  # Works for both tensors, nd.arrays and lists. We need some way to determine this number
    if hasattr(vs, 'shape') and vs.ndim == 3 and vs.shape[2] > 3:
        vs = vs[:, :, :3]  # Support for extended torch tensors
    assert total_meshes > 0, "Found no vertices"

    running_i = 0
    remaining_meshes, meshes = total_meshes, []
    while remaining_meshes > 0:
        curr_n_meshes = min(remaining_meshes, max_plots_per_screen)

        # Figure out size:
        n_rows = math.floor(math.sqrt(curr_n_meshes))
        n_cols = math.ceil(curr_n_meshes / n_rows)
        shape = (n_rows, n_cols)
        p = plotter(shape=shape)
        r, c = np.unravel_index(range(curr_n_meshes), shape)
        for i in range(curr_n_meshes):

            p.subplot(r[i], c[i])
            sub_screen_index = running_i + i
            # TODO - this decompose function is dangerous. Corner case: n_meshes == true length of
            #  single obj - is there a better solution? (problems: n_v,n_f,n_lines,camera_pos == n_meshes. Very uncommon
            #  besides camera pos - quick fix for now. Also problem with clim?
            if camera_pos is None:
                camera_pos_i = camera_pos
            elif len(camera_pos) == 3:
                if total_meshes == 3 and not is_number(camera_pos[0][0]):
                    camera_pos_i, link_plots = camera_pos[sub_screen_index], False
                    # Different camera positions may not be linked
                else:
                    camera_pos_i = camera_pos
            else:
                camera_pos_i, link_plots = camera_pos[sub_screen_index], False

            if clim is None:
                clim_i = clim
            elif len(clim) == 2:
                if total_meshes == 2 and not is_number(clim[0]):
                    clim_i = clim[i]
                else:
                    clim_i = clim
            else:
                clim_i = clim[i]

            p = add_mesh(p, **hybrid_kwarg_index(sub_screen_index, total_meshes, v=vs, f=fs, n=ns, lines=lines,
                                                 style=style, smooth_shading=smooth_shading, eye_dome=eye_dome,
                                                 depth_peeling=depth_peeling,
                                                 lighting=lighting,
                                                 # camera_pos=camera_pos,
                                                 color=colors, normal_color=normal_color, edge_color=edge_color,
                                                 line_color=line_color,
                                                 cmap=cmap, show_scalar_bar=show_scalar_bar, show_edges=show_edges,
                                                 normal_scale=normal_scale, point_size=point_size,
                                                 line_width=line_width, grid_on=grid_on, opacity=1, title=titles,
                                                 axes_on=axes_on, unit_sphere_on=unit_sphere_on),
                         camera_pos=camera_pos_i, clim=clim_i)
            if ret_meshes:
                meshes.append(p.mesh)
            if ext_func is not None:
                ext_func(p, i)  # TODO - wrong index?

        if link_plots:
            p.link_views()
        p.show(screenshot="aa.png",auto_close=auto_close, full_screen=True)  # interactive_update=not auto_close, interactive=auto_close,
        running_i += curr_n_meshes
        remaining_meshes -= curr_n_meshes
    if ret_meshes:
        return p.image, meshes
    return p.image
