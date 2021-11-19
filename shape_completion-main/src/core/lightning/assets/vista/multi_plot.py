import math

import numpy as np

from lightning.assets.vista.external import center_by_l2_mean,hybrid_kwarg_index
from lightning.assets.vista.basic_plot import plotter, add_mesh
from lightning.assets.vista.plot_util.container import is_number


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

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
        p.show(auto_close=auto_close, full_screen=True)  # interactive_update=not auto_close, interactive=auto_close,
        running_i += curr_n_meshes
        remaining_meshes -= curr_n_meshes
    if ret_meshes:
        return p, meshes
    return p


def plot_mesh_grid(vs, fs=None, ns=None, lines=None, spacer_x=2, spacer_y=2, colors='w',  # Input
                   style='surface', smooth_shading=True, lighting=None,
                   # eye_dome=False, depth_peeling=False,
                   # Global arg
                   camera_pos=None,
                   normal_color='lightblue', edge_color='darkblue', line_color='darkblue', cmap='rainbow',
                   show_edges=False, clim=None, show_scalar_bar=False,  # Color options
                   normal_scale=1, point_size=6, line_width=1,  # Scales
                   grid_on=False, opacity=1.0, title=None, axes_on=False,  # Misc
                   auto_close=True
                   ):
    total_meshes = len(vs)  # Works for both tensors, nd.arrays and lists. We need some way to determine this number
    if hasattr(vs, 'shape') and vs.ndim == 3 and vs.shape[2] > 3:
        vs = vs[:, :, :3]  # Support for extended torch tensors
    assert total_meshes > 0, "Found no vertices"
    n_rows = math.floor(math.sqrt(total_meshes))
    n_cols = math.ceil(total_meshes / n_rows)
    p = plotter()
    r, c = np.unravel_index(range(total_meshes), (n_rows, n_cols))

    for i in range(total_meshes):

        # TODO - get rid of this
        if clim is None:
            clim_i = clim
        elif len(clim) == 2:
            if total_meshes == 2 and not is_number(clim[0]):
                clim_i = clim[i]
            else:
                clim_i = clim
        else:
            clim_i = clim[i]

        args_i = hybrid_kwarg_index(i, total_meshes, v=vs, f=fs, n=ns, lines=lines,
                                    style=style, smooth_shading=smooth_shading, lighting=lighting,
                                    color=colors, normal_color=normal_color, edge_color=edge_color,
                                    line_color=line_color, cmap=cmap, show_edges=show_edges,
                                    normal_scale=normal_scale, point_size=point_size, show_scalar_bar=show_scalar_bar,
                                    line_width=line_width, grid_on=grid_on, opacity=opacity)
        args_i['camera_pos'] = camera_pos
        args_i['clim'] = clim_i
        # Naieve Spacing:
        args_i['v'] = center_by_l2_mean(args_i['v']) + np.array([r[i] * spacer_x, c[i] * spacer_y, 0]).reshape(1, 3)
        p = add_mesh(p, **args_i)

    if title:
        p.add_text(title, font_size=15, position='upper_edge')
    if axes_on:
        p.add_axes()

    p.show(auto_close=auto_close, full_screen=True)  # interactive_update=not auto_close, interactive=auto_close,
    return p


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def _mesh_grid_test():
    # from cfg import Assessssts
    # v, f = Assets.MAN.load()
    # v = np.load(r"R:\Mano\data\DFaust\DFAUST_VERT_PICK\50002\chicken_wings\00000.npy")
    from color import all_colormap_names
    import meshio
    file = meshio.read(r"R:\Mano\data\DFaust\DFaust\full\50007\jiggle_on_toes\00076.OFF", "off")
    # print(mesh)
    # print(file.__dict__)
    v, f = file.points, file.get_cells_type("triangle")
    # print(f)
    plot_mesh_grid(vs=[v], fs=f, cmap=all_colormap_names()[0], colors=v[:, 0],
                   spacer_x=1, spacer_y=1)


def _mesh_montage_test():
    from cfg import Assets
    from external import vertex_normals, face_normals, edges
    from vis.color import all_colormap_names
    v, f = Assets.HAND.load()
    vn = vertex_normals(v, f, normalized=False)
    fn = face_normals(v, f, normalized=False)
    e = edges(v, f)

    camera_pos = np.array([(0.2759962011548741, 0.1921074582877638, -1.7200614751015812),
                           (0.022573115432591384, -0.0049156500333470965, 0.002758298917835779),
                           (-0.24509965409732554, -0.958492103686741, -0.14566758984598366)])

    # plot_mesh_montage([v] * 9, f)
    plot_mesh_montage([v] * 9, [f] * 9)
    plot_mesh_montage([v] * 2, f, camera_pos=[camera_pos * 5, camera_pos], link_plots=False)

    # plot_mesh_montage([v] * 2, f,colors=np.arange(v.shape[0]), cmap=['jet','rainbow'])
    all_names = all_colormap_names()  # Very, very cool!
    # plot_mesh_montage([v] * len(all_names), f, colors=np.arange(f.shape[0]), cmap=all_names, titles=all_names)
    color = np.random.choice([True, False], size=len(v), replace=True)
    plot_mesh_montage([v] * len(all_names), f, style='spheres', colors=color, cmap=all_names, titles=all_names)
    # camera_pos=camera_pos)
    # all_names = all_color_names()  # Very, very cool!
    # plot_mesh_montage([v] * len(all_names), f, colors=all_names, titles=all_names,camera_pos=camera_pos)
    # from numeric.np import split_to_rows
    # fs = split_to_rows(f, 5)
    # plot_mesh_montage([v] * len(fs), fs, camera_pos=camera_pos)


if __name__ == '__main__':
    _mesh_grid_test()
