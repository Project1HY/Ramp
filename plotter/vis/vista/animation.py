from pathlib import Path

import numpy as np

from vis.vista.multi_plot import plot_mesh_montage, add_mesh, plotter
from util.container import stringify
from util.execution import busy_wait


# TODO - insert explicit mesh naming, so we do not relay on updating the last mesh.
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def multianimate(vss, fs=None, titles=None, color='lightcoral', **plt_args):
    print('Orient the view, then press "q" to start animation')
    p, ms = plot_mesh_montage(vs=[vs[0] for vs in vss], fs=fs, titles=titles, auto_close=False, colors=color,
                              ret_meshes=True, **plt_args)

    num_frames_per_vs = [len(vs) for vs in vss]
    longest_sequence = max(num_frames_per_vs)
    for i in range(longest_sequence):  # Iterate over all frames
        for mi, m in enumerate(ms):
            if i < num_frames_per_vs[mi]:
                p.update_coordinates(points=vss[mi][i], mesh=m, render=False)
                if i == num_frames_per_vs[mi] - 1:
                    for k, actor in p.renderers[mi]._actors.items():
                        if k.startswith('PolyData'):
                            actor.GetProperty().SetColor([0.8] * 3)  # HACKY!
        for renderer in p.renderers:  # TODO - is there something smarter than this?
            renderer.ResetCameraClippingRange()
        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def animate(vs, f=None, gif_name=None, titles=None, first_frame_index=0, pause=0.05, color='w',
            callback_func=None, setup_func=None, **plt_args):
    p = plotter()
    add_mesh(p, vs[first_frame_index], f, color=color, as_a_single_mesh=True, **plt_args)
    if setup_func is not None:
        setup_func(p)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    if titles is not None:
        titles = stringify(titles)
        p.add_text(titles[first_frame_index], position='upper_edge', name='my_title')
    p.show(auto_close=False, full_screen=True)

    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, v in enumerate(vs):
        if titles is not None:
            p.add_text(titles[i], position='upper_edge', name='my_title')
        if callback_func is not None:
            v = callback_func(p, v, i, len(vs))
        p.update_coordinates(v, render=False)
        # p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program

        if gif_name is not None:
            p.write_frame()

        if i == len(vs) - 1:
            for k, actor in p.renderer._actors.items():
                actor.GetProperty().SetColor([0.8] * 3)  # HACKY!

        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.


def animate_color(colors, v, f=None, gif_name=None, titles=None, first_frame_index=0, pause=0.05, setup_func=None,
                  reset_clim=False, **plt_args):
    # TODO - find some smart way to merge to animate
    # TODO - fix jitter in the title change
    p = plotter()
    add_mesh(p, v, f, color=colors[first_frame_index], as_a_single_mesh=True, **plt_args)
    if setup_func is not None:
        setup_func(p)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    if titles is not None:
        titles = stringify(titles)
        p.add_text(titles[first_frame_index], position='upper_edge', name='my_title')
    p.show(auto_close=False, full_screen=True)

    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, color in enumerate(colors):
        p.update_scalars(color, render=False)
        if reset_clim:
            p.update_scalar_bar_range(clim=[np.min(color), np.max(color)])
        if titles is not None:
            p.add_text(titles[i], position='upper_edge', name='my_title')
        # p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program

        if gif_name is not None:
            p.write_frame()

        if i == len(colors) - 1:
            for k, actor in p.renderer._actors.items():
                actor.GetProperty().SetColor([0.8] * 3)  # HACKY!

        p.update()
    p.show(full_screen=False)  # To allow for screen hanging

