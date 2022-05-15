from geom.mesh.io.base_plot import plotter, stringify, add_mesh, busy_wait


def animate(vs, f=None, gif_name=None, titles=None, first_frame_index=0, pause=0.05,
            callback_func=None, setup_func=None, **plt_args):

    colors = plt_args['colors']
    
    p = plotter()
    #assert False, f'colors are {colors} first index is {first_frame_index}'
    #color = 'w' if colors is None else colors[first_frame_index]
    color ='w'

    add_mesh(p, vs[first_frame_index], f, color=color , as_a_single_mesh=True, **plt_args)
    if setup_func is not None:
        setup_func(p)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    if titles is not None:
        titles = stringify(titles)
        p.add_text(titles[first_frame_index], position='upper_edge', name='my_title')
    p.show(auto_close=False, full_screen=True)
    colors = ['w']*len(vs) if colors is not None else colors
    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, (v,color) in enumerate(zip(vs,colors)):
        if titles is not None:
            p.add_text(titles[i], position='upper_edge', name='my_title')
        if callback_func is not None:
            v = callback_func(p, v, i, len(vs))
        p.update_coordinates(v, render=False)
        p.update_scalars(color, render=False)
        
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
