import numpy as np
import pyvista as pv

from lightning.assets.vista.external import face_barycenters, num_edges, edge_centers, point_cloud_radius, edges, last_axis_norm, torch2numpy, \
    unique_rows, l2_normalize
from lightning.assets.vista.utils import concat_cell_qualifier, plotter, color_to_pyvista_color_params

# TODO -
#  * Add Shadow Support
#  * Add Legend Support
#  * Add Texture Support
#  * Merge lines & general lines
#  * Add in automatic scaling support for all glyph functions
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
pv.set_plot_theme('document')  # Change global behaviour

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
"""
                ambient/diffuse/specular strength/specular exponent/specular color
'Shiny',        0.3,	  0.6,	       0.9,	            20,		        1.0
'Dull',	        0.3,	  0.8,	       0.0,	            10,		        1.0
'Metal',        0.3,	  0.3,	       1.0,	            25,		        0.5
'VTK_Shadows'   0.7       0.7          0.51             30 
"""


# ---------------------------------------------------------------------------------------------------------------------#
#                                            Visualization Functions
# ---------------------------------------------------------------------------------------------------------------------#
def plot_mesh(v, f=None, n=None, lines=None,  # Input
              style='surface', smooth_shading=True, eye_dome=False, depth_peeling=False, lighting=None,  # Global arg
              camera_pos=None,
              color='w', normal_color='lightblue', edge_color='darkblue', line_color='darkblue', cmap='rainbow',
              show_edges=False, clim=None, show_scalar_bar=False,  # Color options
              normal_scale=1, point_size=6, line_width=1,  # Scales
              grid_on=False, opacity=1.0, title=None, axes_on=False, unit_sphere_on=False
              # Misc
              ):
    p = plotter()
    add_mesh(p, v=v, f=f, n=n, lines=lines,
             style=style, smooth_shading=smooth_shading, eye_dome=eye_dome, depth_peeling=depth_peeling,
             lighting=lighting,
             camera_pos=camera_pos,
             color=color, normal_color=normal_color, edge_color=edge_color, line_color=line_color,
             cmap=cmap, show_edges=show_edges, clim=clim, show_scalar_bar=show_scalar_bar,
             normal_scale=normal_scale, point_size=point_size, line_width=line_width,
             grid_on=grid_on, opacity=opacity, title=title, axes_on=axes_on, unit_sphere_on=unit_sphere_on)
    p.show()
    return p


def plot_surf(x, y, z, grid_on=True, **plt_args):
    # TODO - make this a bit smarter
    p = plotter()
    add_surf(p, x, y, z, **plt_args)
    if grid_on:
        p.show_bounds(**{'grid': 'back', 'location': 'all', 'ticks': 'both'})
    p.show()


# ---------------------------------------------------------------------------------------------------------------------#
#                                              Additional Functions
# ---------------------------------------------------------------------------------------------------------------------#

def add_mesh(p, v, f=None, n=None, lines=None,  # Input
             style='surface', smooth_shading=True, eye_dome=False, depth_peeling=False, lighting=None,  # Global arg
             camera_pos=None,
             color='w', normal_color='lightblue', edge_color='darkblue', line_color='darkblue', cmap='rainbow',
             show_edges=False, clim=None, show_scalar_bar=False,  # Color options
             normal_scale=1, point_size=6, line_width=1,  # Scales
             grid_on=False, opacity=1.0, title=None, axes_on=False, as_a_single_mesh=False,  # Misc
             unit_sphere_on=False
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

    if not as_a_single_mesh and style in ['glyphed_spheres', 'sphered_wireframe']:
        add_spheres(p, v, point_size=point_size, color=color, cmap=cmap, smooth_shading=smooth_shading,
                    lighting=lighting, opacity=opacity, resolution=8)

    style = _STYLE_RENAMINGS.get(style, style)  # Translation of styles for insertion to Pyvista
    if style != 'glyphed_spheres':
        color_params = color_to_pyvista_color_params(color)
        # Point Size = Diameter of a point (VTK Manual), expressed in Screen Units - automatically scaled by camera
        p.add_mesh(mesh, style=style, smooth_shading=smooth_shading, cmap=cmap, show_edges=show_edges,
                   point_size=point_size, render_points_as_spheres=render_points_as_spheres,
                   edge_color=edge_color,
                   opacity=opacity, lighting=lighting, clim=clim, line_width=line_width, render_lines_as_tubes=True,
                   show_scalar_bar=show_scalar_bar, stitle='',
                   **light_params, **color_params)

    if n is not None and not as_a_single_mesh:
        add_vector_field(p, v, f, n, color=normal_color, scale=normal_scale)
    if camera_pos is None:
        camera_pos = _N_VERTICES_TO_CAM_POS.get(len(v), None)
    if camera_pos is not None:
        p.camera_position = camera_pos

    if title:
        p.add_text(title, font_size=12, position='upper_edge')
    # if legend:
    #     p.add_legend(labels=[(title, 'w')], size=[0.25, 0.25 / 2])  # TODO - enable legend
    if grid_on:
        p.show_bounds(**{'grid': 'back', 'location': 'all', 'ticks': 'both'})
    if eye_dome:
        p.enable_eye_dome_lighting()
    if depth_peeling:
        p.enable_depth_peeling()
    if axes_on:
        p.add_axes()
    # TODO - enable texture:
    # tex = pv.read_texture(texture)
    # self.pv_mesh.texture_map_to_plane(inplace=True)
    # plotter.add_mesh(self.pv_mesh, texture=tex)

    return p


def add_surf(p, x, y, z, **plt_args):
    # TODO - make this a bit smarter
    grid = pv.StructuredGrid(x, y, z)
    p.add_mesh(grid, **plt_args)
    return p


def add_voxelized_mesh(p, v, f, density_factor=1, color='wheat', opacity=0.5, **plt_args):
    from external import numpy2pyvista
    surface = numpy2pyvista(v, f)
    voxels = pv.voxelize(surface, density=density_factor * surface.length / 100)  # Returns an Unstructured Grid

    color_params = color_to_pyvista_color_params(color)
    p.add_mesh(voxels, opacity=opacity, **color_params, **plt_args)

    # voxels.compute_implicit_distance(surface, inplace=True) # Signed Distance Function - Something here is OFF!
    # p.add_mesh(voxels, opacity=opacity,scalars="implicit_distance", **plt_args ,clim=[-0.005,0])


def add_volume():
    # For Uniform Grids
    # https://docs.pyvista.org/examples/02-plot/volume.html#simple-volume-render
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Glyphes
# ---------------------------------------------------------------------------------------------------------------------#


# noinspection PyTypeChecker
def add_spheres(p, v, point_size=6, color='black', resolution=40, lighting=None, smooth_shading=True, size_ref=None,
                **kwargs):
    # Handle Input Params:
    if size_ref is None:
        rad = point_cloud_radius(v)
    else:
        rad = point_cloud_radius(size_ref)
    if rad > 0:
        point_size *= rad / 300
    v = np.atleast_2d(v)

    src = pv.PolyData(v)
    src["radius"] = point_size * np.ones((len(v)))

    geom = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution)
    glyphed = src.glyph(scale="radius", geom=geom)
    mult_factor = glyphed.n_points / len(v)
    assert int(mult_factor) == mult_factor
    color_params = color_to_pyvista_color_params(color, int(mult_factor))
    light_params = _DEF_PYVISTA_LIGHT_PARAMS if lighting is None else _DEF_LIGHT_PARAMS
    p.add_mesh(glyphed, smooth_shading=smooth_shading, **color_params, **light_params, **kwargs)
    return p


def add_lines(p, v, lines, line_width=1, line_color='darkblue', **plt_args):
    # TODO - Merge add_lines & add_general_lines
    mesh = pv.PolyData(v)
    lines = concat_cell_qualifier(lines)
    mesh.lines = lines
    tubes = mesh.tube(radius=line_width / 1000)  # TODO - Handle automatic scaling
    color_params = color_to_pyvista_color_params(line_color, 80)
    p.add_mesh(tubes, smooth_shading=True, **color_params, **plt_args)
    return p


def add_general_lines(p, v1, v2=None, point_size=6, line_width=1,
                      sphere_color='g', line_color='blue', cmap='rainbow',
                      **plt_args):
    # TODO - Merge add_lines & add_general_lines
    if v2 is not None:
        assert v1.shape == v2.shape
        v1, v2 = v1.astype(np.double), v2.astype(np.double)
        v, _, inv_map, _ = unique_rows(np.concatenate((v1, v2), axis=0))
        lines = inv_map[:, np.newaxis].T.reshape(2, -1).T
        assert np.all(np.diff(lines, axis=1) != 0), "Degenerate rows found"
        add_mesh(p, v=v, style='spheres', lines=lines, line_width=line_width, line_color=line_color,
                 point_size=point_size, color=sphere_color,
                 cmap=cmap, **plt_args)
    else:
        spline = pv.Spline(v1, len(v1) * 3)  # x3 for smoothness
        tube = spline.tube(radius=line_width / 1000)  # TODO - Handle automatic scaling
        color_params = color_to_pyvista_color_params(line_color, 80)
        p.add_mesh(tube, smooth_shading=True, **color_params, **plt_args)
    return p


# noinspection PyTypeChecker
def add_vector_field(p, v, f, vf, color='lightblue', scale=1, normalized=True, cmap='rainbow'):
    # Basic Checkups
    if f is None:
        return add_point_cloud_vector_field(p=p, v=v, vf=vf, color=color, scale=scale, normalized=normalized, cmap=cmap)
    nv, nf, ne = len(v), len(f), num_edges(v, f)
    assert nv != nf and nv != ne and ne != nf, "Cannot determine vector field cells"

    # Handle Input Arguments
    if len(vf) == 3 * nf:  # Align
        vf = np.reshape(vf, (3, nf)).T
    if normalized:
        vf = l2_normalize(vf)
    scale *= point_cloud_radius(v) / 17.5

    # Determine Source Points
    if len(vf) == nf:
        pnt_cloud = pv.PolyData(face_barycenters(v, f))
    elif len(vf) == nv:
        pnt_cloud = pv.PolyData(v)
    elif len(vf) == ne:
        pnt_cloud = pv.PolyData(edge_centers(v, f))
    else:
        raise NotImplementedError('Vector field size not supported ')

    pnt_cloud['vectors'] = vf
    pnt_cloud['glyph_scale'] = last_axis_norm(vf)

    arrows = pnt_cloud.glyph(orient="vectors", scale="glyph_scale", factor=scale)
    # An arrow has 15 triangular faces for every original vertex in v
    p.add_mesh(arrows, colormap=cmap, **color_to_pyvista_color_params(color, 15))
    return p


# noinspection PyTypeChecker
def add_point_cloud_vector_field(p, v, vf, color='lightblue', scale=1, normalized=False, cmap='rainbow'):
    # Handle input arguments:
    if len(vf) == 3 * len(v):
        vf = np.reshape(vf, (3, len(v))).T
    assert len(vf) == len(v)
    if normalized:
        vf = l2_normalize(vf)
    scale *= point_cloud_radius(v) / 17.5

    pnt_cloud = pv.PolyData(v)
    pnt_cloud['vectors'] = vf
    pnt_cloud['glyph_scale'] = last_axis_norm(vf)

    arrows = pnt_cloud.glyph(orient="vectors", scale="glyph_scale", factor=scale)
    # An arrow has 15 triangular faces for every original vertex in v
    p.add_mesh(arrows, colormap=cmap, **color_to_pyvista_color_params(color, 15))
    return p


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

def _lines_test(random_lines=False, N=1):
    if random_lines:
        v1 = np.random.rand(N, 3)
        v2 = np.random.rand(N, 3)
    else:
        v1 = np.array([[0, 0, 1], [0, 0, 2]], dtype=np.float)
        v2 = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float)

    p = plotter()
    add_general_lines(p, v1, v2, point_size=20, line_width=5)
    p.show()


# noinspection PyTypeChecker
def _mesh_plot_test():
    from cfg import Assets
    from external import vertex_normals, face_normals
    v, f = Assets.HAND.load()
    vn = vertex_normals(v, f, normalized=False)
    fn = face_normals(v, f, normalized=False) * 1000
    e = edges(v, f)

    camera_pos = [(0.2759962011548741, 0.1921074582877638, -1.7200614751015812),
                  (0.022573115432591384, -0.0049156500333470965, 0.002758298917835779),
                  (-0.24509965409732554, -0.958492103686741, -0.14566758984598366)]

    def plot(vf=None, vf_color='red', **plt_args):
        p = plotter()
        add_mesh(p, v, f, show_edges=True, camera_pos=camera_pos, **plt_args)
        if vf is not None:
            add_vector_field(p, v, f, vf, color=vf_color)
        p.show()
        pass


    plot(lines=e, line_color=[0, 0, 1])  # Lines
    plot(lines=e, line_color=[0, 255, 0])
    plot(lines=e, line_color='cyan')
    plot(lines=e, line_color=np.array([0, 0, 1]))
    plot(lines=e, line_color=np.array([0, 50, 255]))
    plot(lines=e, line_color=[1] * e.shape[0])
    plot(lines=e, line_color=np.array([1] * e.shape[0]))
    plot(lines=e, line_color=np.arange(e.shape[0]))
    plot(lines=e, line_color=np.arange(e.shape[0]).tolist())
    plot(lines=e, line_color=np.random.rand(e.shape[0], 3))

    plot(color=[0, 0, 1])  # Color Suite:
    plot(color=[0, 255, 0])
    plot(color='cyan')
    plot(color=np.array([0, 0, 1]))
    plot(color=np.array([0, 50, 255]))
    plot(color=[1] * v.shape[0])
    plot(color=[1] * f.shape[0])
    plot(color=np.array([1] * v.shape[0]))
    plot(color=np.array([1] * f.shape[0]))
    plot(color=np.arange(v.shape[0]))
    plot(color=np.arange(f.shape[0]))
    plot(color=np.arange(v.shape[0]).tolist())
    plot(color=np.arange(f.shape[0]).tolist())
    plot(color=np.random.rand(v.shape[0], 3))
    plot(color=np.random.rand(f.shape[0], 3))

    plot(vf=vn)  # Vector Suite
    plot(vf=fn)
    plot(vf=vn, vf_color='cyan')
    plot(vf=fn, vf_color='red')
    plot(vf=vn, vf_color=np.array([0, 0, 1]))
    plot(vf=vn, vf_color=np.array([0, 50, 255]))
    plot(vf=vn, vf_color=[1] * v.shape[0])
    plot(vf=fn, vf_color=[1] * f.shape[0])
    plot(vf=vn, vf_color=np.array([1] * v.shape[0]))
    plot(vf=fn, vf_color=np.array([1] * f.shape[0]))
    plot(vf=vn, vf_color=np.arange(v.shape[0]))
    plot(vf=fn, vf_color=np.arange(f.shape[0]))
    plot(vf=vn, vf_color=np.arange(v.shape[0]).tolist())
    plot(vf=fn, vf_color=np.arange(f.shape[0]).tolist())
    plot(vf=vn, vf_color=np.random.rand(v.shape[0], 3))
    plot(vf=fn, vf_color=np.random.rand(f.shape[0], 3))

    plot(style='surface', color=np.arange(v.shape[0]), title='hello')  # Style
    plot(style='points', color=np.arange(v.shape[0]), )
    plot(style='spheres', color=np.arange(v.shape[0]), )
    plot(style='wireframe', color=np.arange(v.shape[0]), )
    plot(style='sphered_wireframe', color=np.arange(v.shape[0]), )

    plot(style='surface', color=[0, 1, 0], )
    plot(style='points', color=[0, 1, 0], )
    plot(style='spheres', color=[0, 1, 0], )
    plot(style='wireframe', color=[0, 1, 0], )
    plot(style='sphered_wireframe', color=[0, 1, 0], )

    plot(style='surface', color='red', )
    plot(style='points', color='red', )
    plot(style='spheres', color='red', )
    plot(style='wireframe', color='red', )
    plot(style='sphered_wireframe', color='red', )

    plot(style='surface', color=np.random.rand(v.shape[0], 3), )
    plot(style='points', color=np.random.rand(v.shape[0], 3), )
    plot(style='spheres', color=np.random.rand(v.shape[0], 3), )
    plot(style='wireframe', color=np.random.rand(v.shape[0], 3), )
    plot(style='sphered_wireframe', color=np.random.rand(v.shape[0], 3), )

def comp(path):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(path)

    vertices, faces =  plydata['vertex'].data,  plydata['face'].data
    # print(vertices)
    vert_new = np.array(list([list(vertex) for vertex in vertices]))
    faces_new = np.array(list([list(face) for face in faces])).squeeze()
    # print(faces_new.squeeze().shape)
    # print(faces_new[1])
    # print(faces_new.shape)
    plot_mesh(vert_new, faces_new)
# -------------------------------------------------------asd--------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    #_lines_test()
    comp(r"C:\Users\ido.iGIP1\hy\Ramp\shape_complFFcometion-main\src\core\results\debug_experiment\version_19\completions\DFaustProj\gt_50007_jiggle_on_toes_5_2_tp_50007_jiggle_on_toes_0_res.ply")
    # 284