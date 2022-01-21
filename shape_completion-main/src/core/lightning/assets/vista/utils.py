import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

from lightning.assets.vista.color import parse_color
from lightning.assets.vista.external import torch2numpy


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
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
    p = pv.Plotter(**kwargs)
    return p


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def concat_cell_qualifier(arr):
    # arr = np.expand_dims(arr,axis=1)
    print(arr.shape)
    return np.concatenate((np.full((arr.shape[0], 1), arr.shape[1]), arr), 1)


def color_to_pyvista_color_params(color, repeats=1):
    color = torch2numpy(color)

    if isinstance(color, str) or len(color) == 3:
        return {'color': parse_color(color)}

    else:
        color = np.asanyarray(color)
        if repeats > 1:
            color = np.repeat(color, axis=0, repeats=repeats)
        return {'scalars': color, 'rgb': color.squeeze().ndim == 2}


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def ordered_point_connection(v):
    poly = pv.PolyData()
    poly.points = v
    edges = np.full((len(v) - 1, 3), 2)  # Set first column to 2 - For 2 Cells
    edges[:, 1] = np.arange(0, len(v) - 1)  # Set 2nd column to 0->n-1
    edges[:, 2] = np.arange(1, len(v))  # Set 3rd column to 1->n
    poly.lines = edges
    return poly


def nearest_neighbor_point_connection(v, k=2):
    o = NearestNeighbors(n_neighbors=k).fit(v)  # TODO - add in different metrics?
    targets = o.kneighbors(v, n_neighbors=k + 1, return_distance=False)[:, 1:].flatten()  # Remove self match
    sources = np.tile(np.arange(len(v)), (k, 1)).transpose().flatten()
    edges = np.full((len(sources), 3), 2)
    edges[:, 1] = sources
    edges[:, 2] = targets

    poly = pv.PolyData()
    poly.points, poly.lines = v, edges
    return poly
