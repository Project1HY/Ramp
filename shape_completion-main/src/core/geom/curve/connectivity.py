import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors


def ordered_point_connection(points):
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2)  # Set first column to 2 - For 2 Cells
    cells[:, 1] = np.arange(0, len(points) - 1)  # Set 2nd column to 0->n-1
    cells[:, 2] = np.arange(1, len(points))  # Set 3rd column to 1->n
    poly.lines = cells
    return poly


def nearest_neighbor_point_connection(points, k=2):
    o = NearestNeighbors(n_neighbors=k).fit(points)  # TODO - add in different metrics?
    targets = o.kneighbors(points, n_neighbors=k + 1, return_distance=False)[:, 1:].flatten()  # Remove self match
    sources = np.tile(np.arange(len(points)), (k, 1)).transpose().flatten()
    cells = np.full((len(sources), 3), 2)
    cells[:, 1] = sources
    cells[:, 2] = targets

    poly = pv.PolyData()
    poly.points, poly.lines = points, cells
    return poly
