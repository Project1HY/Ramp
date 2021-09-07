from math import pi
import plotly.graph_objs as go
from functools import reduce
import numpy as np
import matplotlib.cm as cm

# TODO - Correct this file
# ---------------------------------------------------------------------------------------------------------------------#
#                                               Compute
# ---------------------------------------------------------------------------------------------------------------------#
def superformula(phi, m, n1, n2, n3, a=1, b=1,simple=True):
    ''' Computes the position of the point on a superformula curve.
    Superformula has first been proposed by Johan Gielis and is a generalization of superellipse.
    see: http://en.wikipedia.org/wiki/Superformula
    '''
    m /= 4
    t1 = np.cos(m * phi) / a
    t1 = np.abs(t1) ** n2

    t2 = np.sin(m * phi) / b
    t2 = np.abs(t2) ** n3

    return (t1 + t2) ** (-1 / n1)



def supercurve(m=16, n1=0.5, n2=0.5, n3=16, a=1, b=1, n_pts=100, end_point=2 * pi):
    phi = np.linspace(0, end_point, n_pts)
    r= superformula(phi, m, n1, n2, n3, a, b)
    return np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1)


def tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))


def map_z2color(zval, colormap, vmin, vmax):
    # map the normalized value zval to a corresponding color in the colormap

    if vmin > vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    R, G, B, alpha = colormap(t)
    return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'.format(int(G * 255 + 0.5)) + \
           ',' + '{:d}'.format(int(B * 255 + 0.5)) + ')'


def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    # x, y, z are lists of coordinates of the triangle vertices
    # simplices are the simplices that define the triangularization;
    # simplices  is a numpy array of shape (no_triangles, 3)
    # insert here the  type check for input data

    points3D = np.vstack((x, y, z)).T
    tri_vertices = map(lambda index: points3D[index], simplices)  # vertices of the surface triangles
    from util.mesh.plots import plot_mesh
    plot_mesh(points3D,simplices,strategy='wireframe')
    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]  # mean values of z-coordinates of
    # triangle vertices
    min_zmean = np.min(zmean)
    max_zmean = np.max(zmean)
    facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
    I, J, K = tri_indices(simplices)

    triangles = go.Mesh3d(x=x,
                          y=y,
                          z=z,
                          facecolor=facecolor,
                          i=I,
                          j=J,
                          k=K,
                          name=''
                          )

    if plot_edges is None:  # the triangle sides are not plotted
        return [triangles]
    else:
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

        # define the lines to be plotted
        lines = go.Scatter3d(x=Xe,
                             y=Ye,
                             z=Ze,
                             mode='lines',
                             line=dict(color='rgb(50,50,50)', width=1.5)
                             )
        return [triangles, lines]

# function [M] = torus(r,R)
#
# if ~exist('R','var'); R = 5; end % outer radius of torus
# if ~exist('r','var'); r = 2; end % inner tube radius
#
# th=linspace(0,2*pi,36); % e.g. 36 partitions along perimeter of the tube
# phi=linspace(0,2*pi,18); % e.g. 18 partitions along azimuth of torus
# % we convert our vectors phi and th to [n x n] matrices with meshgrid command:
# [Phi,Th]=meshgrid(phi,th);
# % now we generate n x n matrices for x,y,z according to eqn of torus
# x=(R+r.*cos(Th)).*cos(Phi);
# y=(R+r.*cos(Th)).*sin(Phi);
# z=r.*sin(Th);
# [f,v] = surf2patch(x,y,z,'triangles');
# M = Mesh(v,f,sprintf('torus-r%g-r%g',r,R),which(mfilename));
# M = mesh_deduplication(M);
# end



def supershape(pset_1, pset_2, n_pts):
    from scipy.spatial import Delaunay
    theta = np.linspace(-pi, pi, n_pts)
    phi = np.linspace(-pi / 2, pi / 2, n_pts)

    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    points2D = np.vstack([theta, phi]).T
    tri = Delaunay(points2D)  # triangulate the rectangle U
    f = tri.simplices


    r1 = superformula(theta, *pset_1)
    r2 = superformula(phi, *pset_2)
    x = r1 * np.cos(theta) * r2 * np.cos(phi)
    y = r1 * np.sin(theta) * r2 * np.cos(phi)
    z = r1 * np.sin(phi)
    data1 = plotly_trisurf(x, y, z, f)
    axis = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
    )

    layout = go.Layout(
        title='Moebius band triangulation',
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
            aspectratio=dict(
                x=1,
                y=1,
                z=0.5
            ),
        )
    )

    fig1 = go.Figure(data=data1, layout=layout)
    fig1.show()


# ---------------------------------------------------------------------------------------------------------------------#
#                                                   Dataset Generator
# ---------------------------------------------------------------------------------------------------------------------#
class ParamGenerator:
    @staticmethod
    def positive_half_integer(count, n_params=4, scale=5):
        return [np.round(scale * np.random.rand(n_params) * 2) / 2 + 0.5 for _ in range(count)]

    @staticmethod
    def random(count, n_params=4, scale=5):
        return [np.round(scale * np.random.rand(n_params), 2) for _ in range(count)]

    @staticmethod
    def wikipedia2D():
        return [[3, 4.5, 10, 10], [4, 12, 15, 15],
                [7, 10, 6, 6], [5, 4, 4, 4],
                [5, 2, 7, 7], [5, 2, 13, 13],
                [4, 1, 1, 1], [4, 1, 7, 8],
                [6, 1, 7, 8], [2, 2, 2, 2],
                [1, 0.5, 0.5, 0.5], [2, 0.5, 0.5, 0.5],
                [3, 0.5, 0.5, 0.5], [5, 1, 1, 1],
                [2, 1, 1, 1], [7, 3, 4, 17],
                [2, 1, 4, 8], [6, 1, 4, 8],
                [7, 2, 8, 4], [4, 0.5, 0.5, 4],
                [8, 0.5, 0.5, 8], [16, 0.5, 0.5, 16],
                [3, 30, 15, 15], [4, 30, 15, 15]]


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def curve_tester():
    from util.mesh.plots import plot_montage
    # TODO - Truncate numerical instabilities
    n_curves = 25
    # params = ParamGenerator.random(n_curves,scale=10)
    # params = ParamGenerator.wikipedia2D()
    # elems = [supercurve(*p, n_pts=500) for p in params]
    # plot_montage(elems, params)

    # import matplotlib.pyplot as plt
    # m_n1_n2_n3 = np.random.rand(5)
    # m_n1_n2_n3 = [1, 1, 1, 1]
    # m_n1_n2_n3 = [16, 0.5, 0.5, 16]
    m_n1_n2_n3 = [7,2,8,4]
    # xy = supercurve(*m_n1_n2_n3, n_pts=500)
    # plt.plot(xy[:, 0], xy[:, 1])
    # plt.title(' '.join(str(n) for n in m_n1_n2_n3))
    # plt.show()
    supershape(m_n1_n2_n3, m_n1_n2_n3, 100)


if __name__ == '__main__':
    curve_tester()
