import numpy as np
from geom.matrix.eulerangles import euler2mat

def plot_montage(elems, titles=None, adjust=(0.02, 0.03, 0.97, 0.95, 0.5, 0.26), scatter_instead=False):
    # TODO - Move me
    import matplotlib.pyplot as plt
    n_elems = len(elems)
    n_rows = int(np.floor(np.sqrt(n_elems)))
    n_cols = int(np.ceil(n_elems / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, ax in enumerate(axes):
        if i < n_elems:
            el = elems[i]
            if scatter_instead:
                ax.scatter(el[:, 0], el[:, 1])
            else:
                ax.plot(el[:, 0], el[:, 1], linewidth=2)
            if titles is not None:
                ax.set_title(str(titles[i]))
            ax.axis('equal')
        else:
            ax.axis("off")

    fig.subplots_adjust(*adjust)
    plt.show()


def plot_2d_mesh(v2d, f, edge_color="k", face_color="coral", show_axes=False):
    # TODO - move me
    """Plot a 2D mesh using matplotlib.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    plt.axis("equal")
    if not show_axes:
        ax.set_axis_off()

    xmin = np.amin(v2d[:, 0])
    xmax = np.amax(v2d[:, 0])
    ymin = np.amin(v2d[:, 1])
    ymax = np.amax(v2d[:, 1])

    width = xmax - xmin
    xmin -= 0.1 * width
    xmax += 0.1 * width

    height = ymax - ymin
    ymin -= 0.1 * height
    ymax += 0.1 * height

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for cell in f:
        import matplotlib.patches

        poly = matplotlib.patches.Polygon(v2d[cell], ec=edge_color, fc=face_color)
        ax.add_patch(poly)

    # import matplotlib.tri
    # tri = matplotlib.tri.Triangulation(points[:,0], points[:,1], triangles=cells)
    # ax.triplot(tri, '-', lw=1, color="k")
    plt.show()


def cloud2depth(input_points, canvasSize=500, space=200, diameter=25,
                xrot=0, yrot=0, zrot=0, switch_xyz=(0, 1, 2), normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """

    # TODO - find somewhere else to place this
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image
