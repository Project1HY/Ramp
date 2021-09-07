import numpy as np
from geom.mesh.synth.helpers import _compose_from_faces


def ngon(p, n, offset=np.pi / 2):
    k = np.arange(p)
    corners = np.vstack(
        [
            [[0.0, 0.0]],
            np.array(
                [
                    np.cos(2 * np.pi * k / p + offset),
                    np.sin(2 * np.pi * k / p + offset),
                ]
            ).T,
        ]
    )
    faces = [(0, k + 1, k + 2) for k in range(p - 1)] + [[0, p, 1]]
    return _compose_from_faces(corners, faces, n)


def disk(p, n, offset=np.pi / 2):
    k = np.arange(p)
    corners = np.vstack(
        [
            [[0.0, 0.0]],
            np.array(
                [
                    np.cos(2 * np.pi * k / p + offset),
                    np.sin(2 * np.pi * k / p + offset),
                ]
            ).T,
        ]
    )
    faces = [(0, k + 1, k + 2) for k in range(p - 1)] + [[0, p, 1]]

    def edge_adjust(edge, verts):
        if 0 in edge:
            return verts
        dist = np.sqrt(np.einsum("ij,ij->i", verts, verts))
        return verts / dist[:, None]

    def face_adjust(face, bary, verts, corner_verts):
        assert face[0] == 0
        edge_proj_bary = np.array([np.zeros(bary.shape[1]), bary[1], bary[2]]) / (
                bary[1] + bary[2]
        )
        edge_proj_cart = np.dot(corner_verts.T, edge_proj_bary).T
        dist = np.sqrt(np.einsum("ij,ij->i", edge_proj_cart, edge_proj_cart))
        return verts / dist[:, None]

    return _compose_from_faces(
        corners, faces, n, edge_adjust=edge_adjust, face_adjust=face_adjust
    )


def rectangle(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nx=11, ny=11, variant="zigzag"):
    def _up(xmin, xmax, ymin, ymax, nx, ny):
        # Create the vertices.
        x_range = np.linspace(xmin, xmax, nx)
        y_range = np.linspace(ymin, ymax, ny)
        nodes = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T

        # Create the elements (cells).
        # a = [i + j*nx]
        a = np.add.outer(np.arange(nx - 1), nx * np.arange(ny - 1))
        elems0 = np.array([a, a + 1, a + nx + 1]).reshape(3, -1).T
        elems1 = np.array([a, a + 1 + nx, a + nx]).reshape(3, -1).T
        elems = np.concatenate([elems0, elems1])

        return nodes, elems

    def _down(xmin, xmax, ymin, ymax, nx, ny):
        # Create the vertices.
        x_range = np.linspace(xmin, xmax, nx)
        y_range = np.linspace(ymin, ymax, ny)
        nodes = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T

        # Create the elements (cells).
        # a = [i + j*nx]
        a = np.add.outer(np.arange(nx - 1), nx * np.arange(ny - 1))
        elems0 = np.array([a, a + 1, a + nx]).reshape(3, -1).T
        elems1 = np.array([a + 1, a + 1 + nx, a + nx]).reshape(3, -1).T
        elems = np.concatenate([elems0, elems1])

        return nodes, elems

    def _center(xmin, xmax, ymin, ymax, nx, ny):
        assert (
                nx % 2 == 1 and ny % 2 == 1
        ), "center mode only works with an odd number of cells"
        # Create the vertices.
        x_range = np.linspace(xmin, xmax, nx)
        y_range = np.linspace(ymin, ymax, ny)
        nodes = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T

        # Create the elements (cells).
        # a = [i + j*nx]
        a = np.add.outer(np.arange(nx - 1), nx * np.arange(ny - 1))

        elems = []
        nx2 = (nx - 1) // 2
        ny2 = (ny - 1) // 2

        # bottom left
        ax0 = a[:nx2, :ny2]
        elems.append(np.array([ax0, ax0 + 1, ax0 + nx + 1]).reshape(3, -1).T)
        elems.append(np.array([ax0, ax0 + 1 + nx, ax0 + nx]).reshape(3, -1).T)

        # bottom right
        ax0 = a[nx2:, :ny2]
        elems.append(np.array([ax0, ax0 + 1, ax0 + nx]).reshape(3, -1).T)
        elems.append(np.array([ax0 + 1, ax0 + 1 + nx, ax0 + nx]).reshape(3, -1).T)

        # top left
        ax0 = a[:nx2, ny2:]
        elems.append(np.array([ax0, ax0 + 1, ax0 + nx]).reshape(3, -1).T)
        elems.append(np.array([ax0 + 1, ax0 + 1 + nx, ax0 + nx]).reshape(3, -1).T)

        # top right
        ax0 = a[nx2:, ny2:]
        elems.append(np.array([ax0, ax0 + 1, ax0 + nx + 1]).reshape(3, -1).T)
        elems.append(np.array([ax0, ax0 + 1 + nx, ax0 + nx]).reshape(3, -1).T)

        elems = np.concatenate(elems)

        return nodes, elems

    def _zigzag(xmin, xmax, ymin, ymax, nx, ny):
        # Create the vertices.
        x_range = np.linspace(xmin, xmax, nx)
        y_range = np.linspace(ymin, ymax, ny)
        nodes = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T

        # Create the elements (cells).
        # a = [i + j*nx]
        a = np.add.outer(np.arange(nx - 1), nx * np.arange(ny - 1))

        # [i + j*nx, i+1 + j*nx, i+1 + (j+1)*nx]
        elems0 = np.dstack([a, a + 1, a + nx + 1])
        # [i+1 + j*nx, i+1 + (j+1)*nx, i + (j+1)*nx] for "every other" element
        elems0[0::2, 1::2, 0] += 1
        elems0[1::2, 0::2, 0] += 1
        elems0[0::2, 1::2, 1] += nx
        elems0[1::2, 0::2, 1] += nx
        elems0[0::2, 1::2, 2] -= 1
        elems0[1::2, 0::2, 2] -= 1

        # [i + j*nx, i+1 + (j+1)*nx,  i + (j+1)*nx]
        elems1 = np.dstack([a, a + 1 + nx, a + nx])
        # [i + j*nx, i+1 + j*nx, i + (j+1)*nx] for "every other" element
        elems1[0::2, 1::2, 1] -= nx
        elems1[1::2, 0::2, 1] -= nx

        elems = np.concatenate([elems0.reshape(-1, 3), elems1.reshape(-1, 3)])

        return nodes, elems

    if variant == "zigzag":
        return _zigzag(xmin, xmax, ymin, ymax, nx, ny)
    elif variant == "center":
        return _center(xmin, xmax, ymin, ymax, nx, ny)
    elif variant == "down":
        return _down(xmin, xmax, ymin, ymax, nx, ny)

    assert variant == "up"
    return _up(xmin, xmax, ymin, ymax, nx, ny)


def triangle(n):
    # Create the mesh in barycentric coordinates
    bary = (
            np.hstack(
                [[np.full(n - i + 1, i), np.arange(n - i + 1)] for i in range(n + 1)]
            )
            / n
    )
    bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])

    # Some applications rely on the fact that not values like -1.4125e-16 appear.
    bary[bary < 0.0] = 0.0
    bary[bary > 1.0] = 1.0

    cells = []
    k = 0
    for i in range(n):
        j = np.arange(n - i)
        cells.append(np.column_stack([k + j, k + j + 1, k + n - i + j + 1]))
        #
        j = j[:-1]
        cells.append(
            np.column_stack([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
        )
        k += n - i + 1

    cells = np.vstack(cells)

    corners = np.array(
        [
            [0.0, -0.5 * np.sqrt(3.0), +0.5 * np.sqrt(3.0)],
            [1.0, -0.5, -0.5],
        ]
    )
    points = np.dot(corners, bary).T

    return points, cells


if __name__ == '__main__':
    from geom.mesh.vis.base_2d import plot_2d_mesh

    # plot_mesh(*ngon(5,11), strategy='mesh',show_edges=True)
    plot_2d_mesh(*triangle(50))
    # plot2d(*ngon(5,11))
