from itertools import tee
from typing import Iterator, Tuple

import numpy as np
from numpy.linalg import norm


def _compose_from_faces(corners, faces, n, edge_adjust=None, face_adjust=None):
    # create corner nodes
    vertices = [corners]
    vertex_count = len(corners)
    corner_nodes = np.arange(len(corners))

    # create edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    edges = list(edges)

    # create edge nodes:
    edge_nodes = {}
    t = np.linspace(1 / n, 1.0, n - 1, endpoint=False)
    corners = vertices[0]
    k = corners.shape[0]
    for edge in edges:
        i0, i1 = edge
        new_vertices = np.outer(1 - t, corners[i0]) + np.outer(t, corners[i1])
        if edge_adjust:
            new_vertices = edge_adjust(edge, new_vertices)
        vertices.append(new_vertices)
        vertex_count += len(vertices[-1])
        edge_nodes[edge] = np.arange(k, k + len(t))
        k += len(t)

    # This is the same code as appearing for cell in a single triangle. On each face,
    # those indices are translated into the actual indices.
    triangle_cells = []
    k = 0
    for i in range(n):
        j = np.arange(n - i)
        triangle_cells.append(np.column_stack([k + j, k + j + 1, k + n - i + j + 1]))
        j = j[:-1]
        triangle_cells.append(
            np.column_stack([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
        )
        k += n - i + 1
    triangle_cells = np.vstack(triangle_cells)

    cells = []
    for face in faces:
        corners = face
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        is_edge_reverted = [False, False, False]
        for k, edge in enumerate(edges):
            if edge[0] > edge[1]:
                edges[k] = (edge[1], edge[0])
                is_edge_reverted[k] = True

        # First create the interior points in barycentric coordinates
        if n == 1:
            num_new_vertices = 0
        else:
            bary = (
                np.hstack(
                    [
                        [np.full(n - i - 1, i), np.arange(1, n - i)]
                        for i in range(1, n)
                    ]
                )
                / n
            )
            bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
            corner_verts = np.array([vertices[0][i] for i in corners])
            vertices_cart = np.dot(corner_verts.T, bary).T

            if face_adjust:
                vertices_cart = face_adjust(face, bary, vertices_cart, corner_verts)

            vertices.append(vertices_cart)
            num_new_vertices = len(vertices[-1])

        # translation table
        num_nodes_per_triangle = (n + 1) * (n + 2) // 2
        tt = np.empty(num_nodes_per_triangle, dtype=int)

        # first the corners
        tt[0] = corner_nodes[corners[0]]
        tt[n] = corner_nodes[corners[1]]
        tt[num_nodes_per_triangle - 1] = corner_nodes[corners[2]]
        # then the edges.
        # edge 0
        tt[1:n] = edge_nodes[edges[0]]
        if is_edge_reverted[0]:
            tt[1:n] = tt[1:n][::-1]
        #
        # edge 1
        idx = 2 * n
        for k in range(n - 1):
            if is_edge_reverted[1]:
                tt[idx] = edge_nodes[edges[1]][n - 2 - k]
            else:
                tt[idx] = edge_nodes[edges[1]][k]
            idx += n - k - 1
        #
        # edge 2
        idx = n + 1
        for k in range(n - 1):
            if is_edge_reverted[2]:
                tt[idx] = edge_nodes[edges[2]][k]
            else:
                tt[idx] = edge_nodes[edges[2]][n - 2 - k]
            idx += n - k

        # now the remaining interior nodes
        idx = n + 2
        j = vertex_count
        for k in range(n - 2):
            for _ in range(n - k - 2):
                tt[idx] = j
                j += 1
                idx += 1
            idx += 2

        cells += [tt[triangle_cells]]
        vertex_count += num_new_vertices

    vertices = np.concatenate(vertices)
    cells = np.concatenate(cells)

    return vertices, cells


DEF_CENTER = np.array([0, 0, 0])


def _align_scale(v, radius=1.0, center=DEF_CENTER):
    # Make an iterator over the adjacent pairs: (-1, 0), (0, 1), ..., (N - 2, N - 1)
    length = norm(v, axis=1).reshape((-1, 1))
    return v / length * radius + center


def _make_pair_range(N: int) -> Iterator[Tuple[int, int]]:
    i, j = tee(range(-1, N))
    next(j, None)
    return zip(i, j)