import numpy as np
import torch
import json
import os
import random


def batch_arc_length(x):
    """
    https://github.com/fwilliams/deep-geometric-prior/blob/master/utils.py
    Compute the arclength of a curve sampled at a sequence of points.
    :param x: A [b, n, d] tensor of minibaches of d-dimensional point sequences.
    :return: A tensor of shape [b] where each entry, i, is estimated arclength for the curve samples x[i, :, :]
    """
    v = x[:, 1:, :] - x[:, :-1, :]
    return torch.norm(v, dim=2).sum(1)


def batch_curvature(x):
    """
    https://github.com/fwilliams/deep-geometric-prior/blob/master/utils.py
    Compute the discrete curvature for a sequence of points on a curve lying in a 2D embedding space
    :param x: A [b, n, 2] tensor where each [i, :, :] is a sequence of n points lying along some curve
    :return: A [b, n-2] tensor where each [i, j] is the curvature at the point x[i, j+1, :]
    """
    # x has shape [b, n, d]
    b = x.shape[0]
    n_x = x.shape[1]
    n_v = n_x - 2

    v = x[:, 1:, :] - x[:, :-1, :]                         # v_i = x_{i+1} - x_i
    v_norm = torch.norm(v, dim=2)
    v = v / v_norm.view(b, n_x-1, 1)                       # v_i = v_i / ||v_i||
    v1 = v[:, :-1, :].contiguous().view(b * n_v, 1, 2)
    v2 = v[:, 1:, :].contiguous().view(b * n_v, 1, 2)
    c_c = torch.bmm(v1, v2.transpose(1, 2)).view(b, n_v)   # \theta_i = <v_i, v_i+1>
    return torch.acos(torch.clamp(c_c, min=-1.0, max=1.0)) / v_norm[:, 1:]


def batch_vertex_normals(x):
    """
    https://github.com/fwilliams/deep-geometric-prior/blob/master/utils.py
    Compute approximated normals for a sequence of point samples along a curve in 2D.
    :param x: A tensor of shape [b, n, 2] where each x[i, :, :] is a sequence of n 2d point samples on a curve
    :return: A tensor of shape [b, n, 2] where each [i, j, :] is the estimated normal for point x[i, j, :]
    """
    b = x.shape[0]
    n_x = x.shape[1]

    n = torch.zeros(x.shape)
    n[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    n[:, -1, :] = (x[:, -1, :] - x[:, -2, :])
    n = n[:, :, [1, 0]]
    n[:, :, 0] = -n[:, :, 0]
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    n[:, 1:, :] = 0.5*(n[:, 1:, :] + n[:, :-1, :])
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    return n
