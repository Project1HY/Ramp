## face area 
# Compute area of all face
#
## Syntax
#   fa = face_area(face,vertex)
#
## Description
#  face  : double array, nf x 3, connectivity of mesh
#  vertex: double array, nv x 3, vertex of mesh
# 
#  fa: double array, nf x 1, area of all faces.
#


# Checked at 6/28/2020 by yanshuai
from numpy import *


def triangle_area(vi, vj, vk):
    vij = vj - vi
    vjk = vk - vj
    vki = vi - vk
    a = linalg.norm(vij, axis=1)
    b = linalg.norm(vjk, axis=1)
    c = linalg.norm(vki, axis=1)
    s = (a + b + c) / 2.0
    fa = sqrt(s * (s - a) * (s - b) * (s - c))
    return fa


def face_area(face, vertex):
    fi = face[:, 0]
    fj = face[:, 1]
    fk = face[:, 2]
    vi = vertex[fi, :]
    vj = vertex[fj, :]
    vk = vertex[fk, :]
    return triangle_area(vi, vj, vk)
