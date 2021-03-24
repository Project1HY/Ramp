from off_parser import read_off,write_off
from scipy.sparse import lil_matrix ,identity
import numpy as np
import pyvista as pv
class Mesh(object):
    def __init__(self,path):
        vertices_faces =  read_off(path)
        self.v=vertices_faces[0]
        self.f=vertices_faces[1]
    def vertex_face_adjacency(self):
        adjacency = lil_matrix((len(self.v),len(self.f)))
        for face_index,face in enumerate(self.f):
            for vert_index in face:
                adjacency[vert_index,face_index]=True
        return adjacency > 0
    def vertex_vertex_adjacency(self):
        v_f_adjacency = self.vertex_face_adjacency()
        return (v_f_adjacency*v_f_adjacency.transpose() - identity(len(self.v)))>0
    def vertex_degree(self):
        v_v_adjecency = self.vertex_vertex_adjacency()
        return v_v_adjecency*np.ones(len(self.v))
    def render_wireframe(self):
        data = pv.DataSet(self)

print(Mesh("example.off").render_wireframe())