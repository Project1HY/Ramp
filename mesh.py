from off_parser import read_off,write_off
from scipy.sparse import lil_matrix 
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
        return adjacency
print(Mesh("example.off").vertex_face_adjacency().todense())