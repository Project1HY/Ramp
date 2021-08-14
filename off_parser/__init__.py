import numpy as np
def __parse_vertex_line_to_tuple(file) :
    line = file.readline().strip()
    if line!="":
        return np.array([float(i) for i in line.strip().split(" ")])
    return None

def __parse_faces_to_tuple(file):
    line = file.readline().strip()
    if line!="":
        return np.array([int(i) for i in line.strip().split(" ")])
    return None

def read_off(path):
    f = open(path,"r")
    next(f)
    metadata = list(map(lambda x: int(x),f.readline().split(" ")))
    vertices = np.stack(list(map(lambda index: __parse_vertex_line_to_tuple(f),range(metadata[0]))))
    faces = np.stack(list(map(lambda index: __parse_faces_to_tuple(f),range(metadata[1]))))
    return (vertices,faces)

def write_off(path,vertices_faces):
    f = open(path,"w")
    f.write("OFF\n")
    vertex_count = len(vertices_faces[0])
    faces_count = len(vertices_faces[1])
    f.write("{} {} 0\n".format(vertex_count,faces_count))
    for vertex in vertices_faces[0]:
        f.write("{} {} {}\n".format(vertex[0],vertex[1],vertex[2]))
    for index,face in enumerate(vertices_faces[1]):
        f.write(' '.join(map(lambda i: str(i),face)))
        if index != len(vertices_faces[1])-1:
            f.write('\n')
            