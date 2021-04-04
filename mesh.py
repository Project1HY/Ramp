from off_parser import read_off,write_off
from scipy.sparse import lil_matrix ,identity
import numpy as np
import pyvista as pv
import matplotlib.colors 
import sklearn.preprocessing 

class Mesh(object):
    def __init__(self,path):
        vertices_faces =  read_off(path)
        self.v=vertices_faces[0]
        self.f=vertices_faces[1]
        self.mesh_poly=pv.PolyData(self.v,self.f)
        
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
        pv.plot(pv.PolyData(self.v,self.f),style="wireframe")

    def render_pointcloud(self,color_func=None):
        if color_func == None:
            color_func=np.random.rand(len(self.f))
        plotter=pv.Plotter()
        plotter.add_mesh(self.mesh_poly,style="points",point_size=100,render_points_as_spheres=True,scalars=color_func)
        plotter.show()

    def render_surface(self,color_func=None,**kwargs):
        plotter=pv.Plotter()
        if color_func == None:
            color_func=np.random.rand(len(self.f))
        if "cmap" in kwargs:
            plotter.add_mesh(self.mesh_poly,cmap=kwargs["cmap"],scalars=color_func)
        else:
            plotter.add_mesh(self.mesh_poly,scalars=color_func)
        plotter.show()

    def calculate_area(self,face):
        edge1 = self.v[face[1]]-self.v[face[2]]
        edge2 = self.v[face[1]]-self.v[face[3]]
        normal=np.cross(edge1,edge2)
        area = np.linalg.norm(normal)/2       
        return area

    def calculate_cross_product(self,face):
        edge1 = self.v[face[1]]-self.v[face[2]]
        edge2 = self.v[face[1]]-self.v[face[3]]
        normal=np.cross(edge1,edge2)
        normal = normal/np.linalg.norm(normal)        
        return normal

    def calculate_face_normals(self):
        normal = np.apply_along_axis(self.calculate_cross_product,axis=1,arr=self.f)
        return normal

    def calculate_face_barycenters(self):
        vertex = np.array(self.v)
        vertex[self.f[0][0]] = vertex[self.f[0][0]]*0
        return self.vertex_face_adjacency().transpose()*vertex/3

    def calculate_face_areas(self):
        areas = np.apply_along_axis(self.calculate_area,axis=1,arr=self.f)
        return areas

    def calculate_barycentric_vertex_area(self):
        return self.vertex_face_adjacency()*self.calculate_face_areas()/3

    def calculate_vertex_normals(self):
        f_normals = self.calculate_face_normals()
        f_areas = self.calculate_face_areas()
        normed_by_area=f_normals*f_areas[:,np.newaxis]
        return self.vertex_face_adjacency()*normed_by_area

    #def edge_face_adjacency(self):

    """def gaussian_curvature(self):
        vertex_areas = self.calculate_barycentric_vertex_area()
        for col in self.vertex_face_adjacency().transpose():
            
            #print(col[1])
            #print("hi")"""

    def visualize_vertex_normals(self, normalized=True, mag=1):
        plotter=pv.Plotter()
        vectors = self.calculate_vertex_normals()
        if normalized:
            mag = mag/np.mean(np.linalg.norm(vectors, axis=1))
        self.mesh_poly.vectors = vectors
        plotter.add_arrows(np.array(self.v), vectors, mag=mag)
        plotter.show()

    def visualize_face_normals(self, normalized=True, mag=1):
        plotter=pv.Plotter()
        vectors = self.calculate_face_normals()
        if normalized:
            mag = mag/np.mean(np.linalg.norm(vectors, axis=1))
        plotter.add_arrows(self.calculate_face_barycenters(), vectors, mag=mag)
        plotter.show()
    
    def calculate_vertex_centroid(self):
        vertices = np.array(self.v)
        mean_vertices = np.mean(vertices, axis=0)
        scalar_func = np.linalg.norm(vertices - mean_vertices, axis=1)
        plotter=pv.Plotter()
        plotter.add_mesh(self.mesh_poly,style="points",point_size=10,render_points_as_spheres=True, scalars=scalar_func)
        plotter.add_points(mean_vertices, render_points_as_spheres=True, point_size = 20, color = 'red')
        plotter.show()

    
            
mesh =Mesh("vase.off")
# print(len(mesh.f))
#print(mesh.gaussian_curvature())
#mesh.visualize_vertex_normals()
mesh.calculate_vertex_centroid()

# mesh.render_surface(vert_deg)