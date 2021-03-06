from off_parser import read_off,write_off
from scipy.sparse import csr_matrix ,identity
import numpy as np
import pyvista as pv
import matplotlib.colors 
import sklearn.preprocessing 
from typing import Union 

class Mesh(object):
    def __init__(self,path):
        vertices_faces =  read_off(path)
        self.v=vertices_faces[0]
        self.mesh_poly=pv.PolyData(vertices_faces[0],vertices_faces[1])
        self.f=vertices_faces[1][:,1:]
        
    def vertex_face_adjacency(self,weight: Union[str, None, np.ndarray] = None):
        row = self.f.ravel()  # Flatten indices
        col = np.repeat(np.arange(len(self.f)), 3)  # Data for vertices        
        if weight is None:
            weight = np.ones(len(col), dtype=np.bool)
        print(weight.shape)
        vf = csr_matrix((weight, (row, col)), shape=(self.v.shape[0], len(self.f)), dtype=weight.dtype)
        return vf

    def vertex_vertex_adjacency(self):
        v_f_adjacency = self.vertex_face_adjacency()
        return (v_f_adjacency*v_f_adjacency.transpose() - identity(len(self.v)))>0

    def vertex_degree(self):
        v_v_adjecency = self.vertex_vertex_adjacency()
        return np.sum(v_v_adjecency,axis=1)

    def render_wireframe(self):
        pv.plot(self.mesh_poly,style="wireframe")

    def render_pointcloud(self,color_func=None):
        # if color_func == None:
            # color_func=np.random.rand(len(self.f))
        plotter=pv.Plotter()
        plotter.add_mesh(self.mesh_poly,style="points",point_size=20,render_points_as_spheres=True,scalars=color_func)
        plotter.show()

    def render_surface(self,color_func=0,**kwargs):
        plotter=pv.Plotter()
        plotter.add_mesh(self.mesh_poly,scalars=np.random.rand(len(self.f)))
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
        vec1 = self.v[self.f[:,0]]-self.v[self.f[:,1]]
        vec2 = self.v[self.f[:,0]]-self.v[self.f[:,2]]
        norm = np.cross(vec1,vec2)    
        return norm

    def calculate_face_barycenters(self):
        vertex = np.array(self.v)
        return self.vertex_face_adjacency().transpose()*vertex/3

    def calculate_face_areas(self):
        return np.linalg.norm(self.calculate_face_normals(), axis=1, keepdims=True)

    def calculate_barycentric_vertex_area(self):
        return self.vertex_face_adjacency()*self.calculate_face_areas()/3

    def calculate_vertex_normals(self):
        f_normals = self.calculate_face_normals()
        f_areas = self.calculate_face_areas()
        normed_by_area=f_normals*f_areas[:,np.newaxis]
        return self.vertex_face_adjacency()*normed_by_area

<<<<<<< HEAD
    #def edge_face_adjacency(self):

    """def gaussian_curvature(self):
        vertex_areas = self.calculate_barycentric_vertex_area()
        for col in self.vertex_face_adjacency().transpose():
            
            #print(col[1])
            #print("hi")"""

    def visualize_vertex_normals(self, normalized=True):
=======
    def gaussian_curvature(self,should_clip=False):
        L1 = np.linalg.norm(self.v[self.f[:,0]]-self.v[self.f[:,1]],axis=1)
        L2 = np.linalg.norm(self.v[self.f[:,0]]-self.v[self.f[:,2]],axis=1)
        L3 = np.linalg.norm(self.v[self.f[:,1]]-self.v[self.f[:,2]],axis=1)
        print("L1 shape is {}".format(L1.shape))
        cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
        cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
        cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)

        ang = np.arccos(np.column_stack((cos1, cos2, cos3)))
        angle_sum = np.asarray(self.vertex_face_adjacency(weight = ang.ravel()).sum(axis=1)).ravel()
        vertex_defect = (2 * np.pi) - angle_sum
        gaussian_curvature = vertex_defect/self.calculate_barycentric_vertex_area().flatten()
        if should_clip is True:
            mean = gaussian_curvature.mean()
            std = gaussian_curvature.std()
            gaussian_curvature = np.clip(gaussian_curvature, mean - 0.2*std, mean + 0.2*std)
        return gaussian_curvature

    def visualize_vertex_normals(self, normalized=True, mag=1):
>>>>>>> cb40f619c6bcf1563993ba620d84fdae2bf03ef3
        plotter=pv.Plotter()
        vectors = self.calculate_vertex_normals()
        #if normalized:
            #mag = mag/np.mean(np.linalg.norm(vectors, axis=1))
        self.mesh_poly.vectors = vectors
<<<<<<< HEAD
        plotter.add_mesh(self.mesh_poly.arrows, scalars = "GlyphScale")
=======
        plotter.add_arrows(np.array(self.v), vectors, mag=mag)
>>>>>>> cb40f619c6bcf1563993ba620d84fdae2bf03ef3
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
<<<<<<< HEAD
        plotter.show()

    
            
mesh =Mesh("sphere_s0.off")
#mesh.render_surface()
# print(len(mesh.f))
#print(mesh.gaussian_curvature())
#mesh.visualize_vertex_normals()
mesh.visualize_face_normals()
#mesh.calculate_vertex_centroid()

# mesh.render_surface(vert_deg)
=======
        plotter.show()    
>>>>>>> cb40f619c6bcf1563993ba620d84fdae2bf03ef3
