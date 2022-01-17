from off_parser import read_off, write_off
from scipy.sparse import csr_matrix, identity
import numpy as np
import pyvista as pv
import matplotlib.colors
import sklearn.preprocessing
from typing import Union
import pickle as pkl
import PIL
import matplotlib.pyplot as plt

class Mesh(object):
    def __init__(self, v):
        # self.f=f
        self.v = v
        # self.f = np.concatenate((np.ones((self.f.shape[0], 1)), self.f), axis=1)
        self.mesh_poly = pv.PolyData(self.v)
        # self.f = vertices_faces[1][:, 1:]

    def vertex_face_adjacency(self, weight: Union[str, None, np.ndarray] = None):
        row = self.f.ravel()  # Flatten indices
        col = np.repeat(np.arange(len(self.f)), 3)  # Data for vertices        
        if weight is None:
            weight = np.ones(len(col), dtype=np.bool)
        print(weight.shape)
        vf = csr_matrix((weight, (row, col)), shape=(self.v.shape[0], len(self.f)), dtype=weight.dtype)
        return vf

    def vertex_vertex_adjacency(self):
        v_f_adjacency = self.vertex_face_adjacency()
        return (v_f_adjacency * v_f_adjacency.transpose() - identity(len(self.v))) > 0

    def vertex_degree(self):
        v_v_adjecency = self.vertex_vertex_adjacency()
        return np.sum(v_v_adjecency, axis=1)

    def render_wireframe(self):
        pv.plot(self.mesh_poly, style="wireframe")

    def render_pointcloud(self, color_func=None):
        # if color_func == None:
        # color_func=np.random.rand(len(self.f))
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_position = ((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0))

        plotter.add_mesh(self.mesh_poly, style="points", point_size=2, render_points_as_spheres=True,
                         scalars=color_func)

        plotter.show(screenshot='airplane.png')
        image = plotter.image
        plt.imshow(image)
        plt.show()
        img=PIL.Image.fromarray(image)

    def render_surface(self, color_func=None, **kwargs):
        plotter = pv.Plotter()
        if color_func == None:
            color_func = np.random.rand(len(self.f))
        if "cmap" in kwargs:
            plotter.add_mesh(self.mesh_poly, cmap=kwargs["cmap"], scalars=color_func)
        else:
            plotter.add_mesh(self.mesh_poly, scalars=color_func)
        plotter.show()

    def calculate_face_normals(self):
        vec1 = self.v[self.f[:, 0]] - self.v[self.f[:, 1]]
        vec2 = self.v[self.f[:, 0]] - self.v[self.f[:, 2]]
        norm = np.cross(vec1, vec2)
        return norm

    def calculate_face_barycenters(self):
        vertex = np.array(self.v)
        return self.vertex_face_adjacency().transpose() * vertex / 3

    def calculate_face_areas(self):
        return np.linalg.norm(self.calculate_face_normals(), axis=1, keepdims=True)

    def calculate_barycentric_vertex_area(self):
        return self.vertex_face_adjacency() * self.calculate_face_areas() / 3

    def calculate_vertex_normals(self):
        f_normals = self.calculate_face_normals()
        f_areas = self.calculate_face_areas()
        normed_by_area = f_normals * f_areas[:, np.newaxis]
        return self.vertex_face_adjacency() * normed_by_area

    def gaussian_curvature(self, should_clip=False):
        L1 = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 1]], axis=1)
        L2 = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 2]], axis=1)
        L3 = np.linalg.norm(self.v[self.f[:, 1]] - self.v[self.f[:, 2]], axis=1)
        print("L1 shape is {}".format(L1.shape))
        cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
        cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
        cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)

        ang = np.arccos(np.column_stack((cos1, cos2, cos3)))
        angle_sum = np.asarray(self.vertex_face_adjacency(weight=ang.ravel()).sum(axis=1)).ravel()
        vertex_defect = (2 * np.pi) - angle_sum
        gaussian_curvature = vertex_defect / self.calculate_barycentric_vertex_area().flatten()
        if should_clip is True:
            mean = gaussian_curvature.mean()
            std = gaussian_curvature.std()
            gaussian_curvature = np.clip(gaussian_curvature, mean - 0.2 * std, mean + 0.2 * std)
        return gaussian_curvature

    def visualize_vertex_normals(self, normalized=True, mag=1):
        plotter = pv.Plotter()
        vectors = self.calculate_vertex_normals()
        if normalized:
            mag = mag / np.mean(np.linalg.norm(vectors, axis=1))
        self.mesh_poly.vectors = vectors
        plotter.add_arrows(np.array(self.v), vectors, mag=mag)
        plotter.show()

    def visualize_face_normals(self, normalized=True, mag=1):
        plotter = pv.Plotter()
        vectors = self.calculate_face_normals()
        if normalized:
            mag = mag / np.mean(np.linalg.norm(vectors, axis=1))
        plotter.add_arrows(self.calculate_face_barycenters(), vectors, mag=mag)
        plotter.show()

    def calculate_vertex_centroid(self):
        vertices = np.array(self.v)
        mean_vertices = np.mean(vertices, axis=0)
        scalar_func = np.linalg.norm(vertices - mean_vertices, axis=1)
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh_poly, style="points", point_size=10, render_points_as_spheres=True,
                         scalars=scalar_func)
        plotter.add_points(mean_vertices, render_points_as_spheres=True, point_size=20, color='red')
        plotter.show()


v = np.load("/Users/yiftachedelstain/Development/Technion/Project/shape_completion/index/00000.npy")
with open("/Users/yiftachedelstain/Development/Technion/Project/shape_completion/index/face_template.pkl",
          "rb") as file:
    f = pkl.load(file)
m = Mesh(v)
color = np.ones((v.shape[0]))
with open("/Users/yiftachedelstain/Downloads/shape_completion-main/src/visualize/smpl_segmentations_data/mixamo_smpl_segmentation.pkl","rb") as file:
    seg = pkl.load(file)
for i in range(20):
    color[seg[list(seg.keys())[i]]] *= (i+1)*2
m.render_pointcloud(color_func=color)



