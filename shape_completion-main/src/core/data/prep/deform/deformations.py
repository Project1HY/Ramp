import numpy as np
from numpy import pi
from abc import ABC
import os

from util.fs import restart_program
from util.strings import purple_banner
from geom.mesh.op.cpu.base import box_center, vertex_adjacency
from geom.mesh.op.cpu.dist import vertex_dist
from geom.matrix.cpu import row_intersect
from geom.mesh.op.cpu.remesh import remove_vertices
from scipy.sparse.csgraph import connected_components
from geom.mesh.vis.base import plot_mesh,plot_mesh_montage

# ---------------------------------------------------------------------------------------------------------------------#
#                                                      TODO
# ---------------------------------------------------------------------------------------------------------------------#
# [1] More deformation ideas: https://www.voronator.com/ - but for that you'll need a volumetric
# module + upsample model, and it reduces spatial resolution due to smoothing
#
# [2] SemanticCuts - install support for cut_at as a function (telling with point to stop by some method)
# [3] SemanticCuts - install support fot src_vi as a function (telling which point to start the cut - i.e.,
# some descriptor?)
# [4] SemanticCuts -  fix trunc_to_mask as specified in the function
# ---------------------------------------------------------------------------------------------------------------------#
#                                                        Base
# ---------------------------------------------------------------------------------------------------------------------#
class DeformationFailureError(Exception):
    pass


class Deformation(ABC):
    def deform(self, v, f):
        # Must return list of dictionaries
        raise NotImplementedError

    def num_expected_deformations(self):
        # Must return int
        raise NotImplementedError

    def name(self, full=True):  # Default
        return self.__class__.__name__.lower()

    def reset(self):  # Default
        raise DeformationFailureError  # Presuming that a failure is not fixable

    def needs_validation(self):  # Default
        # Must return bool
        return False

    def shape_filename_given_shape_name(self, shape_name, idx):  # Default
        return f"{shape_name}_{idx}.npz"


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
# TODO - Install support for this tomorrow
# if len(mask) < self.DANGEROUS_MASK_LEN:
#     masks.append(None)
# else:
#     masks.append({'mask': mask, 'ele': (ele_i, ele_ang), 'azi': (azi_i, azi_ang)})


class SemanticCuts(Deformation):
    SMALLEST_COMP_ALLOWED = 300
    MINIMAL_COMP_TO_RETURN = 500
    SRC_VI_STR_OPT = ('max_x', 'min_x', 'max_y', 'min_y', 'max_z', 'min_z')

    def __init__(self, src_vi='rand_str', cut_at=(0.13, 0.217), dist_map='graph',
                 return_only_mask=True, num_cuts=1,remove_small_comps=True,direct_dist_mode = False):
        """
        :param src_vi: Explicit Vertex Index, 'rand_str', 'rand_vi' or one of SRC_VI_STR_OPT
        :param cut_at: Explicit number of knn OR range from where knn would be chosen at rand
        :param dist_map: As given in vertex_distance_matrix()
        """
        self.dist_map, self.src_vi, self.return_mask, self.num_cuts = dist_map, src_vi, return_only_mask, num_cuts
        self.cut_at,self.remove_small_comps,self.direct_dist_mode = cut_at,remove_small_comps,direct_dist_mode

    def num_expected_deformations(self):
        return self.num_cuts

    def deform(self, v, f):
        # Handle configurability options:
        res = []
        for _ in range(self.num_cuts):
            src_vi = self._find_source(v) if isinstance(self.src_vi, str) else self.src_vi
            if isinstance(self.cut_at, (tuple, list)):
                low, high = self.cut_at
                if self.direct_dist_mode:
                    cut_at = np.random.uniform(low=low,high=high)
                else:
                    if low < 1:  # TODO - loss ability to not cut at all, but meh
                        low, high = int(low * v.shape[0]), int(high * v.shape[0])
                    cut_at = np.random.randint(low=low, high=high)
            else:  # Presume integer/float
                cut_at = self.cut_at
                if not self.direct_dist_mode and cut_at < 1:
                    cut_at = int(cut_at * v.shape[0])

            res.append(self._cut_algorithm(v, f, src_vi, cut_at))
        return res

    def _cut_algorithm(self, v, f, src_vi, cut_at):
        # Find the nearest neighbors to src according to the metric:
        if self.direct_dist_mode:
            goners = np.where(vertex_dist(v, f, src_vi, self.dist_map) < cut_at)
        else:
            neighbors = np.argsort(vertex_dist(v, f, src_vi, self.dist_map))
            # Mark all the closest neighbors as dead
            goners = neighbors[:cut_at]
        # Destroy the goners, leaving a mesh with (hopefully) a large connected comps and possibly tiny connected comps
        v_dec, f_dec = remove_vertices(v, f, goners)

        if self.remove_small_comps:
            # Now remove any tiny connected components
            A = vertex_adjacency(v_dec, f_dec)
            n_comps, comp_labels = connected_components(A, directed=False, connection='weak', return_labels=True)
            # Count the comps by their labels
            bin_counts = np.bincount(comp_labels)
            goners = np.full((v_dec.shape[0],), False, dtype=bool)
            # For each comp, if it is smaller than the thershold, mark it for removal
            for label, count in enumerate(bin_counts):
                if count < self.SMALLEST_COMP_ALLOWED:
                    goners |= (comp_labels == label)
            # Compute the remaining vertices
            remaining = v_dec.shape[0] - np.sum(goners)
            # if the remaining component is too small, return failure
            if remaining < self.MINIMAL_COMP_TO_RETURN:
                return None
            # Remove the tiny components by vertex mask
            v_dec, _f_dec = remove_vertices(v_dec, f_dec, vi=goners)
        # Choose whether to return the actual mesh or just the remaining vertex indices
        # plot_mesh(v_dec,f_dec)
        if self.return_mask:
            # Compute it:
            _, mask = row_intersect(v, v_dec, assume_unique=False)
            # plot_mesh_montage([v_dec, v[mask, :]])
            # assert len(mask) == v_dec.shape[0]
            return {'mask': mask}
        else:
            return {'v': v_dec, 'f': f_dec}

    def _find_source(self, v):
        if self.src_vi == 'rand_str':
            src_vi = np.random.choice(self.SRC_VI_STR_OPT)
        elif self.src_vi == 'rand_vi':
            return np.random.randint(low=0, high=v.shape[0])
        else:
            src_vi = self.src_vi

        if src_vi == 'max_x':
            return np.argmax(v[:, 0])
        elif src_vi == 'min_x':
            return np.argmin(v[:, 0])
        elif src_vi == 'max_y':
            return np.argmax(v[:, 1])
        elif src_vi == 'min_y':
            return np.argmin(v[:, 1])
        elif src_vi == 'max_z':
            return np.argmax(v[:, 2])
        elif src_vi == 'min_z':
            return np.argmin(v[:, 2])
        else:
            raise NotImplementedError(f'Unknown source find method: {src_vi}')


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
class Projection(Deformation):
    DANGEROUS_MASK_LEN = 100

    def __init__(self, n_azimutals=10, n_azimutal_subset=None, n_elevations=10, n_elevation_subset=None, r=2.5):
        from data.prep.collaterals.external_tools.pyrender.lib import render
        self.render = render
        assert n_elevations % 2 == 1, "Must input an odd number of elevation angles for backwards compatibility"
        self.n_azimutals, self.n_elevations, self.r = n_azimutals, n_elevations, r

        if n_azimutal_subset is None:
            n_azimutal_subset = self.n_azimutals
        else:
            assert 1 <= n_azimutal_subset <= n_azimutals
        self.n_azimutal_subset = n_azimutal_subset

        if n_elevation_subset is None:
            n_elevation_subset = self.n_elevations
        else:
            assert 1 <= n_elevation_subset <= n_elevation_subset
        self.n_elevation_subset = n_elevation_subset

        self.azimutal_range = np.linspace(0, 2 * pi, self.n_azimutals, endpoint=False)
        self.elevation_range = np.linspace(-pi / 2, pi / 2, n_elevations + 2)[1:-1]
        self.azimutal_range_i = np.arange(self.n_azimutals)
        self.elevation_range_i = np.arange(self.n_elevations)

        assert len(self.elevation_range) == self.n_elevations

        # Needed by Render
        self.render_info = {'Height': int(480), 'Width': int(640), 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}
        if os.name != 'nt':
            self.render.setup(self.render_info)
        self.world2cam_mats = self._prep_world2cam_mats()

    def deform(self, v, f):

        v, f = v.astype('float32'), f.astype('int32')
        v = box_center(v)

        # Randomly truncate the angles:
        if self.n_elevation_subset == self.n_elevations:
            ele_angs_i, ele_angs = self.elevation_range_i, self.elevation_range
        else:
            ele_angs_i = np.random.choice(self.elevation_range_i, size=self.n_elevation_subset, replace=False).sort()
            ele_angs = self.elevation_range[ele_angs_i]

        if self.n_azimutal_subset == self.n_azimutals:
            azi_angs_i, azi_angs = self.azimutal_range_i, self.azimutal_range
        else:
            azi_angs_i = np.random.choice(self.azimutal_range_i, size=self.n_azimutal_subset, replace=False).sort()
            azi_angs = self.azimutal_range[azi_angs_i]

        masks = []
        for ele_i, ele_ang in zip(ele_angs_i, ele_angs):
            for azi_i, azi_ang in zip(azi_angs_i, azi_angs):
                # render.setup(self.render_info)
                context = self.render.set_mesh(v, f)
                self.render.render(context, self.world2cam_mats[ele_i][azi_i])
                mask, _, _ = self.render.get_vmap(context, self.render_info)  # vindices, vweights, findices
                mask = np.unique(mask)
                # Sanity:
                if len(mask) < self.DANGEROUS_MASK_LEN:
                    masks.append(None)
                else:
                    masks.append({'mask': mask, 'ele': (ele_i, ele_ang), 'azi': (azi_i, azi_ang)})

            # render.clear()

        return masks

    def name(self, full=True):
        if full:
            return f'{super().name()}_azi_{self.n_azimutal_subset}_' \
                   f'{self.n_azimutals}_ele_{self.n_elevation_subset}_{self.n_elevations}'
        else:
            return super().name()

    def num_expected_deformations(self):
        return self.n_azimutal_subset * self.n_elevation_subset

    def reset(self):
        purple_banner('RESTARTING')
        restart_program()
        # render.reset()
        # render.setup(self.render_info)

    def needs_validation(self):
        return True

    def _prep_world2cam_mats(self):

        # cam2world = np.array([[0.85408425, 0.31617427, -0.375678, 0.56351697 * 2],
        #                      [0., -0.72227067, -0.60786998, 0.91180497 * 2],
        #                      [-0.52013469, 0.51917219, -0.61688, 0.92532003 * 2],
        #                      [0., 0., 0., 1.]], dtype=np.float32)
        # cam2world = np.array([[-1, 0, 0, 0],
        #                      [0., 1, 0, 0],
        #                      [0, 0, -1, 20],
        #                      [0., 0., 0., 1.]], dtype=np.float32)

        # rotate the mesh elevation by 30 degrees
        # Rx = np.array([[1, 0, 0, 0],
        #                [0., np.cos(self.elevation_ang), -np.sin(self.elevation_ang), 0],
        #                [0, np.sin(self.elevation_ang), np.cos(self.elevation_ang), 0],
        #                [0., 0., 0., 1.]], dtype=np.float32)
        # # Rz = np.array([[np.cos(self.elevation_ang), -np.sin(self.elevation_ang), 0, 0],
        # #                [np.sin(self.elevation_ang), np.cos(self.elevation_ang), 0,0],
        # #                [0, 0,1,0],
        # #                [0, 0, 0, 1]], dtype=np.float32)
        # cam2world = np.matmul(Rx, cam2world)

        world2cam_mats = []
        for elevation_ang in self.elevation_range:
            cos_ang, sin_ang = np.cos(elevation_ang), np.sin(elevation_ang)
            # Rotate around the +X axis
            cam2world = np.array([[1, 0, 0, 0],
                                  [0., -cos_ang, -sin_ang, self.r * sin_ang],
                                  [0, sin_ang, -cos_ang, self.r * cos_ang],
                                  [0., 0., 0., 1.]], dtype=np.float32)

            ele_world2cam_mats = []
            for azimutal_ang in self.azimutal_range:
                cos_ang, sin_ang = np.cos(azimutal_ang), np.sin(azimutal_ang)
                Ry = np.array([[cos_ang, 0, -sin_ang, 0],
                               [0., 1, 0, 0],
                               [sin_ang, 0, cos_ang, 0],
                               [0., 0., 0., 1.]], dtype=np.float32)
                world2cam = np.linalg.inv(np.matmul(Ry, cam2world)).astype('float32')
                ele_world2cam_mats.append(world2cam)

            world2cam_mats.append(ele_world2cam_mats)
        return world2cam_mats


# ---------------------------------------------------------------------------------------------------------------------#
#                                                Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

def _semantic_cuts_tester():
    from geom.mesh.io.base import read_mesh
    from geom.mesh.vis.base import plot_mesh_montage
    from cfg import TEST_MESH_HAND_PATH, TEST_MESH_HUMAN_PATH
    v1, f1 = read_mesh(TEST_MESH_HUMAN_PATH)
    src_meths = ['max_x', 'min_x', 'max_y', 'min_y', 'max_z', 'min_z']
    for dist_meth in ['euclidean_graph', 'graph', 'euclidean_cloud']:
        res_v, res_f = [], []
        for src_meth in src_meths:
            d = SemanticCuts(src_vi='rand_str', dist_map=dist_meth, cut_at=[0.13, 0.217], num_cuts=1)
            # v_res, f_res = d.deform(v1, f1)
            mask = d.deform(v1, f1)
            mask = mask[0]['mask']
            res_v.append(v1[mask, :])
            # print(v1.shape[0] - len(mask))
            # res_v.append(v_res)
            # res_f.append(f_res)
        plot_mesh_montage(res_v, None, strategy='spheres', lighting=True, grid_on=True, labelb=src_meths)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    _semantic_cuts_tester()
