from human_mesh_utils import get_body_mesh_of_single_frame_from_npz_file
from human_mesh_utils import get_body_models_dict

import sys
import trimesh
import numpy as np
import os
sys.path.insert(0, '..')
from core.geom.mesh.vis.base import plot_mesh_montage
#from core.data.prep.deform.projection import Projection
from core.data.prep.deform.projection import ProjectionOld,ProjectionCached
#from core.data.prep.deform.projection import ProjectionOld,ProjectionCached

base_dataset_dir = '/home/yiftach.ede/datasets/'
def main():
    mesh=get_sample_mesh()
    vb=[mesh.vertices]
    fb=[mesh.faces]
    strategy='mesh' #can be wireframe or spheres
    print('a')
    #plot_mesh_montage(vb=vb,fb=fb,strategy=strategy)
    #deform_mesh_hardcoded(mesh=mesh)
    deform_mesh_hardcoded2(mesh=mesh)

def get_sample_mesh():
    model_npz=os.path.join(base_dataset_dir,'data/amass_dir/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz')
    frameID=200
    comp_device='cpu'
    body_models_dict=get_body_models_dict(comp_device=comp_device)
    mesh=get_body_mesh_of_single_frame_from_npz_file(model_npz=model_npz,frameID=frameID,comp_device='cpu',body_models_dict=body_models_dict,
            render_root_orient=True,render_trans=False)
    return mesh

def deform_mesh_hardcoded(mesh:trimesh.Trimesh):
    d = ProjectionOld(n_azimuthals=3, n_azimuthal_subset=None, n_elevations=3, n_elevation_subset=None, r=20)
    v=mesh.vertices
    f=mesh.faces
    res_v = [v[dct['mask'], :] for dct in d.deform(v, f)]
    strategy='spheres' #can be wireframe or spheres
    plot_mesh_montage(vb=res_v,strategy=strategy)


def deform_mesh_hardcoded2(mesh:trimesh.Trimesh):
    d = ProjectionCached(n_azimuthals=10, n_azimuthal_subset=None, n_elevations=1, n_elevation_subset=None, r=3)
    #d.set_minimum_mask_len(100)
    num_expected=d.num_expected_deformations()
    v=mesh.vertices
    res_v=[]
    for n in range(0,num_expected,1):
        dct=d.deform(v=v,n=n)
        res_v.append(v[dct['mask'], :])
    res_v.append(np.array(d.get_all_cameras_points())) # only for debug purpuse
    strategy='spheres' #can be wireframe or spheres
    plot_mesh_montage(vb=res_v,strategy=strategy)

if __name__=="__main__":
    main()
