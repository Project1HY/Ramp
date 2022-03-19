import sys
import human_mesh_utils
import trimesh
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
sys.path.insert(0, '..')
from geom.mesh.vis.base import plot_mesh_montage


def main():
    print('a')

def test1():
    mesh_male=get_human_model(gender='male')
    mesh_female=get_human_model(gender='female')
    #mesh_male.ProjectionOld
    v=[mesh_male.vertices,mesh_female.vertices]
    f=[mesh_male.faces,mesh_female.faces]
    strategy = 'mesh'
    plot_mesh_montage(vb=v,fb=f,strategy=strategy)

def test2():
    plot_mesh_montage(vb=[get_neutral_humen_mesh_model_point_cloud()])

def get_human_model(gender:str,body_models_dict:dict=None,comp_device:str='cpu',
        rendering_params_list:list=None,render_root_orient:bool=True,
        render_trans:bool=False)->trimesh.Trimesh:
    assert(gender in human_mesh_utils.get_gender_list())
    if body_models_dict==None:
        body_models_dict=human_mesh_utils.get_body_models_dict(comp_device=comp_device)

    if rendering_params_list==None:rendering_params_list=human_mesh_utils.get_full_rendering_params_list()
    if not render_root_orient:rendering_params_list.remove('root_orient')
    if not render_trans:rendering_params_list.remove('trans')

    body_model=body_models_dict[gender]
    body_params=get_defult_body_params_for_body_model(body_model)
    body_posed=human_mesh_utils.update_body_model_with_body_params(body_model=body_model,
            body_parms=body_params,rendering_params_list=rendering_params_list)
    mesh=human_mesh_utils.get_body_mesh_of_single_frame(body_model=body_posed,frameID=0)
    return mesh


def get_defult_body_params_for_body_model(body_model:BodyModel)->dict:
    return {k:getattr(body_model,f'init_{k}') for k in human_mesh_utils.get_full_rendering_params_list()}

def get_neutral_humen_mesh_model_point_cloud()->np.array:
    return get_human_model(gender='neutral').vertices



if __name__=="__main__":
    test1()
    test2()

