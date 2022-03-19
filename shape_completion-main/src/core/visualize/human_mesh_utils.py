import os
import torch
import sys
import re
import argparse
import cli_utils
import trimesh
import pickle
import numpy as np
from numpy.lib.npyio import NpzFile
import cv2
from tqdm import trange
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import colors
from sys import platform

def get_defult_dir_values()->dict():
    #get defult dir values and beats and dmpls num
    cached_dataset_objects_dir=os.path.join(os.curdir,'cached_objects')
    #smpl_segmentation_dir=os.path.join(os.curdir,'smpl_segmentations_data')
    smpl_segmentation_file=os.path.join('/','home','yiftach.ede','Ramp','shape_completion-main','src','core','visualize','smpl_segmentations_data','mixamo_smpl_segmentation.pkl')
    soft_link_dir=os.path.join(os.curdir,'soft_links')
    num_betas=16
    num_dmpls=8
    sampling_dir=os.path.join(os.curdir,'sampling')
    if platform == "linux" or platform == "linux2":
        amass_dir=os.path.join('/','media','omer','7B59-9DC0','Technion','Projects','ProjectB','AmassDataset','amass_dir') #defult folder for linux
        base_smplh_and_dmpls_dir=os.path.join('/','home','yiftach.ede','datasets','data') #defult folder for linux
        smplh_dir=os.path.join(base_smplh_and_dmpls_dir,'SLMPH/slmph/') #defult folder for linux
        dmpl_dir=os.path.join(base_smplh_and_dmpls_dir,'DMPL_DIR/dmpls/') #defult folder for linux
    else:
        amass_dir=os.path.join('D:/','Users','Omer','data','datasets','AMASS','tar','amass_dir')
        base_smplh_and_dmpls_dir=os.path.join('D:/','Users','Omer','data','prior_dirs')
        smplh_dir=os.path.join(base_smplh_and_dmpls_dir,'SLMPH','slmph') #defult folder for windwos
        dmpl_dir=os.path.join(base_smplh_and_dmpls_dir,'DMPL','dmpls') #defult folder for windwos
    res=dict()
    res['smplh_dir']=smplh_dir
    res['dmpl_dir']=dmpl_dir
    #res['smpl_segmentation_dir']=smpl_segmentation_dir
    res['smpl_segmentation_file']=smpl_segmentation_file
    res['amass_dir']=amass_dir
    res['sampling_dir']=sampling_dir
    res['cached_dataset_objects_dir']=cached_dataset_objects_dir
    res['cached_amass_object_dir']=cached_dataset_objects_dir
    res['cached_hit_object_dir']=cached_dataset_objects_dir
    res['soft_link_dir']=soft_link_dir
    res['num_betas']=num_betas
    res['num_dmpls']=num_dmpls
    return res

def assertVaildEnviroment()->None:
    errStr = ''
    if platform == "linux" or platform == "linux2":
        if os.environ['LANG']!='en_US': # importent for pyglet - inside MeshViewer
            errStr+='run this code with LANG=en_US enviroment varible.that\'s critical for MeshViewer packge.'
            errStr+='cur LANG value is:{}'.format(os.environ['LANG'])
    if errStr != '':
        raise Exception("unvalid enviroment:\n"+errStr)

def is_valid_render_list(args)->bool:
    return ((args.render_pose_body) or (args.render_pose_hand) or (args.render_betas) or (args.render_dmpls))

def is_valid_skmpl_and_dmpl_dirs(args)->(bool,str):
        errStr = ''
        genders = get_gender_list()
        models_dirs = {'smplh':args.smplh_dir,'dmpl':args.dmpl_dir}
        for model_type,_dir in models_dirs.items():
            for gender in genders:
                model_path = os.path.join(_dir,gender,'model.npz')
                if not os.path.exists(model_path):
                    errStr+='model prior files error:missing gender {} model of model type {}.please make sure that {} exists\n'.format(gender,model_type,_dir,model_path)
        isValid = True if errStr=='' else False # for readabilty
        return isValid,errStr

def is_valid_npz_file(model_npz:str,bdata:NpzFile=None)->bool:
    if not os.path.exists(model_npz):
        return False
    if bdata==None:bdata = np.load(model_npz)
    return ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']==list(bdata.keys())

def is_valid_npz_file_return_bdata_too(model_npz:str,bdata:NpzFile=None)->(bool,NpzFile):
    #some code duplication - I know
    if not os.path.exists(model_npz):
        return False,None
    if bdata==None:bdata = np.load(model_npz)
    return ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']==list(bdata.keys()),bdata

def get_body_mesh_of_single_frame(body_model:BodyModel,frameID:int):
    body_mesh = trimesh.Trimesh(process=False,vertices=c2c(body_model.v[frameID]), faces=c2c(body_model.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
    #body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
    #body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))
    return body_mesh

def get_number_of_frames(model_npz:str,bdata:NpzFile=None)->int:
    #bdata = np.load(model_npz)
    #bdata = np.load(model_npz) if bdata==None else bdata
    if bdata==None:bdata = np.load(model_npz) 
    return len(bdata['trans'])

def get_shape_params_list()->list:
    return ['betas']

def get_pose_params_list()->list:
    return ['trans', 'root_orient','pose_body', 'pose_hand', 'dmpls']

"""
def get_full_rendering_params_list_without_trans()->list:
    res=get_full_rendering_params_list()
    res.remove('trans')
    return res
"""

def get_full_rendering_params_list()->list:
    return get_shape_params_list() + get_pose_params_list()

def get_rendering_params_list(args)->list:
    res = []
    if args.render_pose_body:res.append('pose_body')
    if args.render_pose_hand:res.append('pose_hand')
    if args.render_betas:res.append('betas')
    if args.render_dmpls:res.append('dmpls')
    if args.render_root_orient:res.append('root_orient')
    assert res != []
    return res

def update_body_model_with_body_params(body_model:BodyModel,body_parms:dict,rendering_params_list:list=get_full_rendering_params_list())->BodyModel:
    try:
        updated_body_model = body_model(**{k:v for k,v in body_parms.items() if k in rendering_params_list})
    except:
        raise Exception("""update_body_model_with_body_params failed.
                maybe animation is too long! try to loader shorter one""")
    return updated_body_model

#TODO some code duplication here (on get_shape_body_params,get_pose_body_params...),can be optimized
def get_shape_body_params(model_npz:str,num_betas :int,comp_device:str,bdata:NpzFile=None)->dict:
    #bdata = np.load(model_npz)
    #bdata = np.load(model_npz) if bdata==None else bdata
    if bdata==None:bdata = np.load(model_npz) 
    shape_body_params= {
        'betas': torch.Tensor(bdata['betas'][:num_betas][np.newaxis]).to(comp_device), # controls the body shape. Body shape is static
    }
    return shape_body_params

def get_pose_body_params(model_npz:str,num_dmpls :int,comp_device:str,bdata:NpzFile=None)->dict:
    #bdata = np.load(model_npz)
    #bdata = np.load(model_npz) if bdata==None else bdata
    if bdata==None:bdata = np.load(model_npz) 
    pose_body_params= {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device),# controls soft tissue dynamics
    }
    return pose_body_params

def get_pose_body_and_hands_of_one_frame(model_npz:str,frameID :int,comp_device:str,bdata:NpzFile=None)->dict:
    #reduced version of get_pose_body_params for sampling
    #bdata = np.load(model_npz)
    #bdata = np.load(model_npz) if bdata==None else bdata
    if bdata==None:bdata = np.load(model_npz) 
    pose_body_params= {
        'pose_body': torch.Tensor(bdata['poses'][frameID, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][frameID, 66:]).to(comp_device), # controls the finger articulation
    }
    return pose_pose_body_and_hands_params

def get_body_params_from_pose_and_shape_params(shape_body_params:dict,pose_body_params:dict,comp_device:str)->dict:
    time_length = len(pose_body_params['trans'])
    new_shape_body_params= {
        'betas': torch.Tensor(np.repeat(shape_body_params['betas'], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    }
    body_parms = {**new_shape_body_params,**pose_body_params} #merge two dictionaries
    #TODO add function that copies the 
    return body_parms

def get_body_params(model_npz:str,num_betas :int,num_dmpls :int,comp_device:str,bdata:NpzFile=None)->dict:
    if bdata==None:bdata = np.load(model_npz)
    shape_body_params = get_shape_body_params(model_npz=model_npz,num_betas=num_betas,comp_device=comp_device,bdata=bdata)
    pose_body_params = get_pose_body_params(model_npz=model_npz,num_dmpls=num_dmpls,comp_device=comp_device,bdata=bdata)
    body_parms = get_body_params_from_pose_and_shape_params(shape_body_params=shape_body_params,pose_body_params=pose_body_params,comp_device=comp_device)
    return body_parms


def get_gender_from_model_file(model_npz:str,bdata:NpzFile=None)->str:
    #bdata = np.load(model_npz) if bdata==None else bdata
    if bdata==None:bdata = np.load(model_npz) 
    return bdata['gender'].item()

def get_gender_list()->list:
    return ['female','male','neutral']

def get_body_model(gender:str,smplh_dir:str, dmpl_dir:str,num_betas :int,num_dmpls :int,comp_device:str)->BodyModel:
    assert gender in get_gender_list()
    bm_fname = os.path.join(smplh_dir,gender,'model.npz')
    dmpl_fname = os.path.join(dmpl_dir,gender,'model.npz')
    body_model = BodyModel(bm_fname=bm_fname,num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
    return body_model

def get_body_pose(model_npz:str,num_betas :int,num_dmpls :int,force_gender:str,smplh_dir:str, dmpl_dir:str,rendering_params_list:list,comp_device:str)->BodyModel:
    body_parms=get_body_params(model_npz=model_npz,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=comp_device)
    gender = get_gender_from_model_file(model_npz=model_npz) if force_gender == 'no' else force_gender
    body_model= get_body_model(smplh_dir=smplh_dir, dmpl_dir=dmpl_dir,num_betas=num_betas,num_dmpls=num_dmpls,gender=gender,comp_device=comp_device)
    body_pose = update_body_model_with_body_params(body_model,body_parms,rendering_params_list)
    return body_pose

def render_body_mesh_all_frames_3D(body_pose:BodyModel,number_of_frames:int,f_name_pattern:str,show_output:bool=False,save_output:bool=True):
    for frameID in trange(number_of_frames):
        f_name = "{}{}.obj".format(f_name_pattern.rsplit( "X.obj", 1 )[ 0 ],frameID)
        render_body_mesh_single_frame_3D(body_pose=body_pose,frameID=frameID,show_output=show_output,save_output=save_output,f_name=f_name)

def render_body_mesh_single_frame_3D(body_pose:BodyModel,frameID:int,show_output:bool,save_output:bool,f_name:str):
    body_mesh = get_body_mesh_of_single_frame(body_model=body_pose,frameID=frameID)
    if show_output:
         body_mesh.show()
    if save_output and not os.path.exists(f_name):
         body_mesh.export(f_name,'obj')

def render_body_mesh_video(body_pose:BodyModel,number_of_frames:int,fps:int,f_name:str):
    imw, imh=1600, 1600
    size=(imw, imh)
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frameID in trange(number_of_frames):
        body_mesh = get_body_mesh_of_single_frame(body_pose,frameID)
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        out.write(body_image)
    out.release()

def get_body_models_dict(comp_device:str)->dict():
    body_models=dict()
    defult_values=get_defult_dir_values()
    smplh_dir=defult_values['smplh_dir']
    dmpl_dir=defult_values['dmpl_dir']
    num_betas=defult_values['num_betas']
    num_dmpls=defult_values['num_dmpls']
    for gender in get_gender_list():
        body_models[gender]=get_body_model(gender=gender,smplh_dir=smplh_dir, dmpl_dir=dmpl_dir,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=comp_device)
    return body_models

def get_body_params_and_gender_of_frameID_list(model_npz:str,frameID_list:list,comp_device:str,
        body_models_dict:dict=None,num_betas:int=get_defult_dir_values()['num_betas'],
        num_dmpls:int=get_defult_dir_values()['num_dmpls'],
        rendering_params_list:list=None,
        render_root_orient:bool=True,render_trans:bool=False,bdata:NpzFile=None)->(np.array,str):
    #is going to load npz file once.
    #some code diplication i know
    if bdata==None:
        is_valid_npz_file,bdata=is_valid_npz_file_return_bdata_too(model_npz=model_npz)
        assert(is_valid_npz_file)
    gender=get_gender_from_model_file(model_npz=model_npz,bdata=bdata)
    body_params=get_body_params(model_npz=model_npz,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=comp_device,bdata=bdata)
    if rendering_params_list==None:
        rendering_params_list=get_full_rendering_params_list()
    if not render_root_orient:
        rendering_params_list.remove('root_orient')
    if not render_trans:
        rendering_params_list.remove('trans')
    for k,v in body_params.items():
        body_params[k]=v[frameID_list,:]#.reshape(len(),-1)
    return body_params,gender


def get_body_mesh_of_single_frame_from_body_param_and_gender_single_frame(
        body_params_single_frame:dict,gender:str,comp_device:str,body_models_dict:dict=None,
        rendering_params_list:list=None,
        render_root_orient:bool=True,render_trans:bool=False):
    #some code diplication i know
    #body_params_single_frame=body_param_and_gender_single_frame_dict['body_params_single_frame']
    #gender=body_param_and_gender_single_frame_dict['gender']
    if body_models_dict==None:
        body_models_dict=get_body_models_dict(comp_device=comp_device)
    body_model=body_models_dict[gender]
    if rendering_params_list==None:
        rendering_params_list=get_full_rendering_params_list()
    if not render_root_orient:
        rendering_params_list.remove('root_orient')
    if not render_trans:
        rendering_params_list.remove('trans')
    frameID=0 #it's now 0 becouse we load a body model with a single frame
    body_posed=update_body_model_with_body_params(body_model=body_model,
            body_parms=body_params_single_frame,rendering_params_list=rendering_params_list)
    body_mesh=get_body_mesh_of_single_frame(body_model=body_posed,frameID=frameID)
    return body_mesh

def get_body_mesh_of_single_frame_from_npz_file(model_npz:str,frameID:int,comp_device:str,
        body_models_dict:dict=None,num_betas:int=None,num_dmpls:int=None,
        rendering_params_list:list=None,
        render_root_orient:bool=True,render_trans:bool=False,bdata:NpzFile=None):
    #if bdata==None we will going to load the model_npz file twice,in any other case we will load it once
    valid_comp_devices=['cpu','cuda']
    assert(comp_device in valid_comp_devices)
    if bdata==None:
        assert(is_valid_npz_file(model_npz=model_npz))
        bdata = np.load(model_npz)
    if body_models_dict==None:
        body_models_dict=get_body_models_dict(comp_device=comp_device)
    if rendering_params_list==None:
        rendering_params_list=get_full_rendering_params_list()
    num_betas=num_betas if num_betas!=None else get_defult_dir_values()['num_betas']
    num_dmpls=num_dmpls if num_dmpls!=None else get_defult_dir_values()['num_dmpls']
    gender=get_gender_from_model_file(model_npz=model_npz,bdata=bdata)
    body_model=body_models_dict[gender]
    body_params=get_body_params(model_npz=model_npz,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=comp_device,bdata=bdata)
    #['trans', 'root_orient','pose_body', 'pose_hand', 'dmpls']
    #can be optimzed a bit with hardcoded lists
    if not render_root_orient:
        rendering_params_list.remove('root_orient')
    if not render_trans:
        rendering_params_list.remove('trans')
    #insted of calling directly to update_body_model_with_body_params
    for k,v in body_params.items():
        body_params[k]=v[frameID,:].reshape(1,-1)
    frameID=0 #it's now 0 becouse we load a body model with a single frame
    body_posed=update_body_model_with_body_params(body_model=body_model,
            body_parms=body_params,rendering_params_list=rendering_params_list)
    body_mesh=get_body_mesh_of_single_frame(body_model=body_posed,frameID=frameID)
    return body_mesh


def compute_distanse_matrix_from_data_matrix(data_matrix:torch.Tensor)->torch.Tensor:
    """
    body_vec_dim=data_matrix.size(1)                              #it's V
    unsampled_frame_vec_1=data_matrix.reshape(-1,body_vec_dim,1)  #dim (N,V,1)
    unsampled_frame_vec_2=unsampled_frame_vec_1.transpose(0,2)    #dim (1,V,N)
    dist_matrix=unsampled_frame_vec_1-unsampled_frame_vec_2       #dim (N,V,N)
    dist_matrix=dist_matrix.norm(dim=1)                           #dim (N,N)
    body_vec_dim=data_matrix.size(1)                              #it's V
    """
    dist_matrix=torch.cdist(data_matrix,data_matrix,p=2)
    return dist_matrix

def covert_path_to_current_os(path:str)->str:
    sperator=os.sep
    if platform == "linux" or platform == "linux2":
        another_sep='\\\\'
    else: #windows path
        another_sep='/'
    return re.sub(another_sep,sperator,path)

def chdir_to_visuallize_folder():
    os.chdir("./src/core")
    vis_folder_name='visualize'
    if os.path.exists(os.path.join(os.curdir,vis_folder_name)):
        return #no need to chdir,already on right dir
    #bit hacky and very ugly, i know.hardcoded for now
    upward_tries=3
    for i in range(upward_tries):
        print(f"yiftach message: curdir is {os.curdir}, vis_folder_name {vis_folder_name}, join: {os.path.join(os.curdir,vis_folder_name)}")
        up_times=i+1
        prefix_upward=''
        for j in range(up_times):
            prefix_upward=os.path.join(prefix_upward,'..')
        path=os.path.join(prefix_upward,vis_folder_name)
        if os.path.exists(path):
            os.chdir(path)
            return
    raise ValueError(f'Could not find visualize folder')
