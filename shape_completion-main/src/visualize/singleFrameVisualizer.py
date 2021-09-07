import os
import torch
import argparse
import utils
import trimesh
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--frame", help="frame number to render", type=int, default=0)
    parser.add_argument("--smplh_dir", help="path to the smplh directory", type=str, default='/home/omer/PycharmProjects/shape_completion_dataset/data/SLMPH/SLMPH/')
    parser.add_argument("--dmpl_dir", help="path to the dmpl directory", type=str, default='/home/omer/PycharmProjects/shape_completion_dataset/data/DMPL_DIR/')

    parser.add_argument("--num_betas", help="number of body parameters", type=int, default=16)
    parser.add_argument("--num_dmpls", help="number of DMPL parameters", type=int, default=8)
    parser.add_argument("--force_gender", help="choose whether to force the gender of the model", type=str, default='no',choices={'no','female','male','neutral'})

    """
    parser.add_argument("-m","--model_npz", help="path to the the model *.npz file path", type=str,required = True ,
            default='/home/omer/PycharmProjects/shape_completion_dataset/data/amass_dir/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz') #FIXME remove the default value after the building of this
    """

    parser.add_argument("-m","--model_npz", help="path to the the model *.npz file path", type=str,
            default='/home/omer/PycharmProjects/shape_completion_dataset/data/amass_dir/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz') #FIXME remove the default value after the building of this

    parser.add_argument("--render_pose_body", help="choose whether to render the pose body of the model (true/false)", type=utils.str_to_bool, default=True)
    parser.add_argument("--render_pose_hand", help="choose whether to render the pose hands of the model (true/false)", type=utils.str_to_bool, default=True)
    parser.add_argument("--render_betas", help="choose whether to render the pose hands of the model (true/false)", type=utils.str_to_bool, default=True)
    parser.add_argument("--render_dmpls", help="choose whether to render the dmpls of the model (true/false)", type=utils.str_to_bool, default=True)
    parser.add_argument("--render_root_orient", help="choose whether to render the rotation orientation of the model (true/false)", type=utils.str_to_bool, default=False)
    parser.add_argument("--render_trans", help="choose whether to render the translation of the model (true/false)", type=utils.str_to_bool, default=False)

    parser.add_argument("-s","--save_output", help="choose whether to save output (true/false)", type=utils.str_to_bool, default=False)
    parser.add_argument("-o","--open_output_window", help="choose whether to open the rendered model on separate window(true/false)", type=utils.str_to_bool, default=True)
    parser.add_argument("--check_if_sure", help="check if the user is sure about all the arguments", type=utils.str_to_bool, default=True)
    args = parser.parse_args()
    return args



def assertVaildInput(args)->None:
    errStr = ''
    # make sure we have something to render
    if not ((args.render_pose_body) or (args.render_pose_hand) or (args.render_betas) or (args.render_dmpls)):
        errStr+='nothing to render.\n'
    # make sure skmplh and dmpl paths are ok
    genders = ['female','male','neutral']
    models_dirs = {'smplh':args.smplh_dir,'dmpl':args.dmpl_dir}
    for model_type,_dir in models_dirs.items():
        for gender in genders:
            model_path = os.path.join(_dir,gender,'model.npz')
            if not os.path.exists(model_path):
                errStr+='model prior files error:missing gender {} model of model type {}.please make sure that {} exists\n'.format(gender,model_type,_dir,model_path)
    if not os.path.exists(args.model_npz):
                errStr+='model file {} not exists'.format(args.model_npz)
    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)

def assertVaildEnviroment()->None:
    errStr = ''
    if os.environ['LANG']!='en_US': # importent for pyglet - inside MeshViewer
        errStr+='run this code with LANG=en_US enviroment varible.that\'s critical for MeshViewer packge'
    if errStr != '':
        raise Exception("unvalid enviroment:\n"+errStr)

def assertVaild(args)->None:
    assertVaildEnviroment()
    assertVaildInput(args)

def get_gender(model_npz:str,force_gender:str)->str:
    assert force_gender in ['no','female','male','neutral']
    if force_gender is not 'no':
        return force_gender
    return np.load(model_npz)['gender'].item()

def get_body_params(model_npz:str,num_betas :int,num_dmpls :int)->dict:
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdata = np.load(model_npz)
    time_length = len(bdata['trans'])
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }
    return body_parms

def get_body_model(smplh_dir:str, dmpl_dir:str,num_betas :int,num_dmpls :int,gender:str)->BodyModel:
    assert gender in ['female','male','neutral']
    bm_fname = os.path.join(smplh_dir,gender,'model.npz')
    dmpl_fname = os.path.join(dmpl_dir,gender,'model.npz')
    body_model = BodyModel(bm_fname=bm_fname,num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)
    return body_model

def get_rendering_params_list(args)->list:
    res = []
    if args.render_pose_body:res.append('pose_body')
    if args.render_pose_hand:res.append('pose_hand')
    if args.render_betas:res.append('betas')
    if args.render_dmpls:res.append('dmpls')
    if args.render_root_orient:res.append('root_orient')
    if args.render_trans:res.append('trans')
    assert res != []
    return res


def render_body_mesh(body_model:BodyModel,body_parms:dict,rendering_params_list:list,frameID:int=0):
    body_pose = body_model(**{k:v for k,v in body_parms.items() if k in rendering_params_list})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[frameID]), faces=c2c(body_model.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
    body_mesh.show()


def main(args):
    body_parms=get_body_params(model_npz=args.model_npz,num_betas=args.num_betas,num_dmpls=args.num_dmpls)
    gender = get_gender(model_npz=args.model_npz,force_gender=args.force_gender)
    body_model= get_body_model(smplh_dir=args.smplh_dir, dmpl_dir=args.dmpl_dir,num_betas=args.num_betas,num_dmpls=args.num_dmpls,gender=gender)
    render_body_mesh(body_model=body_model,body_parms=body_parms,rendering_params_list=get_rendering_params_list(args),frameID=0)

    print(get_rendering_params_list(args))
    print('a')
    print('b')

if __name__ == '__main__':
    args = parse_args()
    if args.check_if_sure and not utils.userIsSure(args):
        exit()
    assertVaild(args)
    main(args)
