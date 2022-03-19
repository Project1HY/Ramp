import sys

from human_mesh_utils import get_body_mesh_of_single_frame_from_npz_file,get_body_models_dict
from mapping_classes import Amass,Dataset,Actor,Animation
from save_load_obj import get_obj,load_obj,save_obj
from sampling_main import get_defult_values
import mapping_utils
import random
import sys
import os
import tqdm

sys.path.insert(0, '..')
from data.index import HierarchicalIndexTree
from geom.mesh.vis.base import plot_mesh_montage

def main():
    print('a')

def return_dict_for_animation(animation:Animation,num_projections:int)->dict():
    res=dict()
    for frame_num in range(animation.num_of_frames):
        res[frame_num]=num_projections
    return res

def return_dict_for_actor_full(actor:Actor,num_projections:int)->dict():
    res=dict()
    for animation in actor.animations:
        res[animation.animation_name]=return_dict_for_animation(animation=animation,num_projections=num_projections)
    return res

def return_reduced_dict_for_actor(actor:Actor,num_projections:int,index_list:list)->dict():
    res=dict()
    for index in index_list:
        animation_name,frameID=actor.get_animation_name_and_frameID_from_index(index)
        if animation_name not in res:
            res[animation_name]=dict()
        res[animation_name][frameID]=num_projections
    return res

def return_reduced_fps_dict_for_actor(actor:Actor,num_projections:int,num_of_frames_from_each_actor:int)->dict():
    reduced_fps_list=actor.fps_full_sampling[:num_of_frames_from_each_actor]
    return return_reduced_dict_for_actor(actor=actor,num_projections=num_projections,
            index_list=reduced_fps_list)

def return_reduced_random_dict_for_actor(actor:Actor,num_projections:int,num_of_frames_from_each_actor:int)->dict():
    random_list=random.sample(range(actor.num_of_frames),num_of_frames_from_each_actor)
    return return_reduced_dict_for_actor(actor=actor,num_projections=num_projections,
            index_list=random_list)

def return_dict_for_dataset(dataset:Dataset,num_projections:int,mode:str,num_of_frames_from_each_actor:int=None,take_only_first_n_actors:int=None)->dict():
    #num_of_frames_from_each_actor is hardcoded for now
    valid_modes_prefix=['full','reduced_random','reduced_fps']
    mode_start_with_prefix=False
    for prefix in valid_modes_prefix:
        if mode.startswith(prefix):
            mode_start_with_prefix=True
            break
    assert(mode_start_with_prefix)
    #return_dict_for_actor_func
    if mode.startswith('full'):
        assert(num_of_frames_from_each_actor==None)
        assert(take_only_first_n_actors==None)
        return_dict_for_actor_func=return_dict_for_actor_full
    elif mode.startswith('reduced_random'):
        #return_dict_for_actor_func=return_reduced_random_dict_for_actor
        return_dict_for_actor_func=lambda actor,num_projections : return_reduced_random_dict_for_actor(
                actor=actor,num_projections=num_projections,num_of_frames_from_each_actor=num_of_frames_from_each_actor)
    else: # mode=='reduced_fps'
        assert(mode.startswith('reduced_fps'))
        return_dict_for_actor_func=lambda actor,num_projections : return_reduced_fps_dict_for_actor(
                actor=actor,num_projections=num_projections,num_of_frames_from_each_actor=num_of_frames_from_each_actor)
    #assuming each actor can be taken from another dataset
    #do it in order to check on a dataset that contain many actors from many diffrent datasets
    res=dict()
    for i,actor in enumerate(dataset.actors):
        if take_only_first_n_actors!=None and i>=take_only_first_n_actors:
            break
        if actor.taken_from_dataset not in res.keys():
            res[actor.taken_from_dataset]=dict()
        res[actor.taken_from_dataset][actor.actor_name]=return_dict_for_actor_func(actor=actor,num_projections=num_projections)
    return res


def return_hit_for_dataset(dataset:Dataset,num_projections:int,mode:str,num_of_frames_from_each_actor:int=None,
        take_only_first_n_actors:int=None,in_memory:bool=False)->HierarchicalIndexTree:
    dataset_dict=return_dict_for_dataset(dataset=dataset,mode=mode,
            num_projections=num_projections,num_of_frames_from_each_actor=num_of_frames_from_each_actor,
            take_only_first_n_actors=take_only_first_n_actors)
    hit = HierarchicalIndexTree(dataset_dict, in_memory=in_memory)
    hit.init_double_cluster_hi_list()
    hit.init_triple_cluster_hi_list()
    return hit

"""
def return_fps_hit_from_hit_and_dataset(dataset:Dataset,hit:HierarchicalIndexTree,
        num_of_frames_from_each_actor:int,in_memory:bool=False)->HierarchicalIndexTree:
    res=dict()
    for actor in dataset.actors:
        partial_hi=(actor.taken_from_dataset,actor.actor_name)
        first_csi=hit.partial_hi2first_csi(partial_hi)
        csi_list=[first_csi+sample for sample in actor.fps_full_sampling[:num_of_frames_from_each_actor]]
        chi_list=[hit.csi2chi(csi=csi) for csi in csi_list]
        for dataset_name,actor_name,animation_name,frameID in chi_list:
            partial_hi=tuple([dataset_name,actor_name,animation_name,frameID])
            num_projections=hit.partial_hi2si_rank(partial_hi)
            #pretty ugly but i know
            if dataset_name not in res:
                res[dataset_name]=dict()
            if actor_name not in res[dataset_name]:
                res[dataset_name][actor_name]=dict()
            if animation_name not in res[dataset_name][actor_name]:
                res[dataset_name][actor_name][animation_name]=dict()
            #res[dataset_name][actor_name][animation_name] is a dict
            res[dataset_name][actor_name][animation_name][frameID]=num_projections
    hit = HierarchicalIndexTree(res, in_memory=in_memory)
    return hit
"""



#def path_2_data_full
def create_data_softlinks_tree(softlinks_dir_root:str,dataset:Dataset,hit:HierarchicalIndexTree)->None:
    print('create_data_softlinks_tree')
    if not os.path.exists(softlinks_dir_root):
        print('cannot create data softlinks tree')
        return
    for dcsi in tqdm.trange(hit.num_index_double_clusters()):
        dchi=hit.dcsi2dchi(dcsi)
        dataset_name,actor_name,animation_name=dchi
        symlink_path_dir=os.path.join(softlinks_dir_root,dataset_name,actor_name)
        symlink_path_full=os.path.join(symlink_path_dir,animation_name)
        if os.path.exists(symlink_path_full):
            continue
        if not os.path.exists(symlink_path_dir):
            os.makedirs(symlink_path_dir)
        src_animation_path=dataset.get_animation_path_by_dataset_actor_and_animation_names(taken_from_dataset_name=dataset_name,actor_name=actor_name,animation_name=animation_name)
        os.symlink(src=src_animation_path,dst=symlink_path_full)

def path2dataFullPOC(softlinks_dir_root:str,hit:HierarchicalIndexTree,csi:int,body_models_dict:dict=None,comp_device:str=None,plot:bool=False): # return v 
    #return vertecies
    if body_models_dict==None:
        body_models_dict=get_body_models_dict(comp_device=comp_device)
    if comp_device==None:
        comp_device='cpu'
    chi=hit.csi2chi(csi)
    dataset_name,actor_name,animation_name,frameID=chi
    symlink_path_full=os.path.join(softlinks_dir_root,dataset_name,actor_name,animation_name)
    mesh=get_body_mesh_of_single_frame_from_npz_file(model_npz=symlink_path_full,frameID=frameID,comp_device=comp_device,body_models_dict=body_models_dict)
    v=mesh.vertices
    if plot:
        f=mesh.faces
        plot_mesh_montage(vb=[v],fb=[f])
    return v

def path2dataFull_csiList(softlinks_dir_root:str,hit:HierarchicalIndexTree,csi_list:list,body_models_dict:dict=None,comp_device:str=None)->None:
    if body_models_dict==None:
        body_models_dict=get_body_models_dict(comp_device=comp_device)
    if comp_device==None:
        comp_device='cpu'
    #chi_list = [hit.csi2chi(csi) for csi in csi_list]
    v_list = []
    for csi in csi_list:
        v=path2dataFullPOC(softlinks_dir_root=softlinks_dir_root,hit=hit,csi=csi,body_models_dict=body_models_dict,comp_device=comp_device,plot=False)
        v_list.append(v)

    print('a')
    plot_mesh_montage(vb=v_list)
    print('a')

def path2dataFullFPS16SomeActor(softlinks_dir_root:str,dataset:Dataset,hit:HierarchicalIndexTree):
    actor=dataset.actors[5]
    partial_hi=(actor.taken_from_dataset,actor.actor_name)
    first_csi=hit.partial_hi2first_csi(partial_hi)
    csi_list=[first_csi+sample for sample in actor.fps_full_sampling[:16]]
    for csi in csi_list:
        chi=hit.csi2chi(csi=csi)
        #cur_partial_chi=tuple(list(chi)[:1])
        assert(chi[:2]==partial_hi[:2])
    path2dataFull_csiList(softlinks_dir_root=softlinks_dir_root,hit=hit,csi_list=csi_list)


def create_hit_for_dataset()->dict:
    defult_values=get_defult_values()
    #dataset_name='MPImosh_dataset_obj.pkl'
    #dataset_name='MPImosh_dataset_obj.pkl'
    dataset_name='mini_amass_dataset_obj.pkl' # assume dataset will have fps
    hit_dataset_name='hit_'+dataset_name
    default_dataset_object_obj_filepath=os.path.join(defult_values['cached_dataset_objects_dir'],dataset_name)
    default_hit_dataset_object_obj_filepath=os.path.join(defult_values['cached_hit_object_dir'],hit_dataset_name)
    #res['cached_hit_object_dir']=cached_dataset_objects_dir
    #res['soft_link_dir']=soft_link_dir
    #overwriteCache=False
    #assuming dataset object exisit
    dataset=load_obj(f_name=default_dataset_object_obj_filepath)
    hit=get_obj(obj_file_name=default_hit_dataset_object_obj_filepath,function_to_create_obj=lambda:return_hit_for_dataset(dataset=dataset,num_projections=10),overwriteCache=False) # hardcoded for now
    #softlinks_dir_root='./soft_link_dir' # hardcoded for now
    default_soft_link_dir=defult_values['soft_link_dir']
    create_data_softlinks_tree(softlinks_dir_root=default_soft_link_dir,dataset=dataset,hit=hit)
    print('a')

    #path2dataFullPOC(softlinks_dir_root=default_soft_link_dir,hit=hit,csi=1,comp_device='cpu')
    #path2dataFull_csiList(softlinks_dir_root=default_soft_link_dir,hit=hit,csi_list=list(range(1,800,100)))
    path2dataFullFPS16SomeActor(softlinks_dir_root=default_soft_link_dir,dataset=dataset,hit=hit)

    print('a')


if __name__=="__main__":
    create_hit_for_dataset()
    main()
