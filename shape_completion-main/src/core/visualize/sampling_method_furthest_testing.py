#Quick and dirty file to test sampling
import os
import torch
import mapping_utils
import tqdm
import random
import trimesh
from save_load_obj import load_obj,get_obj
from mapping_classes import Dataset,Actor
import sampling_main
import sampling
import sampling_method_furthest
import sys
import human_mesh_utils

sys.path.insert(0, '..')
from core.geom.mesh.vis.base import plot_mesh_montage

#hardcoded for now
num_betas=16
num_dmpls=8

def get_testing_values()->dict(): # maybe make the code stndalone with this dict without the use of 'get_defult_values'
    testing_values=dict()
    #testing_values['mini_dataset_name']='TCDhandMocap' #'EKUT' # change this
    #testing_values['mini_dataset_name']='SFU' #'EKUT' # change this
    testing_values['mini_dataset_name']='MPImosh'
    testing_values['theta_weights']=get_theta_weights_path()
    testing_values['furtherst_sampling_permutation_root_dir']=get_permutation_results_path()
    #testing_values['comp_device']=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testing_values['comp_device']='cpu'
    return testing_values

def get_permutation_results_path()->str:
    furtherst_sampling_permutation_dir_name='furtherst_sampling_permutation_dir'
    full_file_path=os.path.join(os.curdir,furtherst_sampling_permutation_dir_name)
    return full_file_path

def get_theta_weights_path()->str:
    theta_weights_gender='neutral'
    ##theta_weights_f_name='theta_params_for_{}.pkl'.format(theta_weights_gender)
    theta_dir_name='theta_vector_dir'
    fileName='theta_params_for_{}.pkl'.format(theta_weights_gender)
    full_file_path=os.path.join(os.curdir,theta_dir_name,fileName)
    return full_file_path

def create_amass_obj_if_needed():
    defult_values=sampling_main.get_defult_values()
    default_amass_obj_filepath=os.path.join(defult_values['cached_amass_object_dir'],'amass_obj.pkl') # not so buitiful I know
    sampling.save_amass_object(input_amass_dir=defult_values['amass_dir'],
            output_amass_obj_full_filepath=default_amass_obj_filepath,overwriteCache=defult_values['overwrite_cache'])

def get_mini_dataset_obj_full_path(mini_dataset_name:str):
    defult_values=sampling_main.get_defult_values()
    output_object_filename='{}_dataset_obj.pkl'.format(mini_dataset_name)
    out_file=os.path.join(defult_values['cached_dataset_objects_dir'],'{}_dataset_obj.pkl'.format(mini_dataset_name))
    return out_file

def create_dataset_obj_if_needed(mini_dataset_name:str):
    defult_values=sampling_main.get_defult_values()
    out_file=get_mini_dataset_obj_full_path(mini_dataset_name=mini_dataset_name)
    input_amass_obj_full_filepath=os.path.join(defult_values['cached_amass_object_dir'],'amass_obj.pkl') # not so buitiful I know
    sampling.save_internal_dataset_in_amass_obj(input_amass_obj_full_filepath=input_amass_obj_full_filepath,
            req_dataset_name=mini_dataset_name,
            output_dataset_full_filepath=out_file,overwriteCache=defult_values['overwrite_cache'],add_fps_full_sampling=False)

def get_dataset_obj(mini_dataset_name:str)->Dataset:
    dataset_obj_file_name=get_mini_dataset_obj_full_path(mini_dataset_name=mini_dataset_name)
    dataset=load_obj(f_name=dataset_obj_file_name)
    return dataset

def get_theta_weights():
    path_to_theta_vector=get_theta_weights_path()
    assert(os.path.exists(path_to_theta_vector))
    theta_weights=load_obj(path_to_theta_vector)
    return theta_weights

def init_random_seed()->None:
    seed=1
    random.seed(seed)

def init_testing_env(mini_dataset_name:str)->None:
    print('init_testing_env')
    print('create_amass_obj_if_needed')
    create_amass_obj_if_needed()
    print('create_dataset_obj_if_needed')
    create_dataset_obj_if_needed(mini_dataset_name=mini_dataset_name)

def get_valid_farthest_sampling_methods()->list:
    valid_farthest_sampling_methods=['vectors_loop','distance_matrix']
    return valid_farthest_sampling_methods

def get_sample_dataset_furtherst_method(dataset:Dataset,farthest_sampling_method:str,
        theta_weights:torch.Tensor=None,num_of_frames_to_sample:int=50,
        only_allow_samples_that_diveded_by:int=1,iterations_per_sample:int=1,
        comp_device:str='cpu')->list:
    assert(farthest_sampling_method in get_valid_farthest_sampling_methods())
    sampling=sampling_method_furthest.farthest_sampling_per_actor(dataset=dataset,
            num_of_frames_to_sample=num_of_frames_to_sample,
            comp_device=comp_device,iterations_per_sample=iterations_per_sample,
            only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
            farthest_sampling_method=farthest_sampling_method,theta_weights=theta_weights)
    return sampling

def get_sample_actor_furtherst_method(dataset:Dataset,
        actor_name:str,num_of_frames_to_sample:int, farthest_sampling_method:str,
        theta_weights:torch.Tensor=None, only_allow_samples_that_diveded_by:int=1,
        iterations_per_sample:int=1, comp_device:str='cpu')->list:
    #bit code duplication with get_sample_dataset_furtherst_method
    assert(farthest_sampling_method in get_valid_farthest_sampling_methods())
    init_random_seed()
    sampling=sampling_method_furthest.farthest_sampling_by_actor_name(dataset=dataset,
            num_of_frames_to_sample=num_of_frames_to_sample,actor_name=actor_name,
            comp_device=comp_device,iterations_per_sample=iterations_per_sample,
            only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
            farthest_sampling_method=farthest_sampling_method,theta_weights=theta_weights)
    return sampling

def get_actor_furtherst_sampling_method_permutation(dataset:Dataset,
        actor_name:str,farthest_sampling_method:str,save_to_file_name:str,overwriteCache:bool,
        comp_device:str, theta_weights:torch.Tensor=None, iterations_per_sample:int=1)->list:
    # sample all actor frames permutation
    only_allow_samples_that_diveded_by=1
    num_of_frames_to_sample=dataset.get_actor_obj_by_name(req_actor_name=actor_name).num_of_frames//only_allow_samples_that_diveded_by
    sample_func=lambda: get_sample_actor_furtherst_method(dataset=dataset, actor_name=actor_name,num_of_frames_to_sample=num_of_frames_to_sample, farthest_sampling_method=farthest_sampling_method, theta_weights=theta_weights, only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by, iterations_per_sample=iterations_per_sample, comp_device=comp_device)
    return get_obj(obj_file_name=save_to_file_name,function_to_create_obj=sample_func,overwriteCache=overwriteCache)

def test_1_sample_furtherst_methods_compare(dataset:Dataset):
    defult_values=sampling_main.get_defult_values()
    print('test_1_sample_furtherst_methods_compare')
    only_allow_samples_that_diveded_by=50
    num_of_frames_to_sample=400
    res=[]
    for farthest_sampling_method in get_valid_farthest_sampling_methods():
        init_random_seed()
        sampling=get_sample_dataset_furtherst_method(dataset=dataset,
                farthest_sampling_method=farthest_sampling_method,
                only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
                num_of_frames_to_sample=num_of_frames_to_sample, theta_weights=None)
        res.append(sampling)
    assert(res[0]==res[1])
    print('test 1 pass!')

def get_mesh_from_frame_num(dataset:Dataset,body_models_dict:dict,frame_num:int,comp_device:str)->trimesh.Trimesh:
    assert(list(body_models_dict.keys())==human_mesh_utils.get_gender_list())
    #conside move this function to another file
    cur_frame=dataset.get_frame_by_number(num=frame_num)
    frameID=cur_frame.get_frame()
    model_npz=cur_frame.get_file_path()
    gender=human_mesh_utils.get_gender_from_model_file(model_npz=model_npz)
    body_model=body_models_dict[gender]
    body_params=human_mesh_utils.get_body_params(model_npz=model_npz,num_betas=num_betas,num_dmpls=num_dmpls,comp_device=comp_device)
    rendering_params_list=human_mesh_utils.get_full_rendering_params_list()
    rendering_params_list.remove('trans')
    #rendering_params_list.remove('root_orient') #TODO comment this line if we want to render the root_orient too.
    body_posed=human_mesh_utils.update_body_model_with_body_params(body_model=body_model,
            body_parms=body_params,rendering_params_list=rendering_params_list)
    body_mesh=human_mesh_utils.get_body_mesh_of_single_frame(body_model=body_posed,frameID=frameID)
    return body_mesh

def get_mesh_list_from_sampled_frames(dataset:Dataset,body_models_dict:dict,sampled_frames:list,comp_device:str)->list:
    mesh_list=[]
    #for frame_num in sampled_frames:
    for i in tqdm.trange(0,len(sampled_frames)):
        frame_num=sampled_frames[i]
        mesh_list.append(get_mesh_from_frame_num(dataset=dataset,body_models_dict=body_models_dict,frame_num=frame_num,comp_device=comp_device))
    return mesh_list

def get_mesh_list_from_N_bounding_frames(dataset:Dataset,body_models_dict:dict,actor_name:str,sampled_frames:list,N:int,first_frames:bool,obj_file_name:str,overwriteCache:bool,comp_device:str)->list:
    def get_mesh_list():
        if first_frames:
            n_bounding_samples_list=sampled_frames[:N]
        else:#last frames:
            n_bounding_samples_list=sampled_frames[-N:]
        return get_mesh_list_from_sampled_frames(dataset=dataset,body_models_dict=body_models_dict,sampled_frames=n_bounding_samples_list,comp_device=comp_device)
    """
    func=lambda: get_sample_actor_furtherst_method(dataset=dataset, actor_name=actor_name,num_of_frames_to_sample=num_of_frames_to_sample, farthest_sampling_method=farthest_sampling_method, theta_weights=theta_weights, only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by, iterations_per_sample=iterations_per_sample, comp_device=comp_device)
    """
    return get_obj(obj_file_name=obj_file_name,function_to_create_obj=get_mesh_list,overwriteCache=overwriteCache)

def get_furtherst_sampling_permutaion_N_bounding_frames_for_single_actor(dataset:Dataset,
        actor_name:str ,N:int,farthest_sampling_method:str,overwriteCache:str,comp_device:str,
        body_models_dict:dict,iterations_per_sample:int=1,theta_weights:torch.Tensor=None):
    print('get furtherst sampling permutaion {} bounding frames for actor {} in dataset {}'.format(N,actor_name,dataset.dataset_name))
    testing_values=get_testing_values()
    root_dir=testing_values['furtherst_sampling_permutation_root_dir']
    num_of_frames_for_actor=dataset.get_actor_obj_by_name(req_actor_name=actor_name).num_of_frames
    root_dir=os.path.join(root_dir,dataset.dataset_name,actor_name+"_total_n_frames_for_actor_"+str(num_of_frames_for_actor))
    if theta_weights==None:
        append_dir='without_theta_normalization'
    else:
        append_dir='with_theta_normalization'
    root_dir=os.path.join(root_dir,append_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    def create_result_sampling_from_N_bounding_frames(first_frames:bool):
        sampling_file=os.path.join(root_dir,'sampling.obj')
        sampled_frames=get_actor_furtherst_sampling_method_permutation(dataset=dataset,
                actor_name=actor_name, farthest_sampling_method=farthest_sampling_method,
                save_to_file_name=sampling_file,overwriteCache=overwriteCache, theta_weights=theta_weights,
                iterations_per_sample=iterations_per_sample, comp_device=comp_device)
        theta_norm_str='with'+('out' if (theta_weights==None) else '')+' theta weights'
        mesh_list_filename_base='mesh_list_first_{}'.format(N) if first_frames else 'mesh_list_last_{}'.format(N)
        mesh_list_filename_base+='_'+theta_norm_str.replace(' ','_')
        mesh_list_filename=mesh_list_filename_base+'_obj.pkl'
        mesh_file=os.path.join(root_dir,mesh_list_filename)
        sampled_mesh_list=get_mesh_list_from_N_bounding_frames(dataset=dataset,
                body_models_dict=body_models_dict,sampled_frames=sampled_frames,N=N,actor_name=actor_name,
                    first_frames=first_frames,obj_file_name=mesh_file,overwriteCache=overwriteCache,comp_device=comp_device)
        plot_list_filename=mesh_list_filename_base+'.jpg'
        plot_list_filename=os.path.join(root_dir,plot_list_filename)
        if not os.path.exists(plot_list_filename):
            if first_frames:
                n_samples_list=sampled_frames[:N]
            else:
                n_samples_list=sampled_frames[-N:]
            vb=[mesh.vertices for mesh in sampled_mesh_list]
            fb=[mesh.faces for mesh in sampled_mesh_list]
            labels=["s:{}.f:{}".format(i,v) for i,v in enumerate(n_samples_list)]
            strategy='mesh' #can be wireframe or spheres
            #create title
            first_or_last_str='first' if first_frames else 'last'
            #theta_norm_str='with'+('out' if (theta_weights==None) else '')+' theta weights'
            title='{} {} sampled frames for actor {}.\ntotal actor frames {}\n{}'.format(N,first_or_last_str,actor_name,num_of_frames_for_actor,theta_norm_str)
            plot_mesh_montage(vb=vb,fb=fb,labelb=labels,strategy=strategy,screenshot=plot_list_filename,off_screen=True,title=title)

    create_result_sampling_from_N_bounding_frames(first_frames=True)
    create_result_sampling_from_N_bounding_frames(first_frames=False)

def sample_furtherst_methods_permutation_check(dataset:Dataset):
    actor=dataset.actors[-4] #last_actor
    actor.save_cache_for_actor()
    #sampling_1=sampling_method_furthest.farthest_sampling_all_frames_for_actor(actor=actor,iterations_per_sample=1,farthest_sampling_method='vectors_loop',comp_device='cpu')
    init_random_seed()
    sampling_2=sampling_method_furthest.farthest_sampling_all_frames_for_actor(actor=actor,comp_device='cpu')
    print(sampling_2)
    #assert(sampling_1==sampling_2)
    print('a')

def test_sample_furtherst_methods_permutation_for_dataset(dataset:Dataset,N:int,
        max_actor_frames:int, body_models_dict:dict,
        comp_device:str='cpu',theta_weights:torch.Tensor=None,
        farthest_sampling_method:str='distance_matrix',overwriteCache:bool=False):
    print('a')
    key_func=lambda actor: actor.num_of_frames<=max_actor_frames
    actor_name_list=[actor.actor_name for actor in dataset.get_actors_by_key(key_func)]
    theta_weights_list=[None]
    if theta_weights!=None:
        theta_weights_list.append(theta_weights)
    for actor_name in actor_name_list:
        for cur_theta_weights in theta_weights_list:
            init_random_seed()
            get_furtherst_sampling_permutaion_N_bounding_frames_for_single_actor(dataset=dataset,
                    actor_name=actor_name ,N=N,farthest_sampling_method=farthest_sampling_method,overwriteCache=overwriteCache,
                    comp_device=comp_device,body_models_dict=body_models_dict, theta_weights=cur_theta_weights)

def test_sample_furtherst_methods_permutation_for_amass(N:int,
        max_actor_frames:int, comp_device:str,theta_weights:torch.Tensor=None,
        farthest_sampling_method:str='distance_matrix',overwriteCache:bool=False):
    body_models_dict:dict=human_mesh_utils.get_body_models_dict(comp_device=comp_device)
    amass_dir=sampling_main.get_defult_values()['cached_amass_object_dir']
    amass = mapping_utils.get_amass_obj(root_dataset_dir=amass_dir,overwriteCache=False)
    for amass_int_dataset in amass.dataset_list:
        dataset_name=amass_int_dataset.dataset_name
        if amass_int_dataset.num_of_frames['total']>200000:
            print('ommiting dataset {}'.format(dataset_name))
            continue
        init_testing_env(mini_dataset_name=dataset_name)
        dataset=get_dataset_obj(mini_dataset_name=dataset_name)
        test_sample_furtherst_methods_permutation_for_dataset(dataset=dataset,
                N=N, max_actor_frames=max_actor_frames,theta_weights=theta_weights,
                body_models_dict=body_models_dict,comp_device=comp_device,
                farthest_sampling_method=farthest_sampling_method,overwriteCache=overwriteCache)


"""
def test_2_sample_furtherst_methods_compare(dataset:Dataset):
    print('test_2_sample_furtherst_methods_compare')
    init_random_seed()

    # hardcoded for now change it if it causeing prolems cab be 'distance_matrix'or 'vectors_loop'
    #farthest_sampling_method='vectors_loop' 
    farthest_sampling_method='distance_matrix' # this is much faster on smaller datasets..
    only_allow_samples_that_diveded_by=1
    body_models_dict=human_mesh_ustils.get_body_models_dict()
    actor=dataset.actors[-1] # sample the actor with the least frames
    actor_name=actor.actor_name
    num_of_frames_to_sample=actor.num_of_frames//only_allow_samples_that_diveded_by
    sampled_frames=get_sample_actor_furtherst_method(dataset=dataset, actor_name=actor_name,
            farthest_sampling_method=farthest_sampling_method,
            num_of_frames_to_sample=num_of_frames_to_sample,
            only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
            theta_weights=None)

    print('plot the first frames:')

    n_first_samples=16
    n_first_samples_list=sampled_frames[:n_first_samples]
    sampled_mesh_list=get_mesh_list_from_sampled_frames(dataset=dataset,
            body_models_dict=body_models_dict,sampled_frames=n_first_samples_list)

    vb=[mesh.vertices for mesh in sampled_mesh_list]
    fb=[mesh.faces for mesh in sampled_mesh_list]
    labels=["s:{}.f:{}".format(i,v) for i,v in enumerate(n_first_samples_list)]
    strategy='mesh' #can be wireframe or spheres
    _data_mat(vb=vb,fb=fb,labelb=labels,strategy=strategy,screenshot='first16.jpg')

    print('plot the last frames:')

    n_last_samples_list=sampled_frames[-n_first_samples:] # last 16 frames
    sampled_mesh_list=get_mesh_list_from_sampled_frames(dataset=dataset,
            body_models_dict=body_models_dict,sampled_frames=n_last_samples_list,comp_device=comp_device)

    vb=[mesh.vertices for mesh in sampled_mesh_list]
    fb=[mesh.faces for mesh in sampled_mesh_list]
    labels=["s:{}.f:{}".format(i,v) for i,v in enumerate(n_first_samples_list)]
    strategy='mesh' #can be wireframe or spheres
    plot_mesh_montage(vb=vb,fb=fb,labelb=labels,strategy=strategy,screenshot='last16.jpg')

    print('a')
    print('a')
    print('a')
    print('a')
"""

def debug_matrix_sampling():
    dataset=get_dataset_obj(mini_dataset_name='ACCAD')
    #theta_weights are none for now
    actor=dataset.get_actor_obj_by_name('Male2Running_c3d')
    sampling_method_furthest.farthest_sampling_by_actor_name(dataset=dataset,num_of_frames_to_sample=actor.num_of_frames,
            actor_name=actor.actor_name,iterations_per_sample=1,only_allow_samples_that_diveded_by=1,
            farthest_sampling_method='distance_matrix',comp_device='cpu',theta_weights=None)

def main():
    print('init')
    testing_values=get_testing_values()
    mini_dataset_name=testing_values['mini_dataset_name']
    init_testing_env(mini_dataset_name=mini_dataset_name)
    print('sampling test begin')
    dataset=get_dataset_obj(mini_dataset_name=mini_dataset_name)
    #test_1_sample_furtherst_methods_compare(dataset=dataset)
    #test_2_sample_furtherst_methods_compare(dataset=dataset)
    #test_3_sample_furtherst_methods_debug(dataset=dataset)
    sample_furtherst_methods_permutation_check(dataset=dataset)

    """
    #debug_matrix_sampling()
    test_sample_furtherst_methods_permutation_for_amass(N=16,
            max_actor_frames=3000,theta_weights=get_theta_weights(),
            comp_device=get_testing_values()['comp_device'])
    """

if __name__=="__main__":
    main()
