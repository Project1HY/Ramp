import os
import torch
import random
import datetime
from save_load_obj import get_obj,load_obj,save_obj
from mapping_classes import Dataset
from sampling_method_random import sample_random
from sampling_method_furthest import farthest_sampling_per_actor
from sampling_method_furthest import farthest_sampling_all_frames_for_actor
import mapping_utils
import sampling_utils
import tqdm

# creating objects

def save_amass_object(input_amass_dir:str,output_amass_obj_full_filepath:str,overwriteCache:bool)->None:
    get_obj(obj_file_name=output_amass_obj_full_filepath,function_to_create_obj=lambda:mapping_utils.getAmass(input_amass_dir),overwriteCache=overwriteCache)

def save_mini_amass_obj(input_amass_obj_full_filepath:str,output_mini_amass_full_filepath:str,overwriteCache:bool,add_fps_full_sampling:bool)->None:
    def get_mini_amass_obj():
        amass = load_obj(f_name=input_amass_obj_full_filepath)
        mini_amass_dataset=mapping_utils.get_mini_dataset_actor_list(amass=amass)
        #mini_amass_dataset.create_cache_for_dataset() #save this mapping for futhre use #TODO uncomment this!
        if add_fps_full_sampling:
            sample_full_dataset(dataset=mini_amass_dataset)
        return mini_amass_dataset
    get_obj(obj_file_name=output_mini_amass_full_filepath,function_to_create_obj=get_mini_amass_obj,overwriteCache=overwriteCache)

def save_internal_dataset_in_amass_obj(input_amass_obj_full_filepath:str,
        req_dataset_name:str,output_dataset_full_filepath:str,overwriteCache:bool,
        add_fps_full_sampling:bool)->None:
    def get_internal_dataset_obj()->Dataset:
        amass = load_obj(f_name=input_amass_obj_full_filepath)
        relevant_dataset=None
        for dataset in amass.dataset_list:
            if dataset.dataset_name==req_dataset_name:
                relevant_dataset=dataset
        if relevant_dataset==None:
            dataset_list_names=[dataset.dataset_name for dataset in amass.dataset_list]
            raise Exception('{} is not a valid amass dataset.\n please try one of the folowing.{}'.format(req_dataset_name,dataset_list_names))
        #relevant_dataset.create_cache_for_dataset()
        if add_fps_full_sampling:
            sample_full_dataset(dataset=relevant_dataset)
        return relevant_dataset
    get_obj(obj_file_name=output_dataset_full_filepath,function_to_create_obj=get_internal_dataset_obj,overwriteCache=overwriteCache)

# sampling dataset objects

def basic_sampling_assert_valid_input(dataset:Dataset,method:str,num_of_frames_to_sample:int,output_sample_dir:str)->None:
    #TODO (optional:add more valid input checks,currently this function is very naive)
    errStr = ''
    valid_sampling_method=['sample_furtherst','sample_random']
    if num_of_frames_to_sample > dataset.num_of_frames['total']:
       errStr+="worng usage of sample,num_of_frames_to_sample shold be smaller or equal to num of frames is the dataset"
    if method not in valid_sampling_method:
        errStr+="worng sampling method.please coose sampling method: {}".format(valid_sampling_method)
    if not os.path.exists(output_sample_dir):
        errStr+="worng output_sample_dir {} not exists".format(output_sample_dir)
    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)

def assert_sampling_furthest_valid_input(dataset:Dataset,num_of_frames_to_sample:str,comp_device:str,
        num_of_sampling_iterations:int,num_of_adject_frames_removal_for_each_sample:int,
        only_allow_samples_that_diveded_by:str,path_to_theta_vector:str)->None:
    errStr = ''

    actor_with_least_frames,actor_with_max_frames=dataset.get_actor_with_least_and_max_frame()

    #min check

    first_idx,last_idx=dataset.get_first_inclusive_and_last_exlusize_number_by_actor_name(actor_name=actor_with_least_frames.actor_name)
    minimum_frames_list_len=last_idx-first_idx
    actual_minimum_frame_list_we_sample_from=minimum_frames_list_len//only_allow_samples_that_diveded_by
    num_of_frames_we_remove_per_each_sample=num_of_adject_frames_removal_for_each_sample*2+1
    total_frames_we_remove_from_unsapled_list=num_of_frames_we_remove_per_each_sample*num_of_frames_to_sample
    if actual_minimum_frame_list_we_sample_from<total_frames_we_remove_from_unsapled_list:
        errStr = '''worng sample arguments:\n
        actual_minimum_frame_list_we_sample_from:{},\n
        total_frames_we_remove_from_unsapled_list:{}'''.format(\
                actual_minimum_frame_list_we_sample_from,\
                total_frames_we_remove_from_unsapled_list)

    #max check

    #TODO make max memory sampling check

    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)

def sample(method:str,num_of_frames_to_sample:str,dataset_object_full_filepath:str,output_sample_dir:str,
        save_npz_visualization_file:bool,save_sampling_histogram_figure:bool,hist_n_bins:int,
        show_fig:bool,seed:int=None, **kwargs)->None:

    """
    kwargs should be:
    - comp_device
    - num_of_sampling_iterations
    - padding_frames_removal
    - only_allow_samples_that_diveded_by
    """
    dataset=load_obj(f_name=dataset_object_full_filepath)
    if num_of_frames_to_sample==-1:
        num_of_frames_to_sample=dataset.num_of_frames['total']
        if method=='sample_furtherst':
            num_of_frames_to_sample=num_of_frames_to_sample//kwargs["only_allow_samples_that_diveded_by"]
    basic_sampling_assert_valid_input(dataset=dataset,method=method,\
            num_of_frames_to_sample=num_of_frames_to_sample,output_sample_dir=output_sample_dir)


    if seed==None:
        print("Warning: using random seed for sampling")
        seed=datetime.now()
    random.seed(seed)

    sampling=[]

    if method=='sample_random':
        print('sample_random')
        sampling=sample_random(dataset=dataset,num_of_frames_to_sample=num_of_frames_to_sample)
    else:
        #on this case kwargs should contain:
        #comp_device
        #num_of_sampling_iterations
        #
        #only_allow_samples_that_diveded_by
        #path_to_theta_vector
        assert([k for k,_ in kwargs.items()]==['comp_device','iterations_per_sample',
            'only_allow_samples_that_diveded_by','path_to_theta_vector','farthest_sampling_method'])
        comp_device=kwargs["comp_device"]
        iterations_per_sample=kwargs["iterations_per_sample"]
        only_allow_samples_that_diveded_by=kwargs["only_allow_samples_that_diveded_by"]
        path_to_theta_vector=kwargs["path_to_theta_vector"]
        farthest_sampling_method=kwargs['farthest_sampling_method']

        theta_weights=load_obj(path_to_theta_vector) if path_to_theta_vector!='' else None
        sampling=farthest_sampling_per_actor(dataset=dataset,num_of_frames_to_sample=num_of_frames_to_sample,\
                comp_device=comp_device,iterations_per_sample=iterations_per_sample,
                only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
                theta_weights=theta_weights,farthest_sampling_method=farthest_sampling_method)
        #method is 'sample_furtherst'
        print('sample_furtherst')

    assert(sampling!=[])

    results_dir=os.path.join(output_sample_dir,'{}_results_for_dataset_{}_num_of_samples_{}'
            .format(method,dataset.dataset_name,num_of_frames_to_sample))
    if method=='sample_furtherst':
        results_dir+="_method_"+kwargs["farthest_sampling_method"]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    sampling_result_obj_path=os.path.join(results_dir,'sampling.pkl')
    sampling_histogram_obj_path=os.path.join(results_dir,'sampling_histogram.jpg')
    sampling_npz_viz_file_path=os.path.join(results_dir,'sampling_visualization_file.npz')

    if save_npz_visualization_file:
        sampling_utils.create_model_npz_file_from_sampling(dataset=dataset,sampling=sampling,file_path=sampling_npz_viz_file_path)
    if save_sampling_histogram_figure:
        sampling_utils.create_plot_from_sampling(dataset_name=dataset.dataset_name,
                sampling=sampling,file_path=sampling_histogram_obj_path,
                sampling_method_name=method,n_bins=hist_n_bins,show_fig=show_fig)
    #save the sampling
    save_obj(obj_to_save=sampling,f_name=sampling_result_obj_path)

    return

def sample_full_dataset(dataset:Dataset,comp_device:str=None)->None:
    print('sample dataset')
    #for actor in dataset.actors:
    if comp_device==None:
        comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using comp device:{}'.format(comp_device))
    for i in tqdm.trange(len(dataset.actors), position=0, desc='Actor'):
        actor=dataset.actors[i]
        sampling=farthest_sampling_all_frames_for_actor(actor,comp_device=comp_device)
        actor.fps_full_sampling=sampling
    return
