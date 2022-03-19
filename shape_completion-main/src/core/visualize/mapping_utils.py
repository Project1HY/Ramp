from human_mesh_utils import is_valid_npz_file
#from save_load_obj import load_obj,save_obj
from save_load_obj import get_obj
from mapping_classes import Amass,Dataset,Actor,Animation,Frame
from mapping_classes import split_path_into_list, get_datset_name_from_model_npz, get_actor_name_from_model_npz, get_animation_name_from_model_npz, get_num_of_seconds_from_num_of_frames, get_num_of_minutes_from_num_of_frames #can be better
#from sampling import sample
from sampling_utils import create_model_npz_file_from_sampling
from sampling_utils import create_plot_from_sampling
import matplotlib.pyplot as plt
import mapping_graph_utils
import numpy as np
import torch
import tqdm
import os
import pickle
import matplotlib
import pickle

#TODO can be better - code duplication
def dataset_exists_in_list(dataset_name:str,dataset_list:list)->(bool,int):
    for index,cur_dataset in enumerate(dataset_list):
        if cur_dataset.dataset_name == dataset_name:
            return True,index
    return False,-1

#TODO can be better - code duplication
def actor_exists_in_dataset(actor_name:str,dataset:Dataset)->(bool,int):
    for index,cur_actor in enumerate(dataset.actors):
        if cur_actor.actor_name== actor_name:
            return True,index
    return False,-1

def getAmass(dir_name:str)->Amass:
    assert os.path.exists(dir_name)
    print('scanning dataset folder')
    dataset_list = list()
    valid_model_npz_files = list()
    for root, _, files in os.walk(dir_name, topdown=False):
       for f_name in files:
          if f_name.endswith(('.npz')):
              model_npz=os.path.join(root, f_name)
              if is_valid_npz_file(model_npz):
                  valid_model_npz_files.append(model_npz)
    for i in tqdm.trange(len(valid_model_npz_files)):
        model_npz=valid_model_npz_files[i]
        dataset_name = get_datset_name_from_model_npz(model_npz=model_npz,root_dataset_dir=dir_name)
        actor_name =  get_actor_name_from_model_npz(model_npz=model_npz)
        dataset_exists,dataset_num=dataset_exists_in_list(dataset_name=dataset_name,dataset_list=dataset_list)
        if not dataset_exists:
            cur_dataset_obj=Dataset(dataset_name=dataset_name)
            dataset_list.append(cur_dataset_obj)
            dataset_num=len(dataset_list)-1
        #get actor or create one if it not exists
        actor_exists,actor_num=actor_exists_in_dataset(actor_name=actor_name,dataset=dataset_list[dataset_num])
        if not actor_exists:
            cur_actor=Actor(model_npz=model_npz,taken_from_dataset=dataset_list[dataset_num].dataset_name)
            dataset_list[dataset_num].actors.append(cur_actor)
            actor_num=len(dataset_list[dataset_num].actors)-1
        dataset_list[dataset_num].actors[actor_num].add_animation_from_model_npz_file(model_npz=model_npz)
    #update all datasets
    for dataset in dataset_list:
        dataset._update_dataset_statistics()
    #sorting
    reverse_list = True
    for dataset in dataset_list:
        for actor in dataset.actors:
            #sort all animations for each actor
            actor.animations.sort(key=lambda animation: animation.num_of_frames,reverse=reverse_list)
        #sort all actors
        dataset.actors.sort(key=lambda actor: actor.num_of_frames,reverse=reverse_list)

    dataset_list.sort(key=lambda dataset: dataset.num_of_frames['total'],reverse=reverse_list)
    amass = Amass(dataset_list=dataset_list)
    return amass

def get_amass_obj(root_dataset_dir:str,root_cached_dict_dir:str='./cached_objects',overwriteCache=False):
    path_of_amass_obj=os.path.join(root_cached_dict_dir,'amass_obj.pkl')
    return get_obj(obj_file_name=path_of_amass_obj,function_to_create_obj=lambda:getAmass(root_dataset_dir),overwriteCache=overwriteCache)

def get_mini_amass_obj(root_dataset_dir:str,root_cached_dict_dir:str='./cached_objects',overwriteCache=False):
    path_of_mini_amass_obj=os.path.join(root_cached_dict_dir,'mini_amass_dataset_obj.pkl')
    amass = get_amass_obj(root_dataset_dir=root_dataset_dir,overwriteCache=overwriteCache)
    def get_mini_amass_dataset()->Dataset:
        mini_amass_dataset=get_mini_dataset_actor_list(amass=amass)
        mini_amass_dataset.create_cache_for_dataset() #save this mapping for futhre use
        return mini_amass_dataset
    funciton_to_create_mini_amass_obj=get_mini_amass_dataset
    return get_obj(obj_file_name=path_of_mini_amass_obj,function_to_create_obj=funciton_to_create_mini_amass_obj,overwriteCache=overwriteCache)

def get_color_list(n_colors:int)->list:
    color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    if n_colors>len(color_list):
        return []
    return color_list[:n_colors]

def plot_statistics(amass:Amass,root_plot_dir:str=os.path.join(os.curdir,'plots'))->None:
    cur_dir=os.path.join(root_plot_dir,'amass','info')
    plot_and_save_statistics_on_amass_level(amass=amass,root_dir_to_save_figures=cur_dir)
    base_cur_dir=os.path.join(root_plot_dir,'amass','datasets')
    for dataset in amass.dataset_list:
        cur_dir=os.path.join(base_cur_dir,dataset.dataset_name)
        plot_and_save_statistics_on_dataset_level(dataset=dataset,root_dir_to_save_figures=os.path.join(cur_dir,'info'))
        """
        for actor in dataset.actors:
            plot_and_save_statistics_on_actor_level(actor=actor,root_dir_to_save_figures=os.path.join(cur_dir,'actors',actor.actor_name))
        """

def get_actors_and_datasets_satisfying_conditions_from_amass(amass:Amass,valid_dataset_condition,valid_actor_condition)->(list,list):
    actor_list = []
    coresponding_dataset_list = []
    for dataset in amass.get_datasets_by_key(dataset_should_be_on_list_func=valid_dataset_condition):
        for actor in dataset.get_actors_by_key(actor_should_be_on_list_func=valid_actor_condition):
            actor_list.append(actor)
            coresponding_dataset_list.append(dataset.dataset_name)
    return actor_list,coresponding_dataset_list

def get_reduced_version_of_an_actor(actor:Actor,target_number_of_frames:int,taken_from_dataset:str)->Actor:
    #sort all animations for each actor
    if actor.num_of_frames < target_number_of_frames:
        return actor
    reverse=False
    actor.animations.sort(key=lambda animation: animation.num_of_frames,reverse=reverse)
    filtered_animation_list = []
    num_of_frames_for_actor=0
    for animation in actor.animations:
        if num_of_frames_for_actor >= target_number_of_frames:
            break
        #add animation
        if num_of_frames_for_actor+animation.num_of_frames >= target_number_of_frames:
            #cut the last animation if needed
            last_animation_frames_to_get=target_number_of_frames-num_of_frames_for_actor
            assert(last_animation_frames_to_get+num_of_frames_for_actor==target_number_of_frames)
            animation=Animation(file_path=animation.file_path,num_of_frames=last_animation_frames_to_get)
        filtered_animation_list.append(animation)
        num_of_frames_for_actor+=animation.num_of_frames
    #create new actoor
    assert filtered_animation_list != []
    new_reduced_actor = Actor(actor_name=actor.actor_name,gender=actor.gender,betas=actor.betas,num_betas=actor.num_betas,taken_from_dataset=taken_from_dataset)
    for animation in filtered_animation_list:
        new_reduced_actor._add_animation(animation=animation)
    return new_reduced_actor

def get_mini_dataset_actor_list(amass:Amass)->Dataset:
    relevant_subset_of_the_amass_datasets=['KIT','BMLrub','CMUa','EyesJapanDataset','BMLmovi']
    minimum_frames_per_actor=10000 #hardcoded for now
    #minimum_frames_per_actor=4000 #hardcoded for now
    #minimum_frames_per_actor=200 #hardcoded for now #TODO just to check on my linux computer
    dataset_condition=lambda dataset:dataset.dataset_name in relevant_subset_of_the_amass_datasets
    actor_condition=lambda actor:actor.num_of_frames>=minimum_frames_per_actor
    actor_list,coresponding_dataset_list=get_actors_and_datasets_satisfying_conditions_from_amass(amass=amass,valid_dataset_condition=dataset_condition,valid_actor_condition=actor_condition)
    reduced_actor_list=[]
    for i,(actor,dataset_name) in enumerate(zip(actor_list,coresponding_dataset_list)):
            reduced_actor=get_reduced_version_of_an_actor(actor=actor,target_number_of_frames=minimum_frames_per_actor,taken_from_dataset=dataset_name)
            reduced_actor_list.append(reduced_actor)
            #break #TODO remove this line.for debug only!
    reverse_list = True
    mini_amass_dataset = Dataset(dataset_name="mini_amass",actors=reduced_actor_list)
    mini_amass_dataset.actors.sort(key=lambda actor: actor.num_of_frames,reverse=reverse_list) #sort by length of each one
    return mini_amass_dataset

def print_statistics(root_dataset_dir:str,root_plot_dir:str=os.path.join(os.curdir,'plots'),overwriteCache=False)->None:
    amass = get_amass_obj(root_dataset_dir=root_dataset_dir,overwriteCache=overwriteCache)
    #plot_statistics(amass=amass,root_plot_dir=root_plot_dir)
    mini_amass_dataset=get_mini_dataset_actor_list(amass=amass) # 3.3 million frames

#use this for actual requested sampling
def sample_mini_amass(root_dataset_dir:str,method:str,num_of_frames_to_sample:int=None,overwriteCache=False)->list:
    mini_amass_dataset = get_mini_amass_obj(root_dataset_dir=root_dataset_dir,overwriteCache=overwriteCache)
    sampling=sample(dataset=mini_amass_dataset,method=method,num_of_frames_to_sample=num_of_frames_to_sample,seed=1)
    file_path_root='./sampling'
    full_file_path=os.path.join(file_path_root,'dataset_{}_sample_method_{}_num_of_samples_{}.npz'.format('mini_amass',method,num_of_frames_to_sample))
    create_model_npz_file_from_sampling(dataset=mini_amass_dataset,sampling=sampling,file_path=full_file_path)
    create_plot_from_sampling(dataset=mini_amass_dataset,sampling=sampling,file_path=full_file_path)
    return sampling

def plot_and_save_statistics_on_amass_level(amass:Amass,root_dir_to_save_figures:str):
    if not os.path.exists(root_dir_to_save_figures):
        os.makedirs(root_dir_to_save_figures)
    def grouped_gender_bar(fig_file_name:str,fig_folder:str,get_dataset_dict,y_label:str,title:str):
        mapping_graph_utils.get_grouped_bar_charts_with_x_labels(
                x_labels=[dataset.dataset_name for dataset in amass.dataset_list],
                data_label_bar_1='male',
                data_label_bar_2='female',
                data_bar_1=[get_dataset_dict(dataset)['male'] for dataset in amass.dataset_list],
                data_bar_2=[get_dataset_dict(dataset)['female'] for dataset in amass.dataset_list],
                y_label=y_label,
                title=title,
                path_to_save_fig=os.path.join(fig_folder,fig_file_name),
                show=False
        )

    def grouped_non_gender_bar(fig_file_name:str,fig_folder:str,get_dataset_dict,y_label:str,title:str):
        mapping_graph_utils.get_bar_chart_with_x_labels(
                x_labels=[dataset.dataset_name for dataset in amass.dataset_list],
                data_bar=[get_dataset_dict(dataset)['total'] for dataset in amass.dataset_list],
                y_label=y_label,
                title=title,
                path_to_save_fig=os.path.join(fig_folder,fig_file_name),
                show=False
        )
    def make_2_bar_plots(get_dataset_dict,parameter_name:str):
          base_title='number of {} by dataset'.format(parameter_name)
          grouped_gender_title=base_title+' and gender'
          base_file_name=base_title.replace(' ','_')+'.png'
          base_grouped_file_name=grouped_gender_title.replace(' ','_')+'.png'
          grouped_non_gender_bar(fig_file_name=base_file_name,fig_folder=root_dir_to_save_figures,get_dataset_dict=get_dataset_dict,y_label='number of {}'.format(parameter_name),title=base_title)
          grouped_gender_bar(fig_file_name=base_grouped_file_name,fig_folder=root_dir_to_save_figures,get_dataset_dict=get_dataset_dict,y_label='number of {}'.format(parameter_name),title=grouped_gender_title)
    make_2_bar_plots(get_dataset_dict=lambda dataset:dataset.num_of_actors,parameter_name="actors")
    make_2_bar_plots(get_dataset_dict=lambda dataset:dataset.num_of_frames,parameter_name="frames")
    make_2_bar_plots(get_dataset_dict=lambda dataset:dataset.num_of_animations,parameter_name="animations")
    make_2_bar_plots(get_dataset_dict=lambda dataset:dataset.num_of_minutes_in_120_fps_video,parameter_name="minutes in video (fps:120)")

def plot_and_save_statistics_on_dataset_level(dataset:Dataset,root_dir_to_save_figures:str):
    if not os.path.exists(root_dir_to_save_figures):
        os.makedirs(root_dir_to_save_figures)
    def grouped_bar(fig_file_name:str,fig_folder:str,get_data,y_label:str,title:str,actor_list:list):
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        mapping_graph_utils.get_bar_chart_with_x_labels(
                x_labels=[actor.actor_name for actor in actor_list],
                data_bar=[get_data(actor) for actor in actor_list],
                y_label=y_label,
                title=title,
                path_to_save_fig=os.path.join(fig_folder,fig_file_name),
                x_labels_font_size=5,
                show=False
        )
    def grouped_bar_warp(get_data,parameter_name:str,gender:str):
        gender_str = '' if gender == 'total' else gender+' '
        actor_list = dataset.actors if gender == 'total' else dataset.get_only_spesific_actor_list_by_gender(gender)
        title = 'num of {} for {}actors in dataset {}'.format(parameter_name,gender_str,dataset.dataset_name)
        y_label = 'number of {}'.format(parameter_name)
        fig_file_name = title.replace(' ','_')
        grouped_bar(fig_file_name=fig_file_name,fig_folder=root_dir_to_save_figures,get_data=get_data,y_label=y_label,title=title,actor_list=actor_list)
    for gender in ['total','male','female']:
        grouped_bar_warp(get_data=lambda actor:actor.num_of_animations,parameter_name='animations',gender=gender)
        grouped_bar_warp(get_data=lambda actor:actor.num_of_frames,parameter_name='frames',gender=gender)
        grouped_bar_warp(get_data=lambda actor:actor.num_of_minutes_in_120_fps_video,parameter_name="minutes in video (fps:120)",gender=gender)

def plot_and_save_statistics_on_actor_level(actor:Actor,root_dir_to_save_figures:str):
    if not os.path.exists(root_dir_to_save_figures):
        os.makedirs(root_dir_to_save_figures)
    #some code duplication
    def grouped_bar(fig_file_name:str,fig_folder:str,get_data,y_label:str,title:str):
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        mapping_graph_utils.get_bar_chart_with_x_labels(
                x_labels=[animation.animation_name for animation in actor.animations],
                data_bar=[get_data(animation) for animation in actor.animations],
                y_label=y_label,
                title=title,
                path_to_save_fig=os.path.join(fig_folder,fig_file_name),
                x_labels_font_size=1,
                show=False
        )
    def grouped_bar_warp(get_data,parameter_name:str):
        title = 'num of {} for actor {}'.format(parameter_name,actor.actor_name)
        y_label = 'number of {}'.format(parameter_name)
        fig_file_name = title.replace(' ','_')
        grouped_bar(fig_file_name=fig_file_name,fig_folder=root_dir_to_save_figures,get_data=get_data,y_label=y_label,title=title)
    grouped_bar_warp(get_data=lambda animation:animation.num_of_frames,parameter_name='frames')
    grouped_bar_warp(get_data=lambda animation:animation.num_of_seconds_in_120_fps_video,parameter_name='seconds in video (fps:120)')

def example_plot_from_amss(amass:Amass):
    mapping_graph_utils.get_grouped_bar_charts_with_x_labels(
            x_labels=[dataset.dataset_name for dataset in amass.dataset_list],
            data_label_bar_1='male',
            data_label_bar_2='female',
            data_bar_1=[dataset.num_of_frames['male'] for dataset in amass.dataset_list],
            data_bar_2=[dataset.num_of_frames['female'] for dataset in amass.dataset_list],
            y_label='number of frames',
            title='number of frames by dataset and gender',
            path_to_save_fig=os.path.join(os.curdir,'my_fig.png')
            )

def example_plot_2_from_amss(amass:Amass):
    mapping_graph_utils.get_bar_chart_with_x_labels(
            x_labels=[dataset.dataset_name for dataset in amass.dataset_list],
            data_bar=[dataset.num_of_frames['total'] for dataset in amass.dataset_list],
            y_label='number of frames',
            title='number of frames by group',
            )
