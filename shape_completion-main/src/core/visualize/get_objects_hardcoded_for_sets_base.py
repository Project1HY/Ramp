import os
import numpy as np
# almost all the functions here don't get parameters and only returns the reveant objects
from mapping_classes import Amass,Dataset
from human_mesh_utils import get_defult_dir_values
from human_mesh_utils import get_body_mesh_of_single_frame_from_npz_file
from human_mesh_utils import chdir_to_visuallize_folder
from mapping_utils import get_amass_obj
from save_load_obj import get_obj,load_obj
from sampling import save_mini_amass_obj
from create_hit_for_amass import return_hit_for_dataset
from smpl_segmentations import SegmentationManger
import sys
sys.path.insert(0, '..')
from data.index import HierarchicalIndexTree
from data.base import CompletionDataset
from data.sets import DatasetMenu

# with caching

def get_defult_values()->dict:
    return get_defult_dir_values()

def get_amass_object_f()->Amass:
    defult_values=get_defult_values()
    return get_amass_obj(root_dataset_dir=defult_values['amass_dir'],
            root_cached_dict_dir=['cached_amass_object_dir'],
            overwriteCache=False)

def get_mini_amass_dataset_object_f()->Dataset:
    defult_values=get_defult_values()
    output_mini_amass_full_filepath=os.path.join(defult_values['cached_dataset_objects_dir'],'mini_amass_dataset_obj.pkl')
    input_amass_obj_full_filepath=os.path.join(defult_values['cached_amass_object_dir'],'amass_obj.pkl')
    #currnetly add_fps_full_sampling is hardcoded to false on purpuse
    save_mini_amass_obj(input_amass_obj_full_filepath=input_amass_obj_full_filepath,
            output_mini_amass_full_filepath=output_mini_amass_full_filepath,
            overwriteCache=False,add_fps_full_sampling=False)
    dataset=load_obj(output_mini_amass_full_filepath)
    dataset.fix_root_dir_for_local_machine_if_needed(new_root=defult_values['amass_dir'])
    return dataset

def get_hit_object_f(dataset:Dataset,num_projections:int,mode:str,
        take_only_first_n_actors:int=None,
        num_of_frames_from_each_actor:int=None)->HierarchicalIndexTree:
    #valid_modes 'full','reduced_random','reduced_fps'
    if mode=='full':
        assert(num_of_frames_from_each_actor==None)
        assert(take_only_first_n_actors==None)
    defult_values=get_defult_values()
    output_mini_amass_hit_full_filepath=os.path.join(
            defult_values['cached_dataset_objects_dir'],
            f'hit_mini_amass_dataset_mode_{mode}_n_frames_per_actor_{num_of_frames_from_each_actor}_obj.pkl')
    def create_hit_func():
       return return_hit_for_dataset(dataset=dataset,num_projections=num_projections,mode=mode,
               num_of_frames_from_each_actor=num_of_frames_from_each_actor,
               take_only_first_n_actors=take_only_first_n_actors)
    return get_obj(obj_file_name=output_mini_amass_hit_full_filepath,
            function_to_create_obj=create_hit_func,overwriteCache=False)

def get_valid_amass_ds_modes()->list:
    return ['full', 'reduced_random', 'reduced_fps', 'reduced_fps_debug', 'reduced_random_debug']

def get_valid_methods()->list:
    return [ 'fp2Np_same_pose','fp2Np_other_pose','fp2Np_multiple_poses' ]

def get_valid_prior_types()->list:
    return ['pppc','single_full_shape']

def get_completion_dataset_f(mode:str)->CompletionDataset:
    assert(mode in get_valid_amass_ds_modes())
    #old_dir=os.getcwd()
    #chdir_to_visuallize_folder()
    #valid_modes 'full','reduced_random','reduced_fps'
    defult_values=get_defult_values()
    output_path=os.path.join(defult_values['cached_dataset_objects_dir'],f'completion_dataset_mode_{mode}_obj.pkl')
    dataset_str=None#'MiniAmassFull' if mode=='full' elif mode==
    if mode=='full':
        dataset_str='MiniAmassFull'
    elif mode=='reduced_random':
        dataset_str='MiniAmassReducedRandom'
    elif mode=='reduced_fps':
        dataset_str='MiniAmassReducedFPS'
    elif mode=='reduced_fps_debug':
        dataset_str='MiniAmassReducedFPSDebug'
    elif mode=='reduced_random_debug':
        dataset_str='MiniAmassReducedRandomDebug'
    else:
        assert(False)
    def get_completion_dataset_f():
       #return 1
       ds: CompletionDataset = DatasetMenu.order(dataset_str)
       if mode.startswith('reduced_'):
           ds.init_data_cache()
       return ds
    """
    obj=get_obj(obj_file_name=output_path,
            function_to_create_obj=get_completion_dataset_f,overwriteCache=False)
    os.chdir(old_dir)
    return obj
    """
    return get_obj(obj_file_name=output_path,
            function_to_create_obj=get_completion_dataset_f,overwriteCache=False)



def get_segmentation_manger_f()->SegmentationManger:
    return SegmentationManger(n_joints=6,
            include_full_segmentation=True,
            segmentation_dict_filepath=get_defult_values()['smpl_segmentation_file'])

def get_completion_dataset(mode:str)->CompletionDataset:
    return visualize_chdir_warp(lambda:get_completion_dataset_f(mode=mode))

def get_hit_object(dataset:Dataset,num_projections:int,mode:str,
        take_only_first_n_actors:int=None,
        num_of_frames_from_each_actor:int=None)->HierarchicalIndexTree:
    return visualize_chdir_warp(lambda:get_hit_object_f(dataset=dataset,
        num_projections=num_projections,mode=mode,
        take_only_first_n_actors=take_only_first_n_actors,
        num_of_frames_from_each_actor=num_of_frames_from_each_actor))

def get_mini_amass_dataset_object()->CompletionDataset:
    return visualize_chdir_warp(lambda:get_mini_amass_dataset_object_f())

def get_amass_object()->CompletionDataset:
    return visualize_chdir_warp(lambda:get_amass_object_f())


def get_segmentation_manger()->SegmentationManger:
    return visualize_chdir_warp(lambda:get_segmentation_manger_f())

#TODO add get reduced random hit
def get_faces_templete_from_dataset(dataset:Dataset)->np.array:
    model_npz_example=dataset.actors[0].animations[0].file_path
    mesh=get_body_mesh_of_single_frame_from_npz_file(model_npz=model_npz_example,
            frameID=0,comp_device='cpu',body_models_dict=None)
    #FIXME make sure that mesh  faces is from np.array type
    return mesh.faces

def visualize_chdir_warp(get_obj_func):
    #return the requested obj
    old_dir=os.getcwd()
    chdir_to_visuallize_folder()
    obj=get_obj_func()
    os.chdir(old_dir)
    return obj

