import os
import torch
import re
import numpy as np
from human_mesh_utils import get_gender_list
from human_mesh_utils import get_body_params
from human_mesh_utils import compute_distanse_matrix_from_data_matrix
from human_mesh_utils import covert_path_to_current_os
import tqdm


#hardcoded for now:
num_betas =16
num_dmpls =8

class Frame:
    def __init__(self,frameID:int, file_path:str):
        self._frameID:int=frameID
        self._file_path:str=file_path
    def get_frame(self)->int:
        return self._frameID
    def get_file_path(self)->int:
        return self._file_path
    def __eq__(self,other):
        return self.get_frame()==other.get_frame() and self.get_file_path()==other.get_file_path()
    def __hash__(self):
        return hash((self._frameID,self._file_path))
    def get_concat_pose_vector(self)->torch.Tensor:
        bdata = np.load(self._file_path)
        pose=torch.Tensor(bdata['poses'][self._frameID, 3:66])
        return pose

    def get_full_model_npz_vector_in_numpy_for_frame(self)->dict:
        #cpu hardcoded for now
        all_body_params_for_model_npz_file=get_body_params(model_npz=self._file_path,num_betas=num_betas,num_dmpls=num_dmpls,comp_device='cpu')
        res=dict([(k,v[self._frameID,:].cpu().detach().numpy().reshape(1,-1)) for k,v in all_body_params_for_model_npz_file.items()])
        return res

class Animation:
    def __init__(self, file_path:str, num_of_frames:int):
        self.num_of_frames:int=num_of_frames
        self.num_of_seconds_in_120_fps_video:int=get_num_of_seconds_from_num_of_frames(num_of_frames=num_of_frames,frame_per_second=120)
        self.file_path:str=file_path
        self.animation_name:str=get_animation_name_from_model_npz(model_npz=self.file_path)

    def get_concat_pose_vector(self)->torch.Tensor:
        bdata = np.load(self.file_path)
        pose=torch.Tensor(bdata['poses'][:self.num_of_frames, 3:66])
        return pose

    def change_root_dir(self,new_root:str,old_root:str):
        #assume valid input!
        #need to be os agnustic
        relpath=self.file_path[len(old_root)+1:]
        relpath=covert_path_to_current_os(relpath) #TODO add change unix to windows vice verse ADD !!!!
        self.file_path=os.path.join(new_root,relpath)

class Actor:
    def __init__(self,model_npz:str=None, actor_name:str=None, gender:bool=None, betas:str=None, num_betas:int=16,taken_from_dataset:str=None):
        if (actor_name==None  or gender==None  or betas ==None)  and model_npz == None:
            raise RuntimeError('worng call to actor costructur')
        if model_npz != None:
            bdata=np.load(model_npz)
            self.actor_name:str=get_actor_name_from_model_npz(model_npz=model_npz)
            self.gender:str=bdata['gender'].item()
            self.betas:str=str(torch.Tensor(bdata['betas'][:num_betas][np.newaxis]))
        else:
            self.actor_name:str=actor_name
            self.gender:str=gender
            self.betas:str=betas
        self.animations:list=[] #animations
        self.num_of_frames:int=0
        self.num_of_minutes_in_120_fps_video:int=0
        self.num_of_animations:int=0
        self.num_betas:int=num_betas
        self._data_mat=None
        self._dist_mat=None
        self.fps_full_sampling=None
        self.taken_from_dataset=taken_from_dataset

    def _add_animation(self,animation:Animation):
        self.animations.append(animation)
        self.num_of_animations=len(self.animations)
        self.num_of_frames+=animation.num_of_frames
        self.num_of_minutes_in_120_fps_video+=get_num_of_seconds_from_num_of_frames(num_of_frames=animation.num_of_frames,frame_per_second=120)

    def add_animation_from_model_npz_file(self,model_npz:str)->None:
        #NOTE assuming the animation is not already exists
        bdata=np.load(model_npz)
        num_of_frames=len(bdata['trans'])

        gender= bdata['gender'].item()
        assert gender == self.gender

        betas=str(torch.Tensor(bdata['betas'][:self.num_betas][np.newaxis]))
        assert betas == self.betas

        animation = Animation(file_path=model_npz,num_of_frames=num_of_frames)
        self._add_animation(animation=animation)


    def get_data_mat(self)->torch.Tensor:
        if self._data_mat!=None:
            return self._data_mat
        #print("create_data_mat for actor {}".format(self.actor_name))
        num_of_frames=self.num_of_frames
        #comp_device='cpu'
        res=self.animations[0].get_concat_pose_vector()
        #for cur_animation_num in tqdm.trange(1,self.num_of_animations):
            #animation=self.animations[cur_animation_num]
        for animation in self.animations[1:]:
            cur_vec=animation.get_concat_pose_vector()
            res=torch.cat((res,cur_vec),dim=0)
        #self._data_mat=res
        return res

    #TODO maybe add theta_weights support if needed.now it's feels pretty redundant

    def save_dist_mat(self)->None:
        assert(self._data_mat!=None)
        if self._dist_mat!=None:
            return
        self._dist_mat=compute_distanse_matrix_from_data_matrix(data_matrix=self._data_mat)

    def save_cache_for_actor(self)->None:
        self.save_data_mat()
        #self.save_dist_mat()

    def save_data_mat(self):
        if self._data_mat!=None:
            return
        self._data_mat=self.get_data_mat()
        return

    def get_dist_mat(self)->torch.Tensor:
        if self._dist_mat!=None:
            #self.save_dist_mat()
            return self._dist_mat
        else:
            _dist_mat=compute_distanse_matrix_from_data_matrix(data_matrix=self.get_data_mat())
            return _dist_mat

    def get_animations_by_key(self,animation_should_be_on_list_func)->list:
        res = []
        for animation in self.animations:
            if animation_should_be_on_list_func(animation):
                res.append(animation)
        return res

    def get_animation_name_and_frameID_from_index(self,index)->(str,int):
        #index can be fps_index_for example
        assert(self.fps_full_sampling!=None)
        assert(index in self.fps_full_sampling) #assuming fps full sampling contain all the frames on the actor
        for animation in self.animations:
            if index>animation.num_of_frames:
                index-=animation.num_of_frames
                continue
            else:
                index-=1
                return(animation.animation_name,index)

class Dataset:
    def __init__(self,dataset_name:str,actors:list=None):

        self._num_to_frame:dict=dict()
        self._frame_to_num:dict=dict()
        self._data_mat=None

        self.dataset_name:str=dataset_name
        if actors is None:
            self.actors:list=[]
        else:
        #if self.actors is not None:
            self.actors:list=actors
            self._update_dataset_statistics()
    def _update_dataset_statistics(self):
        gender_list = get_gender_list()
        self.num_of_actors = {}
        self.num_of_animations = {}
        self.num_of_frames = {}
        self.num_of_minutes_in_120_fps_video = {}
        self._data_mat=None
        for gender in gender_list + ['total']:
            self.num_of_actors[gender]=0
            self.num_of_animations[gender]=0
            self.num_of_frames[gender]=0
            self.num_of_minutes_in_120_fps_video[gender]=0
        for actor in self.actors:
            self.num_of_actors[actor.gender]+=1
            self.num_of_animations[actor.gender]+=actor.num_of_animations
            self.num_of_frames[actor.gender]+=actor.num_of_frames
            self.num_of_minutes_in_120_fps_video[actor.gender]+=get_num_of_minutes_from_num_of_frames(num_of_frames=actor.num_of_frames,frame_per_second=120)
        for gender in gender_list:
            self.num_of_actors['total']+=self.num_of_actors[gender]
            self.num_of_animations['total']+=self.num_of_animations[gender]
            self.num_of_frames['total']+=self.num_of_frames[gender]
            self.num_of_minutes_in_120_fps_video['total']+=self.num_of_minutes_in_120_fps_video[gender]
    def get_actors_by_key(self,actor_should_be_on_list_func)->list:
        res = []
        for actor in self.actors:
            if actor_should_be_on_list_func(actor):
                res.append(actor)
        return res

    def get_male_actors_list(self)->list:
        return self.get_actors_by_key(actor_should_be_on_list_func=lambda actor:actor.gender=='male')
    def get_female_actors_list(self)->list:
        return self.get_actors_by_key(actor_should_be_on_list_func=lambda actor:actor.gender=='female')
    def get_only_spesific_actor_list_by_gender(self,gender:str)->list:
        assert gender in ['male','female']
        if gender=='male':
            return self.get_male_actors_list()
        else:
            return self.get_female_actors_list()

    def get_actor_obj_by_name(self,req_actor_name:str)->Actor:
        res=self.get_actors_by_key(actor_should_be_on_list_func=lambda actor:actor.actor_name==req_actor_name)
        assert(len(res)==1)
        return res[0]

    def get_animation_path_by_dataset_actor_and_animation_names(self,taken_from_dataset_name:str,actor_name:str,animation_name:str)->str:
        #assume valid input
        actor_res=self.get_actors_by_key(actor_should_be_on_list_func=lambda actor:actor.actor_name==actor_name and actor.taken_from_dataset == taken_from_dataset_name)
        assert(len(actor_res)==1)
        actor_res=actor_res[0]
        animation_res=actor_res.get_animations_by_key(animation_should_be_on_list_func=lambda animation:animation.animation_name==animation_name)
        assert(len(animation_res)==1)
        animation_res=animation_res[0]
        return animation_res.file_path



    def __get_frame_by_number(self,num:int)->Frame:
        #mapping numbers from '0' up to self.num_of_frames['total']-1 into frames (filepath,frameID)
        assert num<self.num_of_frames['total'] and num >= 0
        for actor in self.actors:
            if num-actor.num_of_frames<0:
                for animation in actor.animations:
                    if num-animation.num_of_frames<0:
                        frameID = num
                        return Frame(file_path=animation.file_path,frameID=frameID)
                    num-=animation.num_of_frames
            num-=actor.num_of_frames

    def __get_number_by_frame(self,frame:Frame)->int:
        #self.num_to_frame:dict=dict()
        #self.frame_to_num:dict=dict()
        frameID=frame.get_frame()
        file_path=frame.get_file_path()
        actor_name=get_actor_name_from_model_npz(model_npz=file_path)
        animation_name=get_animation_name_from_model_npz(model_npz=file_path)
        res=0
        for actor in self.actors:
            if actor.actor_name==actor_name:
                for animation in actor.animations:
                    if animation.animation_name==animation_name:
                        res+=frameID
                        return res
                    else:
                        res+=animation.num_of_frames
            else:
                res+=actor.num_of_frames
        assert(False) # we should not arrive here
        return -1

    def get_frame_by_number(self,num:int)->Frame:
        if self._num_to_frame == dict():
            return self.__get_frame_by_number(num=num)
        else:
            return self._num_to_frame[num]

    def get_number_by_frame(self,frame:Frame)->int:
        if self._frame_to_num == dict():
            return self.__get_number_by_frame(frame=frame)
        else:
            return self._frame_to_num[frame]

    def get_first_inclusive_and_last_exlusize_number_by_actor_name(self,actor_name:str)->(int,int):
        first=0
        for actor in self.actors:
            if actor.actor_name==actor_name:
                return first,first+actor.num_of_frames
            else:
                first+=actor.num_of_frames
        assert(False) #worng actor name we should not arrived here

    def test_dataset_mapping(self)->None:
        print('test_dataset_mapping')
        for num in tqdm.trange(0,self.num_of_frames['total']):
            cur_frame = self.__get_frame_by_number(num=num)
            new_num=self.__get_number_by_frame(cur_frame)
            assert(num==new_num)

    #TODO consider remove - can be found on actor
    def get_data_mat(self,theta_weights:torch.Tensor=None,comp_device:str=None)->torch.Tensor:
        data_matrix=self._data_mat
        #maybe write it better
        if theta_weights!=None:
            body_vec_dim=data_matrix.size(1) #it's V
            if comp_device!=None:
                theta_weights=theta_weights.to(comp_device)
            theta_weights=theta_weights.reshape(1,body_vec_dim)
            data_matrix=torch.mul(data_matrix,theta_weights) #multipication element-wize
        return data_matrix

    def create_cache_for_dataset(self)->None:
        #self.create_bi_directional_mapping()
        #self.create_data_mat()
        #for actor in self.actors:
        print('create cache for dataset {}'.format(self.dataset_name))
        for i in tqdm.trange(len(self.actors)):
            actor=self.actors[i]
            actor.save_cache_for_actor()

    def _change_root_dir_for_all_animations(self,new_root:str,old_root:str):
        for actor in self.actors:
            for animation in actor.animations:
                animation.change_root_dir(new_root=new_root,old_root=old_root)

    #hardcoded for now
    def fix_root_dir_for_local_machine_if_needed(self,new_root:str)->None:
        #check if this works
        cur_filepath=example_file=self.actors[0].animations[0].file_path
        if os.path.exists(cur_filepath):
            #we don't need
            return
        #file do not exists.try new path.
        #this method assumes that the root dir ends with dir that called 'amass_dir'
        last_root_dir_name='amass_dir'
        old_root=cur_filepath.split(last_root_dir_name)[0]+last_root_dir_name
        #relpath=os.path.relpath(cur_filepath, old_root) #we need to be os portabily
        assert(cur_filepath.startswith(old_root))
        relpath=cur_filepath[len(old_root)+1:]
        relpath=covert_path_to_current_os(relpath) #TODO add change unix to windows vice verse ADD !!!!
        new_file=os.path.join(new_root,relpath)
        assert(os.path.exists(new_file))
        #if we pass this assert we can start
        self._change_root_dir_for_all_animations(new_root=new_root,old_root=old_root)

    def get_actor_with_least_and_max_frame(self)->(Actor,Actor):
        actor_with_least_frames=self.actors[0]
        actor_with_max_frames=self.actors[0]
        min_frames=self.actors[0].num_of_frames
        max_frames=self.actors[0].num_of_frames
        for cur_actor in self.actors[1:]:
            if cur_actor.num_of_frames<min_frames:
                min_frames=cur_actor.num_of_frames
                actor_with_least_frames=cur_actor
            if cur_actor.num_of_frames>max_frames:
                max_frames=cur_actor.num_of_frames
                actor_with_max_frames=cur_actor
        return actor_with_least_frames,actor_with_max_frames

class Amass:
    def __init__(self,dataset_list:list):
        self.dataset_list = dataset_list
        self.num_of_actors = {}
        self.num_of_animations = {}
        self.num_of_frames = {}
        self.num_of_minutes_in_120_fps_video = {}
        for gender in get_gender_list() + ['total']:
            self.num_of_actors[gender]=0
            self.num_of_animations[gender]=0
            self.num_of_frames[gender]=0
            self.num_of_minutes_in_120_fps_video[gender]=0
        for dataset in self.dataset_list:
            for gender in get_gender_list() + ['total']:
                self.num_of_actors[gender]+=dataset.num_of_actors[gender]
                self.num_of_animations[gender]+=dataset.num_of_animations[gender]
                self.num_of_frames[gender]+=dataset.num_of_frames[gender]
                self.num_of_minutes_in_120_fps_video[gender]+=dataset.num_of_minutes_in_120_fps_video[gender]
    def get_datasets_by_key(self,dataset_should_be_on_list_func)->list:
        res = []
        for dataset in self.dataset_list:
            if dataset_should_be_on_list_func(dataset):
                res.append(dataset)
        return res

    def get_dataset_name_and_actor_name_for_actor_with_the_least_frames(self):
        #only for debbuging samples function
        min_frames=self.num_of_frames['total']*2 #inf
        min_frame_actor_name=''
        min_frame_dataset_name=''
        for dataset in self.dataset_list:
            for actor in dataset.actors:
                if actor.num_of_frames<min_frames:
                    min_frame_actor_name=actor.actor_name
                    min_frame_dataset_name=dataset.dataset_name
                    min_frames=actor.num_of_frames
        #maybe return a list
        return min_frame_actor_name,min_frame_dataset_name,min_frames

def split_path_into_list(model_npz:str,root_dataset_dir:str):
    return os.path.relpath(path=model_npz,start=root_dataset_dir).split(os.sep)
def get_datset_name_from_model_npz(model_npz:str,root_dataset_dir:str):
    return split_path_into_list(model_npz=model_npz,root_dataset_dir=root_dataset_dir)[0]
def get_actor_name_from_model_npz(model_npz:str):
    return os.path.split(model_npz)[0].split(sep=os.sep)[-1]
def get_animation_name_from_model_npz(model_npz:str):
    return os.path.split(model_npz)[-1][:-4]
def get_num_of_seconds_from_num_of_frames(num_of_frames:int,frame_per_second:int=120):
    return num_of_frames/frame_per_second
def get_num_of_minutes_from_num_of_frames(num_of_frames:int,frame_per_second:int=120):
    return get_num_of_seconds_from_num_of_frames(num_of_frames=num_of_frames,frame_per_second=frame_per_second)/60

