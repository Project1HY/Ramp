import torch
import tqdm
import random
import copy
from mapping_classes import Dataset,Actor
import psutil

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_minimum_distanse_vector_for_unsampled_frames(data_matrix:torch.Tensor,sampled_frames:list,unsampled_frames:list)->torch.Tensor:
    """
    this method used to sample one index from the data_matrix acurding to the sampled_frames and the unsampled_frames lists.
    this funciton assuming that all the data can fit in the memory that the data_matrix is on dims (N,V,S)
    notations:
        -A:the total number of frames in the data_matrix
        -V:the number of features pararms on the feature vector
        -N:the number of unsampled_frames list (the frames that we can sample from)
        -S:the number of sampled_frames list (frames that we sampled already)
    input:
        -datamat:the data matrix we are sampling. dims (A,V)
        -sampled_frames:the frames that we sampled already from the data_matrix.
        -unsampled_frames:the frames that we need to sample from
    output:
        -the minimum distance vector for each unsampled_frames
    """

    assert(len(unsampled_frames)>0)
    assert(len(sampled_frames)>0)
    """
    if len(sampled_frames)==0:
        first_sample=random.randint(a=0,b=len(unsampled_frames)) # sample randomly
    """
    body_vec_dim=data_matrix.size(1) #it's V
    unsampled_frame_vec=data_matrix[unsampled_frames,:].reshape(-1,body_vec_dim,1)  #dim (N,V,1)
    sampled_frame_vec=data_matrix[sampled_frames,:]                                 #dim (S,V)
    sampled_frame_vec=sampled_frame_vec.reshape(-1,body_vec_dim,1).transpose(0,2)   #dim (1,V,S)
    dist=sampled_frame_vec-unsampled_frame_vec                                      #dim (N,V,S)
    dist_for_each_sample=dist.norm(dim=1)                                           #dim (N,S)
    min_dist_for_each_sample=dist_for_each_sample.min(dim=1).values                 #dim (N)
    #return min_dist_for_each_sample.argmax().item() #return index in unsampled_frames
    return min_dist_for_each_sample #return the minimum distances for each sample

def farthest_sample_vector_loop(data_matrix:torch.Tensor,sampled_frames:list,unsampled_frames:list,unsampled_frames_batch_size:int)->int:
    """
    this method sample one element from the data_matrix acurding to the sampled_frames and the unsampled_frames lists.
    this funciton assuming that all the data can fit in the memory that the data_matrix is on dims (N,V,S)
    notations:
        -A:the total number of frames in the data_matrix
        -V:the number of features pararms on the feature vector
        -N:the number of unsampled_frames list (the frames that we can sample from)
        -S:the number of sampled_frames list (frames that we sampled already)
    input:
        -datamat:the data matrix we are sampling. dims (A,V)
        -sampled_frames:the frames that we sampled already from the data_matrix.
        -unsampled_frames:the frames that we need to sample from
        -unsampled_frames_batch_size:the batch size for each iteration of calling get_minimum_distanse_vector_for_unsampled_frames
    output:
        -the index on the unsampled_frames list that we choose to sample
    """
    min_dist_for_each_unsampled_frame=None
    unsampled_frames_batch=chunks(unsampled_frames,unsampled_frames_batch_size)
    for cur_unsampled_frames in unsampled_frames_batch:
        min_dist_for_each_unsampled_frame_on_batch=get_minimum_distanse_vector_for_unsampled_frames(data_matrix=data_matrix,
                sampled_frames=sampled_frames,unsampled_frames=cur_unsampled_frames)
        if min_dist_for_each_unsampled_frame==None:
            min_dist_for_each_unsampled_frame=min_dist_for_each_unsampled_frame_on_batch
        else:
            min_dist_for_each_unsampled_frame=\
                    torch.cat((min_dist_for_each_unsampled_frame,min_dist_for_each_unsampled_frame_on_batch))
    next_list_index_to_sample=min_dist_for_each_unsampled_frame.argmax().item()
    return next_list_index_to_sample

#TODO consider to remove this function compute_distanse_matrix_from_data_matrix.added another versrion of it on sampling_utils
def compute_distanse_matrix_from_data_matrix(data_matrix:torch.Tensor,all_frames_list:list)->torch.Tensor:
    body_vec_dim=data_matrix.size(1) #it's V
    unsampled_frame_vec_1=data_matrix[all_frames_list,:].reshape(-1,body_vec_dim,1)  #dim (N,V,1)
    unsampled_frame_vec_2=unsampled_frame_vec_1.transpose(0,2)                        #dim (1,V,N)
    dist_matrix=unsampled_frame_vec_1-unsampled_frame_vec_2                                  #dim (N,V,N)
    dist_matrix=dist_matrix.norm(dim=1)                                             #dim (N,N)
    return dist_matrix

def farthest_sample_with_dist_matrix(dist_matrix:torch.Tensor,sampled_frames:list,unsampled_frames:list)->int:
    """
    this method sample one index from the unsampled_frames acurding to the sampled_frames.
    this funciton assuming that all the data can fit in the memory a.k.a the dist_matrix that is on dims (N,N)
    notations:
        -N:the number of unsampled_frames list (the frames that we can sample from)
        -S:the number of sampled_frames list (frames that we sampled already)
    input:
        -dist_matrix:the distnace matrix we are using for sampling. dims (N,N)
        -sampled_frames:the frames that we sampled already from the dist_matrix. (len (S))
        -unsampled_frames:the frames that we need to sample from (currently we dont use it,can be used only for assertation)
    output:
        -the next index on unsampled_frames to sample
    """
    dist_matrix=dist_matrix[:,sampled_frames] # dims (N,S)
    dist_matrix=dist_matrix.min(dim=1).values # dims (N) that's the min distanse for each sample
    dist_matrix[sampled_frames]=-1            # avoid chosing duplicated frames
    return dist_matrix.argmax().item()        # that's the next_list_index_to_sample

def farthest_sampling_by_range_of_indcies(dataset:Dataset,num_of_frames_to_sample:int,
        first_idx:int,last_idx:int,iterations_per_sample:int,only_allow_samples_that_diveded_by:int,
        farthest_sampling_method:str,comp_device:str,theta_weights:torch.Tensor=None)->list:
    """
    this method sample group of indecies from the dataset data_matrix acurding to the input indecies.
    for each sampling iteration we:first sample a random index from the unsampled_frames list and afterwards sample multiple smaples with 'farthest sampling' methods
    input:
        -data_matrix:the data matrix we are sampling. dims (A,V)
        -num_of_frames_to_sample:the total number of frames
        -first_idx:first index we allow to sample from (inclusice) ,between [0,A)
        -last_idx:last index we allow to sample from (exlusive) ,between (1,A]
        -iterations_per_sample:the number if iterations for each sampling.
            make this parameter bigger if this function is not faling due memory issue.(It's spliting N into chunks)
    output:
        -the sampled frames
    """
    assert(first_idx<last_idx)
    assert(iterations_per_sample>=1)
    assert(iterations_per_sample>=1)
    assert(farthest_sampling_method in ['vectors_loop','distance_matrix'])
    unsampled_frames=list(range(first_idx,last_idx))
    unsampled_frames=[n for (i,n) in enumerate(unsampled_frames) if i%only_allow_samples_that_diveded_by==0]
    unsampled_frames_batch_size=(len(unsampled_frames)//iterations_per_sample)
    all_frames_list=copy.deepcopy(unsampled_frames)
    sampled_frames=[]
    first_sample=random.randint(a=0,b=len(unsampled_frames)-1) # sample randomly a number between [0,len(unsamplef rarma)]
    sampled_frames.append(unsampled_frames[first_sample])
    unsampled_frames.remove(unsampled_frames[first_sample])

    sample_func=None
    if farthest_sampling_method=='vectors_loop':
        data_matrix = dataset.get_data_mat(theta_weights=theta_weights).to(comp_device)
        def vectors_loop_sample(sampled_frames:list,unsampled_frames:list)->int:
            return farthest_sample_vector_loop(data_matrix=data_matrix,
                    sampled_frames=sampled_frames,unsampled_frames=unsampled_frames,
                    unsampled_frames_batch_size=unsampled_frames_batch_size)
        sample_func=vectors_loop_sample
    else: #'distance_matrix'
        dist_matrix=compute_distanse_matrix_from_data_matrix(
                data_matrix=dataset.get_data_mat(theta_weights=theta_weights).to(comp_device)
                ,all_frames_list=all_frames_list) # we use unsampled_frames_original to include the first sample we sampled already.
        def matrix_sample(sampled_frames:list,unsampled_frames:list)->int:
            sampled_frames=[(i-first_idx)//only_allow_samples_that_diveded_by for i in sampled_frames] #offset
            sample_index=farthest_sample_with_dist_matrix(dist_matrix=dist_matrix,
                        sampled_frames=sampled_frames,unsampled_frames=all_frames_list)
            sample_index_0=sample_index*only_allow_samples_that_diveded_by+first_idx #consider the offset
            sample_index_1=unsampled_frames.index(sample_index_0)        #find index on unsampled frames
            return sample_index_1
        sample_func=matrix_sample

    for _ in tqdm.trange(num_of_frames_to_sample-1): # we sampled one sample already 
        assert(len(unsampled_frames)>0) #make sure we got something to sample from
        next_list_index_to_sample=sample_func(sampled_frames=sampled_frames,unsampled_frames=unsampled_frames)
        next_sample=unsampled_frames[next_list_index_to_sample]
        unsampled_frames.remove(next_sample)
        sampled_frames.append(next_sample)
    #make sure we have the requested size of frames (redundant)
    assert(len(sampled_frames)==num_of_frames_to_sample)
    return sampled_frames

def farthest_sampling_by_actor_name(dataset:Dataset,num_of_frames_to_sample:int,
        actor_name:str,iterations_per_sample:int,only_allow_samples_that_diveded_by:int,
        farthest_sampling_method:str,comp_device:str,theta_weights:torch.Tensor=None)->list:
    #warps farthest_sampling_by_range_of_indcies
    first_idx,last_idx=dataset.get_first_inclusive_and_last_exlusize_number_by_actor_name(actor_name=actor_name)
    return farthest_sampling_by_range_of_indcies(dataset=dataset,num_of_frames_to_sample=num_of_frames_to_sample,
            first_idx=first_idx,last_idx=last_idx,iterations_per_sample=iterations_per_sample,
            only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
            farthest_sampling_method=farthest_sampling_method,comp_device=comp_device,theta_weights=theta_weights)

#rewrite simpler version of farthest_sampling_by_range_of_indcies for sample all of the actor frames,without theta_weights.
def farthest_sampling_all_frames_for_actor(actor:Actor,comp_device:str)->list:
    """
    this method sample group of indecies from the dataset data_matrix acurding to the input indecies.
    for each sampling iteration we:first sample a random index from the unsampled_frames list and afterwards sample multiple smaples with 'farthest sampling' methods.
    IMPORTENT NOTE:THIS METHOD ONLY WORKS WITH DIST MATRIX
    IMPORTENT NOTE 2:THIS METHOD RETURN THE SAMPLING FOR A GIVEN ACTOR WITHOUT DATASET OFFSET
    output:
        -the sampled frames
    """
    # calculate method
    num_of_frames_to_sample=actor.num_of_frames
    """
    if farthest_sampling_method==None:
        actor.
    """
    #assert(farthest_sampling_method in ['vectors_loop','distance_matrix'])
    #assert(farthest_sampling_method in ['distance_matrix'])
    unsampled_frames=list(range(num_of_frames_to_sample))
    all_frames_list=copy.deepcopy(unsampled_frames)
    #unsampled_frames_batch_size=(len(unsampled_frames)//iterations_per_sample)
    sampled_frames=[]
    first_sample=random.randint(a=0,b=len(unsampled_frames)-1) # sample randomly a number between [0,len(unsamplef rarma)]
    sampled_frames.append(unsampled_frames[first_sample])
    unsampled_frames.remove(unsampled_frames[first_sample])

    """
    sample_func=None
    if farthest_sampling_method=='vectors_loop':
        data_matrix=actor.get_data_mat().to(comp_device)
        def vectors_loop_sample(sampled_frames:list,unsampled_frames:list)->int:
            return farthest_sample_vector_loop(data_matrix=data_matrix,
                    sampled_frames=sampled_frames,unsampled_frames=unsampled_frames,
                    unsampled_frames_batch_size=unsampled_frames_batch_size)
        sample_func=vectors_loop_sample
    #else: #'distance_matrix'
    actor.save_dist_mat()
    """
    dist_matrix=actor.get_dist_mat().to(comp_device)
    def matrix_sample(sampled_frames:list,unsampled_frames:list)->int:
        sample_index=farthest_sample_with_dist_matrix(dist_matrix=dist_matrix,
                    sampled_frames=sampled_frames,unsampled_frames=all_frames_list)
        sample_index_1=unsampled_frames.index(sample_index)        #find index on unsampled frames
        return sample_index_1
    sample_func=matrix_sample

    for _ in tqdm.trange(num_of_frames_to_sample-1): # we sampled one sample already 
        assert(len(unsampled_frames)>0) #make sure we got something to sample from
        next_list_index_to_sample=sample_func(sampled_frames=sampled_frames,unsampled_frames=unsampled_frames)
        next_sample=unsampled_frames[next_list_index_to_sample]
        unsampled_frames.remove(next_sample)
        sampled_frames.append(next_sample)
    #make sure we have the requested size of frames (redundant)
    assert(len(sampled_frames)==num_of_frames_to_sample)
    #make sure we got a valid sampling (we sampled all the actor frames)
    assert(all_frames_list==sorted(sampled_frames))
    return sampled_frames

def farthest_sampling_per_actor(dataset:Dataset,num_of_frames_to_sample:int,comp_device:str,iterations_per_sample:int,only_allow_samples_that_diveded_by:int,farthest_sampling_method:str,theta_weights:torch.Tensor=None)->list:
    #maybe this funciton could be more general
    num_of_frames_to_sample_per_actor=((num_of_frames_to_sample//dataset.num_of_actors['total'])+1)
    sampled_frames=[]
    for i in tqdm.trange(0,dataset.num_of_actors['total'], position=1):
        actor_name=dataset.actors[i].actor_name #for each actor in dataset
        actor_sampled_frames=farthest_sampling_by_actor_name(dataset=dataset,\
                actor_name=actor_name,\
                num_of_frames_to_sample=num_of_frames_to_sample_per_actor,
                iterations_per_sample=iterations_per_sample,\
                only_allow_samples_that_diveded_by=only_allow_samples_that_diveded_by,
                comp_device=comp_device,farthest_sampling_method=farthest_sampling_method)
        sampled_frames+=actor_sampled_frames
    sampled_frames=sampled_frames[:num_of_frames_to_sample]
    #uncomment for debug
    print('method {}'.format(farthest_sampling_method))
    print('results')
    print(sampled_frames)
    return sampled_frames

"""
def get_memory_calulation_limit():->int:
    psutil.virtual_memory()
"""
