import torch
from error_metrics_test import get_simple_loaders_from_ds
from error_metrics_test import get_ds
from get_objects_hardcoded_for_sets_base import get_valid_amass_ds_modes
from get_objects_hardcoded_for_sets_base import get_valid_methods,get_valid_prior_types

import sys
sys.path.insert(0, '..')
from core.geom.mesh.vis.base import plot_mesh_montage


def main():
    print('a')

def visualize_sample_method(method:str,num_of_samples:int=3):
    #amass_ds_mode='reduced_random_debug'#this is not really relvant here
    amass_ds_mode='reduced_fps'#this is not really relvant here
    n_views=5
    ds,ldr=get_ds_and_loader(amass_ds_mode=amass_ds_mode,
            method=method,n_views=n_views)
    samples=fetch_meshe_list_from_ldsr(ldr=ldr,num_of_samples=num_of_samples)
    for sample in samples:
        visuallize_single_sample(sample=sample,f=ds._f)

def visualize_batch(amass_ds_mode:str,num_of_samples:int=64,split_into_n_lists:int=4):
    #amass_ds_mode='reduced_fps'
    method='fp2Np_same_pose' #this is not really relvant here
    n_views=5                #this is not really relvant here too
    ds,ldr=get_ds_and_loader(amass_ds_mode=amass_ds_mode,
            method=method,n_views=n_views)
    samples=fetch_meshe_list_from_ldsr(ldr=ldr,num_of_samples=num_of_samples)
    assert(len(samples)==num_of_samples)
    mesh_vis_batch_size=num_of_samples//split_into_n_lists #it's ok that we take the floor
    sample_chunks = [samples[x:x+mesh_vis_batch_size] for x in range(0, num_of_samples, mesh_vis_batch_size)]
    for sample_chunk in sample_chunks:
        visuallize_sample_list_full_shapes(sample_list=sample_chunk,f=ds._f)

def get_ds_and_loader(amass_ds_mode:str,method:str,n_views:int):
    """get_ds_and_loader.
    get the dataset object and the relvant loader for the set of parameters

    Args:
        amass_ds_mode (str): amass_ds_mode, is the dataset we would use.
        valid ds modes are:'full', 'reduced_random', 'reduced_fps', 'reduced_fps_debug', 'reduced_random_debug'
        method (str): method is the sampling method we would use.
        valid methods are:[ 'fp2Np_same_pose','fp2Np_other_pose','fp2Np_multiple_poses' ]
        valid prior types:['pppc','single_full_shape']
        n_views (int): n_views the number of n_views.

    Returns:
        reutrn the ds and the ldr
    """
    assert(amass_ds_mode in get_valid_amass_ds_modes())
    assert(method in get_valid_methods())

    ds = get_ds(amass_ds_mode=amass_ds_mode,pppc_num=n_views)
    ldr=get_simple_loaders_from_ds(ds,method=method,batch_size=1,n_views=n_views)
    return ds,ldr

def fetch_meshe_list_from_ldsr(ldr,num_of_samples:int)->list:
    samples=[]
    for batch in ldr: #for each batch
        for i in range(batch['gt'].size(0)): #for each element on the batch
            if len(samples)>=num_of_samples:
                return samples
            cur_sample={k:v[i] for k,v in batch.items()}
            samples.append(cur_sample)
    return samples

def visuallize_sample_list_full_shapes(sample_list:list,f:torch.Tensor)->None:
    get_full_shape=lambda d:d['gt']
    full_shapes=[get_full_shape(d) for d in sample_list]
    plot_mesh_montage(vb=full_shapes,fb=f,strategy='mesh')

def visuallize_single_sample(sample:dict,f:torch.Tensor)->None:
    #this function should help us examine the diffrent valid_methods
    get_full_shape=lambda d:d['gt']
    get_partial_shape=lambda d:get_full_shape(d)[d['gt_mask'],:]
    prior_dict=sample['partial_point_cloudes']
    mesh_list=[sample]+[{'gt':prior_dict['gt'][i],'gt_mask':prior_dict['gt_mask'][i],} for i in range(len(prior_dict['gt']))] #list of dicts
    full_shapes=[get_full_shape(d) for d in mesh_list]
    patrial_shapes=[get_partial_shape(d) for d in mesh_list]
    n_rows=2
    n_cols=len(full_shapes)
    assert(len(full_shapes)==len(full_shapes))
    mesh_list=full_shapes+patrial_shapes
    strategy=['mesh']*n_cols+['spheres']*n_cols
    f=[f]*len(mesh_list)
    plot_mesh_montage(vb=mesh_list,fb=f,strategy=strategy,force_n_cols=n_cols,force_n_rows=n_rows)

def test_1():
    methods=get_valid_methods()
    for method in methods:
        print(f'visuallize methods {method}')
        visualize_sample_method(method=method)

def test_2():
    print(f'visualize_batch')
    visualize_batch(amass_ds_mode='reduced_fps',num_of_samples=64,split_into_n_lists=4)

if __name__=="__main__":
    print(f'visualize_method')
    test_1()
    #print(f'visualize_batch')
    #test_2()
