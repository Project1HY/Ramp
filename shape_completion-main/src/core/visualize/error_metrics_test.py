import sys
import trimesh
import metrics_calculate
from record_time import test_f_time
sys.path.insert(0,'..')
from core.data.loaders import get_mini_amass_dataset

def main():
    test_get_volumes_speed_test()

class mock_hp:
    def __init__(self,amass_ds_mode:str='reduced_fps_debug',
            split_data:str=tuple([1]),pppc_num:int=1,
            force_actor_alignment_on_data_split:bool=True):
        self.amass_ds_mode=amass_ds_mode
        self.split_data=split_data
        self.force_actor_alignment_on_data_split=force_actor_alignment_on_data_split
        self.pppc_num=pppc_num

def get_mock_ds_and_ldr():
    #ds=get_ds(hp=mock_hp())
    ds=get_mini_amass_dataset(hp=mock_hp())
    ldr=get_simple_loaders_from_ds(ds=ds)
    return ds,ldr

def get_ds(amass_ds_mode:str,pppc_num:int,
            split_data:str=tuple([1]),
            force_actor_alignment_on_data_split:bool=True):
    hp=mock_hp(amass_ds_mode=amass_ds_mode,pppc_num=pppc_num,
            split_data=split_data,
            force_actor_alignment_on_data_split=force_actor_alignment_on_data_split)
    return get_mini_amass_dataset(hp=hp)


def test_get_volumes_short():
    #ds=get_completion_dataset(mode='reduced_fps_debug')
    ds=get_mini_amass_dataset(hp=mock_hp())
    ldr=get_simple_loaders_from_ds(ds=ds)
    print(f'ldr len is {len(ldr)}')
    f=ds._f
    for batch in ldr:
        r=metrics_calculate.get_volumes(v=batch['gt'],f=f)
        v=batch['gt']
        r1=trimesh.Trimesh(vertices=v[0,:], faces=f, process=False)
        r2=r1.volume
        r=error_metrics_calculate.get_areas(v=batch['gt'],f=f)
        print('hi')


def get_attribute_using_trimesh(v,f,attr):
    res=[]
    for i in range(v.size(0)):
        res.append(getattr(trimesh.Trimesh(vertices=v[i,:], faces=f, process=False),attr))
    return res

def get_volume_using_trimesh(v,f):
    return get_attribute_using_trimesh(v=v,f=f,attr='volume')

def get_area_using_trimesh(v,f):
    return get_attribute_using_trimesh(v=v,f=f,attr='area')

def get_attribute_functions(ds):
    torch_volume_f = lambda batch: metrics_calculate.get_volumes(v=batch['gt'],f=ds._f)
    trimesh_volume_f = lambda batch: get_volume_using_trimesh(v=batch['gt'],f=ds._f)
    torch_volume_time_f = lambda batch:test_f_time(lambda:torch_volume_f(batch))
    trimesh_volume_time_f = lambda batch:test_f_time(lambda:trimesh_volume_f(batch))

    torch_area_f = lambda batch: metrics_calculate.get_areas(v=batch['gt'],f=ds._f)
    trimesh_area_f = lambda batch: get_area_using_trimesh(v=batch['gt'],f=ds._f)
    torch_area_time_f = lambda batch:test_f_time(lambda:torch_area_f(batch))
    trimesh_area_time_f = lambda batch:test_f_time(lambda:trimesh_area_f(batch))
    return {
            'torch_volume_f' :torch_volume_f ,
            'trimesh_volume_f' :trimesh_volume_f,
            'torch_volume_time_f' :torch_volume_time_f ,
            'trimesh_volume_time_f' :trimesh_volume_time_f,

            'torch_area_f' :torch_area_f ,
            'trimesh_area_f' :trimesh_area_f,
            'torch_area_time_f' :torch_area_time_f ,
            'trimesh_area_time_f' :trimesh_area_time_f
            }

def flatten(t):
    return [item for sublist in t for item in sublist]

def test_get_volumes_speed_test():
    ds=get_mini_amass_dataset(hp=mock_hp())
    ldr=get_simple_loaders_from_ds(ds=ds)
    #res={attr:{'preformance':[],'errors':{t:[]} for t in ['torch','trimesh']} for attr in ['volume','area']}
    attrs=['volume','area']
    metrics=['value','time']
    types=['torch','trimesh']
    res={attr:{metric:{t:[] for t in types} for metric in metrics } for attr in attrs}
    f=ds._f
    for batch in ldr:
        v=batch['gt']
        res['volume']['value']['trimesh'].append(get_attribute_using_trimesh(v=v,f=f,attr='volume'))
        res['volume']['value']['torch'].append(metrics_calculate.get_volumes(v=batch['gt'],f=f))
        res['volume']['time']['trimesh'].append(test_f_time(lambda: get_attribute_using_trimesh(v=v,f=f,attr='volume')))
        res['volume']['time']['torch'].append(test_f_time(lambda: metrics_calculate.get_volumes(v=batch['gt'],f=f)))

        res['area']['value']['trimesh'].append(get_attribute_using_trimesh(v=v,f=f,attr='area'))
        res['area']['value']['torch'].append(metrics_calculate.get_areas(v=batch['gt'],f=f))
        res['area']['time']['trimesh'].append(test_f_time(lambda: get_attribute_using_trimesh(v=v,f=f,attr='area')))
        res['area']['time']['torch'].append(test_f_time(lambda: metrics_calculate.get_areas(v=batch['gt'],f=f)))

    #convert relevant lists

    for attr in attrs:
        res[attr]['value']['trimesh']=flatten(res[attr]['value']['trimesh'])
        res[attr]['value']['torch']=flatten([t.tolist() for t in res[attr]['value']['torch']])


    avg=lambda l:sum(l)/len(l)
    minus=lambda l,k:[l[i]-k[i] for i in range(len(l))]
    abs_l=lambda l:[abs(l[i]) for i in range(len(l))]

    #final_res=dict()
    #final_res={attr:{for metric in metrics} for attr in attrs}
    class final_res:
        volume_avg_time_trimesh=avg(res['volume']['time']['trimesh'])
        volume_avg_time_torch=avg(res['volume']['time']['torch'])
        volume_error=abs_l(minus(res['volume']['value']['trimesh'],res['volume']['value']['torch']))

        area_avg_time_trimesh=avg(res['area']['time']['trimesh'])
        area_avg_time_torch=avg(res['area']['time']['torch'])
        area_error=abs_l(minus(res['area']['value']['trimesh'],res['area']['value']['torch']))

    for attr in attrs:
        avg_time_trimesh=getattr(final_res,f'{attr}_avg_time_trimesh')
        avg_time_torch=getattr(final_res,f'{attr}_avg_time_torch')
        #diff_time=avg_time_torch-avg_time_trimesh
        torch_faster_by=avg_time_trimesh/avg_time_torch
        error=getattr(final_res,f'{attr}_error')
        avg_error=avg(error)
        max_error=max(error)
        print(f'{attr} results:')
        print(f'errors')
        print(f'max error:{max_error}')
        print(f'avg error:{avg_error}')
        print(f'times')
        print(f'avg time for trimesh {avg_time_trimesh}')
        print(f'avg time for torch {avg_time_torch}')
        if torch_faster_by>1:
            print(f'torch is faster than trimesh by {torch_faster_by}')
        else:
            print(f'trimesh is faster than torch by {1/torch_faster_by}')
    print('a')

def get_simple_loaders_from_ds(ds,method:str='fp2Np_multiple_poses',batch_size:int=10,n_views:int=1,s_dynamic:bool=True): #ds is CompletionDataset.im not writing this directly due import errors
    full_split_aligned:list=[1]
    s_nums:list=[100]
    s_dynamic:list=[s_dynamic]
    s_shuffle:list=[False]
    s_transform:list=None
    global_shuffle:bool=False
    device:str='cpu'
    ds.set_n_views(n_views)
    n_channels:int=3
    ldr=ds.loaders(split=full_split_aligned, s_nums=s_nums,
                      s_transform=s_transform,
                      batch_size=batch_size, device=device, n_channels=n_channels,
                      method=method, s_shuffle=s_shuffle,
                      s_dynamic=s_dynamic)
    return ldr

if __name__=="__main__":
    main()
