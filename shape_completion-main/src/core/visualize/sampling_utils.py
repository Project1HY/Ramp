import tqdm
import matplotlib
from mapping_classes import Dataset,Actor,Animation,Frame
from save_load_obj import save_obj
from human_mesh_utils import is_valid_npz_file
from matplotlib import pyplot as plt
import numpy as np

def create_model_npz_file_from_sampling(dataset:Dataset,sampling:list,file_path:str)->None:
    print('create model npz file from sampling')
    res=dict()
    for index in tqdm.trange(len(sampling)):
        #for sample_num in sampling:
        sample_num=sampling[index]
        cur_frame=dataset.get_frame_by_number(sample_num)
        cur_model=cur_frame.get_full_model_npz_vector_in_numpy_for_frame()
        if res==dict():
            res=cur_model
        else:
            for k,v in res.items():
                res[k]=np.concatenate((v,cur_model[k]),axis=0)
    gender='neutral'#arbitrary
    mocap_framerate=120#arbitrary
    poses=np.concatenate((res['root_orient'],res['pose_body'],res['pose_hand']),axis=1)
    trans=res['trans']
    dmpls=res['dmpls']
    betas=res['betas'][1,:]

    np.savez(file=file_path,trans=trans,gender=gender,mocap_framerate=mocap_framerate,betas=betas,dmpls=dmpls,poses=poses)
    #debug purpuse
    assert(is_valid_npz_file(file_path))

def create_plot_from_sampling(dataset_name:str,sampling_method_name:str,sampling:list,n_bins:int,file_path:str,show_fig:bool=True)->None:
    plt.hist(sampling,n_bins) #TODO make x lim list
    title='sampling results for dataset {}.\n num of samples {}.\nmethod:{}'.format(dataset_name,len(sampling),sampling_method_name)
    plt.title(title, fontsize=10)
    plt.xlabel('frame numbers (num bins:{})'.format(n_bins))
    plt.ylabel('number of sampled frames')
    if show_fig:
        plt.show()
    plt.savefig(file_path)
