import os
#from visualize.import_pickle import import_pickle_right_version
"""
from visualize.import_pickle_version import import_pickle_right_version
import_pickle_right_version()
#import pickle5 as pickle
#hardcoded for now
from sys import platform
def import_pickle_right_version()->None:
    if platform == "linux" or platform == "linux2":
        import pickle5 as pickle
    else:
        import pickle
    return
import_pickle_right_version()
"""
try:
    import pickle5 as pickle
except ImportError:
    import pickle


def save_obj(obj_to_save,f_name:str)->None:
    dir_name=os.path.split(f_name)[0]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(f_name,'wb') as handle:
        pickle.dump(obj_to_save,handle,protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(f_name:str):
    with open(f_name,'rb') as handle:
       res=pickle.load(handle)
    return res

def get_obj(obj_file_name:str,function_to_create_obj=None,overwriteCache=False):
    """
    usage example:
    return get_obj(obj_file_name=path_of_amass_obj,function_to_create_obj=lambda:getAmass(root_dataset_dir),overwriteCache=overwriteCache)
    """
    if function_to_create_obj==None:
        return None
    if not os.path.exists(obj_file_name) or overwriteCache:
        save_obj(obj_to_save=function_to_create_obj(),f_name=obj_file_name)
    obj=load_obj(f_name=obj_file_name)
    return obj
