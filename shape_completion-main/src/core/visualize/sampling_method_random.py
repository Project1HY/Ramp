import random
from mapping_classes import Dataset

def sample_random(dataset:Dataset,num_of_frames_to_sample:int)->list:
    res=random.sample(range(0,dataset.num_of_frames['total']),num_of_frames_to_sample)
    return res
