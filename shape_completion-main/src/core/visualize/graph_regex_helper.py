from computation_manager_test import get_ldr_and_err_mgr

_,err_comp_mgr=get_ldr_and_err_mgr()

def get_bounding_lists():
    #_,err_comp_mgr=get_ldr_and_err_mgr()
    print('s')

def make_spaced_str_from_str_list(str_list:str):
    return make_delimiter_str_from_str_list(str_list,' ')

def make_delimiter_str_from_str_list(str_list:str,delim:str):
    res=''
    for item in str_list:
        if item=='':
            continue
        res+=item+delim
    res=res[:-len(delim)] #remove last delim
    return res

def OR_regex_for_str_list(str_list:list):
    res=make_delimiter_str_from_str_list(str_list,'|')
    res=f'({res})'
    return res


def get_all_segments_OR(with_full:bool=False):
    seg_names_list=err_comp_mgr.get_segmentations_strings()
    if not with_full:
        seg_names_list.remove('Full')
    return OR_regex_for_str_list(seg_names_list)

def generate_points_regex_codes():
    #points:
    segements_strs=[get_all_segments_OR(with_full=True),get_all_segments_OR(with_full=False)]
    points_types=['all points']             #can be ['all points','bounding box points','center of mass point']
    centerazied_types=['centralized'] #can be ['centralized','']
    error_type=['l1 error','l2 error','l infinity']
    normalized_types=['normalized'] #can be ['normalized','']
    res_regex_code=[]
    #bit ugly for now
    for seg_str in segements_strs:
        for point_type in points_types:
            for center_str in centerazied_types:
                for err_str in error_type:
                    for norm_str in normalized_types:
                        param_list=[seg_str,point_type,center_str,err_str,norm_str,'error']
                        res_regex_code.append(make_spaced_str_from_str_list(param_list))
    return res_regex_code

def generate_shapes_regex_codes():
    #shapes:
    segements_strs=[get_all_segments_OR(with_full=True),get_all_segments_OR(with_full=False)]
    quantity_l=['volume','surface area','surface area to volume ratio']   #can be ['volume','surface area','surface area to volume ratio']
    bounding_box_l=[''] #can be ['bounding box','']
    normalized_types=['normalized'] #can be ['normalized','']
    res_regex_code=[]
    #bit ugly for now
    for seg_str in segements_strs:
        for quantity in quantity_l:
            for bounding_box in bounding_box_l:
                for norm_str in normalized_types:
                    param_list=[seg_str,quantity,bounding_box,norm_str,'error']
                    res_regex_code.append(make_spaced_str_from_str_list(param_list))
    return res_regex_code



def test2_get_all_metrics_list():
    metrics_list=err_comp_mgr.get_all_metrics_strings()
    seg_names_list=err_comp_mgr.get_segmentations_strings()
    res=''
    for metric in metrics_list:
        for seg_name in seg_names_list:
            res+=f'{seg_name} {metric}\n'
    print(res)

def print_list(l:list):
    str_to_print=''
    for item in l:
        str_to_print=f'{str_to_print}\n{item}'
    print(str_to_print)

def main1():
    #m_list=test2_get_all_metrics_list()
    points_l=generate_points_regex_codes()
    shapes_l=generate_shapes_regex_codes()
    print('points_l')
    print_list(points_l)
    print('shapes_l')
    print_list(shapes_l)

def main():
    print('a')

if __name__=="__main__":
    main1()
