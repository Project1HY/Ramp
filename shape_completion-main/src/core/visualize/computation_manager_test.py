from computation_manager import ErrorComputationDiffManger
from smpl_segmentations_test import get_seg_manger
from error_metrics_test import get_mock_ds_and_ldr

def main():
    print('a')
def get_ldr_and_err_mgr():
    ds,ldr=get_mock_ds_and_ldr()
    segmentation_manger=get_seg_manger()
    err_mgr=ErrorComputationDiffManger(ds._f,segmentation_manger=segmentation_manger)
    return ldr,err_mgr

def test1():
    ldr,err_comp_mgr=get_ldr_and_err_mgr()
    for batch in ldr:
        shape_1=batch['gt']
        shape_2=batch['gt']+1 # translate the original mesh a bit
        res=err_comp_mgr.get_compute_errors_dict(shape_1=shape_1,shape_2=shape_2)
        #return
        #print(res)
def test2_get_strs():
    ldr,err_comp_mgr=get_ldr_and_err_mgr()
    metrics_list=err_comp_mgr.get_all_metrics_strings()
    seg_names_list=err_comp_mgr.get_segmentations_strings()
    res=''
    """
    for metric in metrics_list:
        for seg_name in seg_names_list:
        res+=f'{smetrics_liste{metric}\n'
        #TODO FINISH THIS
    """
    print(res)



if __name__=="__main__":
    #main()
    #test1()
    test2_get_strs()


