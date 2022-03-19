import smpl_segmentations
import torch
from smpl_segmentation_dict import highlight_segmentation
from human_mesh_utils import get_defult_dir_values
from error_metrics_test import get_mock_ds_and_ldr

def plot_trimseh_list(t_list:list)->None:
    import sys
    sys.path.insert(0, '..')
    from core.geom.mesh.vis.base import plot_mesh_montage
    from core.geom.mesh.vis.base import plot_mesh
    #for deug
    vb,fb=[],[]
    for t in t_list:
        vb.append(t.vertices)
        fb.append(t.faces)
    strategy='mesh'
    plot_mesh_montage(vb=vb,fb=fb,strategy=strategy)

def get_seg_manger():
    seg_f_name=get_defult_dir_values()['smpl_segmentation_file']
    segmentation_manger=smpl_segmentations.SegmentationManger(segmentation_dict_filepath=seg_f_name)
    return segmentation_manger


def get_error_bounding_box_point_clouds(points_1:torch.Tensor,points_2:torch.Tensor,):
    #this test will not assume any coorespondace between points but will assume that the 
    #nearest neighboor from each set of points is symetrical realtion
    #points1 size [8 x 3]
    #points2 size [8 x 3]
    points_1_old=points_1
    points_2_old=points_2
    assert(points_1.size(0)==8)
    assert(points_1.size(1)==3)
    assert(points_2.size(0)==8)
    assert(points_2.size(1)==3)
    points_1=points_1.unsqueeze(2)           #dims [ 8 x 3 x 1 ]
    points_2=points_2.unsqueeze(2)           #dims [ 8 x 3 x 1 ]
    points_2=points_2.transpose(0,2)         #dims [ 1 x 3 x 8 ]
    dist_matrix=points_1-points_2            #dims [ 8 x 3 x 8 ]
    dist_matrix=dist_matrix.norm(dim=1)      #dims [ 8 x 8 ]
    min_values,arg_mins=torch.min(dist_matrix,dim=1)       #dims [ 8 ]
    assert(sorted(arg_mins.tolist())==list(range(8)))
    error=min_values.norm(dim=0)
    return error



def test_segmentation_manger(display_meshes:bool=True,print_res:bool=True,err_area_tresh:int=0.1,err_vol_tresh:int=0.01,num_of_actors_for_each_batch:int=2,num_of_batches:int=1):
    # not very beautiful test..I know we will keep it for now.
    ds,ldr=get_mock_ds_and_ldr()
    segmentation_manger=get_seg_manger()
    def new_print(s:str):
        if print_res:
            print(s)
    for j,batch in enumerate(ldr):
        #for every batch in batch 
        if j>=num_of_batches:
            break
        #compute elements batch size
        segs_meshes_watertight=segmentation_manger.get_meshes_of_segments(batch['gt'],watertight_mesh=True,center=True)
        segs_meshes_not_watertight=segmentation_manger.get_meshes_of_segments(batch['gt'],watertight_mesh=False,center=True)
        segs_meshes_not_watertight_not_center=segmentation_manger.get_meshes_of_segments(
                batch['gt'],watertight_mesh=False,center=False)#for bounding points tests
        segs_volumes=segmentation_manger.get_volumes_of_segments(batch['gt'])
        segs_areas=segmentation_manger.get_areas_of_segments(batch['gt'])

        segs_bounding_boxes_points=segmentation_manger.get_bounding_box_points_of_segments(batch['gt'])
        segs_bounding_boxes_areas=segmentation_manger.get_areas_of_bounding_box_of_segments(batch['gt'])
        segs_bounding_boxes_vols=segmentation_manger.get_volumes_of_bounding_box_of_segments(batch['gt'])

        new_print(f'analays meshes on batch {j}')

        for i in range(batch['gt'].size(0)):
            #for every element in batch 
            if i>=num_of_actors_for_each_batch:
                break
            new_print(f'analays {i} meshe on batch {j}')
            hi=batch['gt_hi'][i]
            new_print(f'hi:{hi}')
            meshs_watertight={}
            meshs_not_watertight={}
            meshs_not_watertight_not_center={}
            vols={}
            areas={}
            bounding_boxes_points={}
            bounding_boxes_areas={}
            bounding_boxes_vols={}
            #allign dict for spesific actor
            for seg_name in segs_areas.keys():
                meshs_watertight[seg_name]=segs_meshes_watertight[seg_name][i]
                meshs_not_watertight[seg_name]=segs_meshes_not_watertight[seg_name][i]
                meshs_not_watertight_not_center[seg_name]=segs_meshes_not_watertight_not_center[seg_name][i]
                vols[seg_name]=segs_volumes[seg_name][i]
                areas[seg_name]=segs_areas[seg_name][i]
                bounding_boxes_points[seg_name]=segs_bounding_boxes_points[seg_name][i]
                bounding_boxes_areas[seg_name]=segs_bounding_boxes_areas[seg_name][i]
                bounding_boxes_vols[seg_name]=segs_bounding_boxes_vols[seg_name][i]

            new_print(f'segmentation vol for mesh {vols}')
            new_print(f'segmentation area for mesh {areas}')

            #area and vol comperason for intire mesh
            vol_total_seg_sum=0
            area_total_seg_sum=0

            for seg_name in vols.keys():
                if seg_name=='Full':
                    continue
                vol_value=vols[seg_name]
                area_value=areas[seg_name]
                vol_total_seg_sum+=vol_value
                area_total_seg_sum+=area_value

            new_print(f'sum all vols except full {vol_total_seg_sum}')
            vol_full_real=vols['Full']
            new_print(f'sum vol full (real) {vol_full_real}')
            vol_error=(vol_total_seg_sum-vol_full_real).item()
            new_print(f'vol error {vol_error}')

            new_print(f'sum all areas except full {area_total_seg_sum}')
            area_full_real=areas['Full']
            new_print(f'sum area full (real) {area_full_real}')
            area_error=(area_total_seg_sum-area_full_real).item()
            new_print(f'area error {area_error}')



            #bounding box check
            #new_print(f'segmentation bounding_box points for mesh {bounding_boxes_points}')
            """
            print('a')
            print(torch.sum(bounding_boxes_points['Head'],dim=0)/8) #head center of mass
            print(bounding_boxes_points['Head']) #head center of mass
            """
            real_bounding_points={seg_name:mesh.bounding_box.vertices for seg_name,mesh in meshs_not_watertight_not_center.items()}
            real_bounding_areas={seg_name:mesh.bounding_box.area for seg_name,mesh in meshs_not_watertight_not_center.items()}
            real_bounding_vols={seg_name:mesh.bounding_box.volume for seg_name,mesh in meshs_not_watertight_not_center.items()}


            bbx_areas_errors={abs(real_bounding_areas[seg_name]-bounding_boxes_areas[seg_name]) for seg_name in meshs_not_watertight.keys()}
            bbx_vols_errors={abs(real_bounding_vols[seg_name]-bounding_boxes_vols[seg_name]) for seg_name in meshs_not_watertight.keys()}

            bbx_err=lambda seg_name:get_error_bounding_box_point_clouds(points_1=torch.Tensor(real_bounding_points[seg_name])
                    ,points_2=torch.Tensor(bounding_boxes_points[seg_name]))#get_bounding_box_error
            bbx_points_errors={bbx_err(seg_name) for seg_name in meshs_not_watertight.keys()}


            new_print(f'segmentation bounding_box points error for mesh {bbx_points_errors}')
            new_print(f'segmentation bounding_box areas error for mesh {bbx_areas_errors}')
            new_print(f'segmentation bounding_box vols error for mesh {bbx_vols_errors}')

            """
            new_print(f'segmentation bounding_box areas for mesh {bounding_boxes_areas}')
            new_print(f'real segmentation bounding_box areas for mesh {real_bounding_areas}')
            new_print(f'segmentation bounding_box vols for mesh {bounding_boxes_vols}')
            new_print(f'real segmentation bounding_box vols for mesh {real_bounding_volume}')
            """

            #L2_bounding_box_error=

            if display_meshes:
                plot_trimseh_list(list(meshs_watertight.values()))
                plot_trimseh_list(list(meshs_not_watertight.values()))

            assert(vol_error<err_vol_tresh)
            assert(area_error<err_area_tresh)

def run_test_1():
    display_meshes=True
    print_res=True
    err_area_tresh=0.1
    err_vol_tresh=0.1
    num_of_actors_for_each_batch=1
    num_of_batches=1
    test_segmentation_manger(display_meshes=display_meshes,
            print_res=print_res,err_area_tresh=err_area_tresh,
            err_vol_tresh=err_vol_tresh,
            num_of_actors_for_each_batch=num_of_actors_for_each_batch,
            num_of_batches=num_of_batches)
def run_test_2():
    display_meshes=False
    print_res=True
    err_area_tresh=0.1
    err_vol_tresh=0.1
    num_of_actors_for_each_batch=10
    num_of_batches=3
    test_segmentation_manger(display_meshes=display_meshes,
            print_res=print_res,err_area_tresh=err_area_tresh,
            err_vol_tresh=err_vol_tresh,
            num_of_actors_for_each_batch=num_of_actors_for_each_batch,
            num_of_batches=num_of_batches)

def main():
    print('a')

if __name__=="__main__":
    main()
    run_test_1()
