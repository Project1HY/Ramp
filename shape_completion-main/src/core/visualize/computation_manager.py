import torch
import re
from smpl_segmentations import SegmentationManger

class ComputationTypeBase():
    def __init__(self):
        raise NotImplementedError
    def get_str(self)->str:
        raise NotImplementedError
    def get_tuple(self)->tuple:
        raise NotImplementedError
    def _generate_str_from_list(self,l:list)->str:
        if len(l)==0:
            return ''
        res=l[0]
        for term in l[1:]:
            if term=='':
                continue
            else:
                res+=f' {term}'
        return res
    def _eq_attr(self, other,attr:str)->bool:
        if isinstance(other, type(self)):
            return getattr(self,attr) == getattr(other,attr)
        return False
    def _eq_attr_list(self, other,attr_list:list)->bool:
        for attr in attr_list:
            if not self._eq_attr(other=other,attr=attr):
                return False
        return True

class ShapeComputationType(ComputationTypeBase):
    def __init__(self,only_bounding_box:bool,quantity:str,normalization:bool):
        self._bounding_box=only_bounding_box
        self._quantity=quantity
        self._normalization=normalization
    def get_str(self)->str:
        shape='bounding box' if self._bounding_box else ''
        normalization='normalized' if self._normalization else ''
        return super()._generate_str_from_list([self._quantity,shape,normalization])#+' errror'
    def get_tuple(self)->tuple:
        return (self._bounding_box,self._quantity,self._normalization)
    def __eq__(self, other)->bool:
        return super()._eq_attr_list(other,['_bounding_box','_quantity','_normalization'])

class PointsComputationType(ComputationTypeBase):
    def __init__(self,point_type:str,quantity:str,centralized:bool,normalization:bool):
        self._point_type=point_type
        self._quantity=quantity
        self._centralized=centralized
        self._normalization=normalization
    def get_str(self)->str:
        centralizion='centralized' if self._centralized else ''
        normalization='normalized' if self._normalization else ''
        return super()._generate_str_from_list([self._point_type,centralizion,self._quantity,normalization])
    def get_tuple(self)->tuple:
        return (self._point_type,self._quantity,self._centralized,self._normalization)
    def __eq__(self, other):
        return super()._eq_attr_list(other,['_point_type','_quantity','_centralized'])

def get_valid_error_computations_type_list():

    res=[]

    #bounding_box=['','bounding box']
    bounding_box=[False,True]
    quantity_for_shapes=['volume']
        # ,'surface area','surface area to volume ratio'] TODO: Enable more when needed
    #normalizations=['','normalized']
    normalizations=[False,True]
    for only_bounding_box in bounding_box:
        for quantity in quantity_for_shapes:
            for normalization in normalizations:
                res.append(ShapeComputationType(only_bounding_box=only_bounding_box,quantity=quantity,normalization=normalization))

    point_types=['all points']
    #,'center of mass point','bounding box points']TODO: Enable more when needed
    quantity_for_points=['l1','l2','l infinity']
    centralizions=[True] #TODO add false if wanted
    normalizations_points=[False] #TODO add true if wanted

    for point_type in point_types:
        for quantity in quantity_for_points:
            for centralizion in centralizions:
                for normalization in normalizations_points:
                    if point_type=='center of mass point' and  centralizion==True:
                        continue
                    res.append(PointsComputationType(point_type=point_type,quantity=quantity,centralized=centralizion,normalization=normalization))

    return res


def get_valid_error_computations_type_list_for_flow():
    res=[]

    #bounding_box=['','bounding box']
    bounding_box=[False]
    quantity_for_shapes=['volume']
        # ,'surface area','surface area to volume ratio'] TODO: Enable more when needed
    #normalizations=['','normalized']
    normalizations=[False]
    for only_bounding_box in bounding_box:
        for quantity in quantity_for_shapes:
            for normalization in normalizations:
                res.append(ShapeComputationType(only_bounding_box=only_bounding_box,quantity=quantity,normalization=normalization))
    return res

class ErrorComputationDiffManger():
    def __init__(self,f:torch.Tensor,segmentation_manger:SegmentationManger,computation_type_list:list=get_valid_error_computations_type_list()):
        assert(set([comp.get_str() for comp in computation_type_list]).issubset(set(comp.get_str() for comp in get_valid_error_computations_type_list())))
        #assert(set(computation_type_list).issubset(set(get_valid_error_computations_type_list()))) I can implement __hash__ to compare it.but this is really redundant right now..
        self._f=f
        self._computation_type_list=computation_type_list
        self._segmentation_manger=segmentation_manger
        #self._pairwise_dist_orders=[1,2,float('inf')]
        #self._pairwise_dist_dict={ i:torch.nn.PairwiseDistance(p=i) for i in self._pairwise_dist_orders }
    def get_f(self)->torch.Tensor:
        return self._f
    def get_computation_list(self)->list:
        return self._computation_type_list
    def get_segmentation_manger(self)->SegmentationManger:
        return self._segmentation_manger
    def get_all_metrics_strings(self)->list:
        return [comp.get_str() for comp in self._computation_type_list]
    def get_segmentations_strings(self)->list:
        return self._segmentation_manger.get_segmentation_list()
    def _get_compute_errors_dict(self,shape_1:dict,shape_2:dict)->dict:
        #NOTE: another idea that can be added in future works: check the errors only on vertceis we saw/not saw on the prior information.
        # now I'm think I'm logging enough data during each step.
        computation=Computation(shape_1=shape_1,shape_2=shape_2,
                f=self._f,segmentation_manger=self._segmentation_manger)#,pairwise_dist_dict=self._pairwise_dist_dict)
        for computation_type in self._computation_type_list:
            computation.compute(computation_type=computation_type)
        return computation.get_computation_res()
    def _make_flat_errors_dict(self,not_flat_metrics_error_dict:dict):
        res={}
        for metric_d_name,metric_d in not_flat_metrics_error_dict.items():
            for seg_name,seg_value in metric_d.items():
                new_key_name=f'{seg_name} {metric_d_name}'
                res[new_key_name]=seg_value
        return res

    def _mean_metrics_over_batch(self,flat_metrics_error_dict:dict):
        return {k:sum(v/v.size(0)) for k,v in flat_metrics_error_dict.items()}

    def get_compute_errors_dict(self,shape_1:dict,shape_2:dict, compute_mean=True):
        res=self._get_compute_errors_dict(shape_1=shape_1,shape_2=shape_2)
        res=self._make_flat_errors_dict(not_flat_metrics_error_dict=res)
        if compute_mean:
            res=self._mean_metrics_over_batch(flat_metrics_error_dict=res)
        return res

class Computation():
    def __init__(self,shape_1:dict,shape_2:dict,f:torch.Tensor,segmentation_manger:SegmentationManger):#,pairwise_dist_dict:dict):
        self._f=f
        self._original_shapes_dict={1:shape_1,2:shape_2} #original full shape 1 is gt and 2 is completion
        self._segmentation_manger=segmentation_manger
        self._cache={1:{},2:{}} #_computations
        self._res=dict()#{1:{},2:{}}
        self._shape_nums=[i for i in self._original_shapes_dict.keys()] # list of ints [1,2]
        #self._pairwise_dist_dict=pairwise_dist_dict

    def _get_valid_attrs(self):
        valid_attrs = ['volume','surface_area','surface_area_to_volume_ratio']
        valid_attrs_bbx = [f'{attr}_bounding_box' for attr in valid_attrs]
        valid_shape_attrs=valid_attrs+valid_attrs_bbx

        valid_points_attrs_not_center= ['all_points','center_of_mass_point','bounding_box_points']
        valid_points_attrs_center= [f'{attr}_centralized' for attr in valid_points_attrs_not_center]
        valid_points_attrs=valid_points_attrs_not_center+valid_points_attrs_center

        res=valid_shape_attrs+valid_points_attrs
        return res

#point_types=['all points','center of mass','bounding box points']
    def _get_all_points(self,shape_num)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='all points',f=lambda:self._segmentation_manger.get_points_of_segments(v=self._original_shapes_dict[shape_num]))
    def _get_center_of_mass_point(self,shape_num)->dict:
        def _get_center_of_mass_point_unsqueezed()->dict:
            d=self._segmentation_manger.get_center_of_mass_points_of_segments(v=self._original_shapes_dict[shape_num])
            d={k:v.unsqueeze(1) for k,v in d.items()} #unsqueeze for later diff computations
            return d
        return self._get_attr(shape_num=shape_num,attr_name='center of mass points',f=_get_center_of_mass_point_unsqueezed)
    def _get_bounding_box_points(self,shape_num)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='bounding box points',f=lambda:self._segmentation_manger.get_bounding_box_points_of_segments(v=self._original_shapes_dict[shape_num]))

    def center_points_with_center_of_mass(self,points_d:dict,center_of_mass_d:dict)->dict:
        return {seg_name:points_d[seg_name]-center_of_mass_d[seg_name] for seg_name in points_d.keys()}

    def _get_all_points_centralized(self,shape_num)->dict:
        def get_all_points_centralized_util()->dict:
            points_d=self._get_all_points(shape_num=shape_num)
            center_of_mass_d=self._get_center_of_mass_point(shape_num=shape_num)
            return self.center_points_with_center_of_mass(points_d=points_d,center_of_mass_d=center_of_mass_d)
        return self._get_attr(shape_num=shape_num,attr_name='all points centralized',f=lambda:get_all_points_centralized_util())

    def _get_bounding_box_points_centralized(self,shape_num:dict)->dict:
        def get_bounding_box_points_centralized_util()->dict:
            points_d=self._get_bounding_box_points(shape_num=shape_num)
            center_of_mass_d=self._get_center_of_mass_point(shape_num=shape_num)
            return self.center_points_with_center_of_mass(points_d=points_d,center_of_mass_d=center_of_mass_d)
        return self._get_attr(shape_num=shape_num,attr_name='bounding box points centralized',f=lambda:get_bounding_box_points_centralized_util())

    def _get_bounding_box(self,shape_num:int)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='bounding_box',f=lambda:self._segmentation_manger.get_bounding_box(v=self._original_shapes_dict[shape_num]))

    def _get_volume(self,shape_num:int)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='volume',f=lambda:self._segmentation_manger.get_volumes_of_segments(v=self._original_shapes_dict[shape_num]))

    def _get_volume_bounding_box(self,shape_num:int)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='bounding box volume',f=lambda:self._segmentation_manger.get_volumes_of_bounding_box_of_segments(v=self._original_shapes_dict[shape_num]))
    def _get_surface_area(self,shape_num:int)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='surface area',f=lambda:self._segmentation_manger.get_areas_of_segments(v=self._original_shapes_dict[shape_num]))

    def _get_surface_area_bounding_box(self,shape_num:int)->dict:
        return self._get_attr(shape_num=shape_num,attr_name='bounding box surface area',f=lambda:self._segmentation_manger.get_areas_of_bounding_box_of_segments(v=self._original_shapes_dict[shape_num]))

    def _compute_surface_area_to_volume(self,volume:dict,surface_area:dict)->dict:
        return {seg_name:surface_area[seg_name]/volume[seg_name] for seg_name in volume.keys()}

    def _get_surface_area_to_volume_util(self,is_bounding_box:bool,shape_num:int)->dict:
        bbx_prefix='bounding box ' if is_bounding_box else ''
        def get_surface_area_to_volume()->dict:
            if is_bounding_box:
                volume=self._get_volume_bounding_box(shape_num=shape_num)
                surface_area=self._get_surface_area_bounding_box(shape_num=shape_num)
            else:
                volume=self._get_volume(shape_num=shape_num)
                surface_area=self._get_surface_area(shape_num=shape_num)
            return self._compute_surface_area_to_volume(volume=volume,surface_area=surface_area)
        return self._get_attr(shape_num=shape_num,attr_name=bbx_prefix+'surface area to volume ratio',f=lambda:get_surface_area_to_volume())

    def _get_surface_area_to_volume_ratio_bounding_box(self,shape_num:int)->dict:
        return self._get_surface_area_to_volume_util(is_bounding_box=True,shape_num=shape_num)

    def _get_surface_area_to_volume_ratio(self,shape_num:int)->dict:
        return self._get_surface_area_to_volume_util(is_bounding_box=False,shape_num=shape_num)

    def _get_attr_for_both_shapes(self,attr_name:str)->(torch.Tensor,torch.Tensor):
        assert(attr_name in self._get_valid_attrs())
        f=getattr(self,f'_get_{attr_name}')
        return {i:f(i) for i in self._shape_nums}

    def _get_attr(self,shape_num:int,attr_name:str,f)->dict:
        #shape_num is number in [1,2]
        #f is the function that not take any input and return the quantity
        return f()
        # assert(shape_num in self._shape_nums)
        # if attr_name in self._cache[shape_num]:
        #     return self._cache[shape_num][attr_name]
        # else:
        #     res=f()
        #     self._cache[shape_num][attr_name]=res
        #     return res

    def _get_error_diff_from_quantities_dict(self,quantities_dict:dict,normalization_needed:bool)->dict:
        #for shapes
        diff_dict={seg_name:torch.abs(quantities_dict[1][seg_name]-quantities_dict[2][seg_name]) for seg_name in quantities_dict[1].keys()}
        if normalization_needed:
            diff_dict={seg_name:(diff_dict[seg_name]/quantities_dict[1][seg_name])*100 for seg_name in diff_dict.keys()}
        return diff_dict

    def _get_norm_of_diff_for_points_dict(self,diff_dist_dict:dict,order,normalization_needed:bool):
        #divide_by= 1 if not normalization_needed else 
        calculate_dist_norm=lambda value:torch.sum(torch.norm(value,dim=2,p=order),dim=1)
        res={seg_name:calculate_dist_norm(value) for seg_name,value in diff_dist_dict.items()}
        if normalization_needed:
            divide_by={seg_name:value.size(1) for seg_name,value in diff_dist_dict.items()}
            normalized=lambda seg_name:res[seg_name]/divide_by[seg_name]
            res={seg_name:normalized(seg_name) for seg_name,value in res.items()}
        return res

    def _get_error_diff_from_dist_dict(self,points_dict:dict,order,normalization_needed:bool)->dict:
        #for points
        diff_dist_dict=self._get_error_diff_from_quantities_dict(quantities_dict=points_dict,normalization_needed=False)
        error_dict=self._get_norm_of_diff_for_points_dict(diff_dist_dict=diff_dist_dict,order=order,normalization_needed=normalization_needed)
        return error_dict

    def _get_order(self,quantity:str):
        assert(quantity.startswith('l'))
        order=quantity
        #_,order=quantity.split(' ')
        if len(order)==2:
            return int(order[1])
        elif order=='l infinity':
            return float('inf')
        else:
            raise ValueError('unvalid order quantity string')


    def _computeShapeDiff(self,shape_computation_type:ShapeComputationType)->dict:
        have_bounding_box,quantity,normalization_needed=shape_computation_type.get_tuple()
        attr_name=re.sub(' normalized','',shape_computation_type.get_str())
        attr_name=re.sub(' ','_',attr_name)
        quantities_dict=self._get_attr_for_both_shapes(attr_name=attr_name)
        return self._get_error_diff_from_quantities_dict(quantities_dict=quantities_dict,normalization_needed=normalization_needed)

    def _computePointDiff(self,point_computation_type:PointsComputationType)->dict:
        #will save it on my cache
        point_type,quantity,centralized,normalization_needed=point_computation_type.get_tuple()
        points_attr_name=re.sub(' normalized','',point_computation_type.get_str())
        points_attr_name=re.sub(' '+quantity,'',points_attr_name)
        points_attr_name=re.sub(' ','_',points_attr_name)
        points_dict_for_both_shapes=self._get_attr_for_both_shapes(attr_name=points_attr_name)
        order=self._get_order(quantity=quantity)
        res=self._get_error_diff_from_dist_dict(points_dict_for_both_shapes,order,normalization_needed)
        #NOTE this can be efficent if we would cache the points distance results
        return res

    def compute(self,computation_type:ComputationTypeBase)->None:
        if type(computation_type)==ShapeComputationType:
            res=self._computeShapeDiff(shape_computation_type=computation_type)
        else:
            res=self._computePointDiff(point_computation_type=computation_type)
        err_name=f'{computation_type.get_str()} error'
        self._res[err_name]=res
        return

    def get_computation_res(self)->dict:
        return self._res
