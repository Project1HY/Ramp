import trimesh
import torch
import numpy as np
from get_human_template import get_human_model
from save_load_obj import load_obj
from smpl_segmentation_dict import get_segmentation
from smpl_segmentation_dict import get_valid_n_joints
from smpl_segmentation_dict import flatten
from metrics_calculate import get_areas,get_volumes,get_bounding_boxes_points
from metrics_calculate import get_bounding_box_areas,get_bounding_box_volumes
from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask

def get_faces_that_will_close_mesh_hole(boundry_edges:list)->list:
    """close_mesh_hole.

    Args:
        boundry_edges (list): list of boundry_edges (each element is array(2))

    Returns:
        list: new triangles to add
    """
    def sort_boundry_edges(boundry_edges:list)->list:
        #naive sort
        flip_edge=lambda edge:list(reversed(edge))
        boundry_edges=[list(e) for e in boundry_edges] #convert it to list
        res=[boundry_edges[0]]
        boundry_edges_len=len(boundry_edges)
        boundry_edges=boundry_edges[1:]
        while len(res)<boundry_edges_len:
            last_edge=res[-1]
            for edge in boundry_edges:
                if last_edge[1]==edge[0]:
                    boundry_edges.remove(edge)
                    res.append(edge)
                    break
                elif last_edge[1]==edge[1]:
                    boundry_edges.remove(edge)
                    res.append(flip_edge(edge))
                    break
                else:
                    continue
        return res

    boundry_edges_sorted=sort_boundry_edges(boundry_edges)

    get_v2_of_edge=lambda edge,v1:edge[1-edge.index(v1)]
    common_v=lambda edge1,edge2:list(set.intersection(set(edge1),set(edge2)))[0]
    last_v_to_triangle=lambda edge1,edge2:get_v2_of_edge(edge2,common_v(edge1,edge2))
    triangle_from_two_edges=lambda edge1,edge2:[edge1[0],edge1[1],last_v_to_triangle(edge1,edge2)]

    new_triangels=[]
    new_edges=[]
    first_vertex=boundry_edges_sorted[0][0]
    second_vertex=boundry_edges_sorted[0][1]
    last_vertex=boundry_edges_sorted[-1][0]
    new_edges.append([first_vertex,second_vertex]) # dummy new edge
    for i,edge in enumerate(boundry_edges_sorted):
        if i==0 or i==len(boundry_edges_sorted)-1:
            continue
        else:
            new_vertex=edge[-1]
            new_edges.append([first_vertex,new_vertex])
            new_triangels.append(triangle_from_two_edges(new_edges[-1],new_edges[-2]))

    return new_triangels

def get_alligned_faces_by_vertex_mask(f:torch.Tensor,vertices_mask:list)->torch.Tensor:
    #f dim [n_faces,3]
    #vertices_mask contain n_vertecies elements of boolians
    f=f.type(dtype=torch.int)
    faces_mask_cnt=[]
    seen_already=0
    for v in vertices_mask:
        faces_mask_cnt.append(seen_already)
        seen_already+=v

    faces_mask_cnt=torch.Tensor(faces_mask_cnt).type(torch.int)
    #convet to numpy
    faces_mask_cnt=faces_mask_cnt.numpy()
    f=f.numpy()
    res=faces_mask_cnt[f]
    #return to torch
    res=torch.Tensor(res)
    return res

def close_mesh(t:trimesh.Trimesh)->(np.ndarray,np.ndarray,list):
    edges_with_deg_idx=trimesh.grouping.group_rows(t.edges_sorted, require_count=1)
    edges_with_deg_1=t.edges_sorted[edges_with_deg_idx,:]
    #for debug

    list_of_bounderies=trimesh.graph.connected_components(edges_with_deg_1)
    faces_to_add=[]
    for boundry_v_idx in list_of_bounderies:
        boundry_edges=[edge for edge in edges_with_deg_1 if set(edge).issubset(boundry_v_idx)]
        faces_to_add+=get_faces_that_will_close_mesh_hole(boundry_edges)
        #boundry is nd_array of the boundery vertex indecies
    faces_new=torch.cat((torch.Tensor(t.faces),torch.Tensor(faces_to_add)),dim=0)
    #num_of_new_faces=faces_to_add.size(0)
    #generate vertex masko
    vertex_seg=sorted([int(i) for i in list(set(flatten(list(faces_new.tolist()))))]) #The sorting on the end is VERY IMPORTANT
    assert(vertex_seg==sorted(vertex_seg))
    vertices_mask=[v in vertex_seg for v in list(range(len(t.vertices)))]
    faces_new=get_alligned_faces_by_vertex_mask(faces_new,vertices_mask)
    vertices_new=t.vertices[vertex_seg,:]

    water_tight_mesh = trimesh.Trimesh(vertices=vertices_new, faces=faces_new,process=False) #make sure we close the mesh.
    trimesh.repair.fix_normals(water_tight_mesh) #neccery,this fix the winding problems and critical for volume calculations
    #TODO change return value
    assert(water_tight_mesh.is_watertight)
    assert(water_tight_mesh.is_winding_consistent)
    watertight_faces=water_tight_mesh.faces
    non_watertight_faces=watertight_faces[list(range(len(t.faces))),:]
    return watertight_faces,vertex_seg,non_watertight_faces

class Segmentation():
    def __init__(self,full_mesh:trimesh,segmentation:dict):
        seg_mesh = trimesh.Trimesh(vertices=full_mesh.vertices, faces=full_mesh.faces[segmentation['faces'],:],process=False)
        self._faces_watertight,self._vertex_seg,self._faces_non_watertight=close_mesh(seg_mesh)
    def get_vertex_seg(self):
        return self._vertex_seg
    def get_faces_watertight(self):
        return self._faces_watertight
    def get_faces_non_watertight(self):
        return self._faces_non_watertight

    def get_open_segment(self,v:torch.Tensor)->(torch.Tensor,torch.Tensor):
        v=v[:,self.get_vertex_seg(),:] #v is v_seg
        f=self.get_faces_non_watertight()
        return v,f

    def get_closed_segment(self,v:torch.Tensor)->(torch.Tensor,torch.Tensor):
        v=v[:,self.get_vertex_seg(),:] #v is v_seg
        f=self.get_faces_watertight()
        return v,f

    def get_center_of_mass(self,v_seg:torch.Tensor)->torch.Tensor:
        """get_center_of_mass_by_segment.
        get a tensor of center of mass for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the center of mass for for. dims:[batch_size x n_vertices_on_seg x 3]

        Returns:
            torch.Tensor: the center of mass of the segements for the input batch. the i'th entry represent the center of mass the i'th model in the batch [batch_size x 3]
        """
        seg_center_of_mass=(torch.sum(v_seg,dim=1)/v_seg.size(1)) # [batch_size x 3]
        return seg_center_of_mass

    def _center_mesh(self,v_seg:torch.Tensor)->torch.Tensor:
        #""_center_mesh.
        #center the mesh vertices acurding to the segment vertices
        #this function is calculating this quantety on the whole batch.
        #Args:
        #    v_seg (torch.Tensor): v_seg the input vertex batch (sliced to the relevnat segment). dims:[batch_size x n_vertices_on_seg x 3]

        #Returns:
        #    torch.Tensor: the vertecis of the segements for the input batch normlized acurding to the segment center of mass [batch_size]
        #""
        seg_center_of_mass=self.get_center_of_mass(v_seg=v_seg).unsqueeze(1) # [batch_size x 1 x  3]
        return v_seg-seg_center_of_mass

    def get_areas_of_segment(self,v:torch.Tensor)->torch.Tensor:
        """get_areas_of_segment.
        get a tensor of areas for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            torch.Tensor: the areas of the segements for the input batch. the i'th entry represent the segment area of the i'th model in the batch [batch_size]
        """
        v,f=self.get_open_segment(v=v)
        return get_areas(v=v,f=f)

    def get_bounding_box_points_of_segments(self,v:torch.Tensor)->torch.Tensor:
        """get_bounding_box_of_segment.
        get a tensor of get_bounding_box for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            torch.Tensor: the areas of the segements for the input batch. the i'th entry represent the segment bounding box of the i'th model in the batch [batch_size x n_vertices_in_bounding_box x 3]
        """
        v,_=self.get_open_segment(v=v)
        return get_bounding_boxes_points(v=v)

    def get_bounding_box_areas_of_segments(self,v:torch.Tensor)->torch.Tensor:
        """get_bounding_box_of_segment.
        get a tensor of get_bounding_box for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            torch.Tensor: the areas of the bounding box segements for the input batch. the i'th entry represent the segment bounding box area of the i'th model in the batch [batch_size ]
        """
        v,_=self.get_open_segment(v=v)
        return get_bounding_box_areas(v=v)

    def get_bounding_box_volumes_of_segments(self,v:torch.Tensor)->torch.Tensor:
        """get_bounding_box_volumes_of_segments.
        get a tensor of volumes of bounding box for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            torch.Tensor: the volume of the bounding box segements for the input batch. the i'th entry represent the segment bounding box volume of the i'th model in the batch [batch_size ]
        """
        v,_=self.get_open_segment(v=v)
        return get_bounding_box_volumes(v=v)

    def get_points_of_segments(self,v:torch.Tensor)->torch.Tensor:
        v,_=self.get_open_segment(v=v)
        return v

    def get_volumes_of_segment(self,v:torch.Tensor)->torch.Tensor:
        """get_volumes_of_segment.
        get a tensor of volumes for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            torch.Tensor: the volumes of the segements for the input batch. the i'th entry represent the segment volume of the i'th model in the batch [batch_size]
        """
        v,f=self.get_closed_segment(v=v)
        v=self._center_mesh(v_seg=v)
        return get_volumes(v=v,f=f)

    def get_mesh_of_segement(self,v:torch.Tensor,center:bool=True,watertight_mesh:bool=True,mask=None)->list:
        """get_mesh_of_segement.
        get a list of  for the given segment of the a mesh.
        this function is calculating this quantety on the whole batch.
        Args:
            t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]

        Returns:
            list: the list of the meshes of the segement for the input batch. the i'th entry represent the segement mesh of the i'th model in the batch [batch_size]
        """
        faces=self.get_faces_watertight() if watertight_mesh else self.get_faces_non_watertight()
        v = v[:,:,:3]
        res=[]
        if mask == None:
            v=v[:,self.get_vertex_seg(),:]
            if center:
                v=self._center_mesh(v_seg=v)

            for i in range(v.size(0)):
                seg_mesh = trimesh.Trimesh(vertices=v[i,:,:], faces=faces, process=False)
                res.append(seg_mesh)
        else:
            for i in range(v.size(0)):
                cur_mask = np.array(mask[i])
                segment = np.array(self.get_vertex_seg())
                actual_mask = np.intersect1d(cur_mask,segment)
                # assert False, f"actual_mask {actual_mask} shape {actual_mask.shape}"
                v_part,f_part = trunc_to_vertex_mask(v[i,:,:],faces,actual_mask)
                
                seg_mesh = trimesh.Trimesh(vertices=v_part, faces=f_part, process=False)
                res.append(seg_mesh)
        return res

class SegmentationManger():
    #segmantation manger for smpl
    def __init__(self,n_joints:int=6,include_full_segmentation:bool=True,segmentation_dict_filepath:str=None,organs=None):
        assert(n_joints in get_valid_n_joints())
        self._human_template=get_human_model(gender='neutral')
        self._segmentations=get_segmentation(n_joints=n_joints,include_full_segmentation=include_full_segmentation,seg_f_name=segmentation_dict_filepath,organs=organs)
        for seg_name,seg in self._segmentations.items():
            self._segmentations[seg_name]=Segmentation(full_mesh=self._human_template,segmentation=seg)
    def get_volumes_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_volumes_of_segment(v=v) for seg_name,seg in self._segmentations.items()}
    def get_areas_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_areas_of_segment(v=v) for seg_name,seg in self._segmentations.items()}
    def get_meshes_of_segments(self,v:torch.Tensor,watertight_mesh:str,center:bool=True,mask=None)->dict:
        return {seg_name:seg.get_mesh_of_segement(v=v,watertight_mesh=watertight_mesh,center=center,mask=mask) for seg_name,seg in self._segmentations.items()}
    def get_segmentation_list(self)->list:
        return [seg_name for seg_name in self._segmentations.keys()]

    # get points

    def get_points_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_points_of_segments(v=v) for seg_name,seg in self._segmentations.items()}
    def get_bounding_box_points_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_bounding_box_points_of_segments(v=v) for seg_name,seg in self._segmentations.items()}
    def get_center_of_mass_points_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_center_of_mass(v_seg=seg.get_points_of_segments(v=v)) for seg_name,seg in self._segmentations.items()}

    # get bounding box attributes 

    def get_areas_of_bounding_box_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_bounding_box_areas_of_segments(v=v) for seg_name,seg in self._segmentations.items()}

    def get_volumes_of_bounding_box_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_bounding_box_volumes_of_segments(v=v) for seg_name,seg in self._segmentations.items()}

    # bounding box attributes
    """
    def get_bounding_box_area_of_segments(self,v:torch.Tensor)->dict:
        return {seg_name:seg.get_bounding_box_of_segment(v=v) for seg_name,seg in self._segmentations.items()}
    """
