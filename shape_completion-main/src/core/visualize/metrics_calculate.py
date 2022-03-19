import torch
import numpy as np

def _get_vertices_batch(v:torch.Tensor,
        f:np.array)->(torch.Tensor,torch.Tensor,torch.Tensor):
    """get_vertices_batch.
    return three tensors that represents the first,second and third vertcies.
    v1,v2,v3
    Args:
        t (torch.Tensor): t the input batch to calculate the vertecis for. dims:[batch_size x n_vertices x 3]
        f (np.array): f the faces for each model in the batch.assuming all the model in the batch have the same faces connectivity. dims: [n_faces x 3]

    Returns:
        torch.Tensor:v1 dims [batch_size x n_faces x 3]
        torch.Tensor:v2 dims [batch_size x n_faces x 3]
        torch.Tensor:v3 dims [batch_size x n_faces x 3]
    """
    v1,v2,v3=v[:,f[:,0],:],v[:,f[:,1],:],v[:,f[:,2],:] # v1,v2,v3 dims [batch_size x n_faces x 3]
    return v1,v2,v3

def get_volumes(v:torch.Tensor,f:np.array)->torch.Tensor:
    """get_volumes.
    get a tensor of volumes for each mesh on a batch.
    this function is calculating this quantety on the whole batch.
    this function assumes that the meshs on this batch are closed (a.k.a watertight).
    this function assumes that each model on the mesh is centralized on the orgin.
    Args:
        t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]
        f (np.array): f the faces for each model in the batch.assuming all the model in the batch have the same faces connectivity. dims: [n_faces x 3]

    Returns:
        torch.Tensor: the volume for the input batch. the i'th entry represent the volume of the i'th model in the batch [batch_size]
    """
    v1,v2,v3=_get_vertices_batch(v=v,f=f)
    # print(f"shape of v1 {v1.shape} shape of v2 {v2.shape}")
    # assert False,f"shape of v1 cross v2 is {torch.cross(v1,v2,dim=2)}"
    return torch.sum(torch.mul(torch.cross(v1,v2,dim=2),v3),dim=(1,2))/6

def get_areas(v:torch.Tensor,f:np.array)->torch.Tensor:
    """_get_areas.
    get a tensor of areas for each mesh on a batch.
    this function is calculating this quantety on the whole batch.
    Args:
        t (torch.Tensor): t the input batch to calculate the volume for. dims:[batch_size x n_vertices x 3]
        f (np.array): f the faces for each model in the batch.assuming all the model in the batch have the same faces connectivity. dims: [n_faces x 3]

    Returns:
        torch.Tensor: the areas for the input batch. the i'th entry represent the area of the i'th model in the batch [batch_size]
    """
    v1,v2,v3=_get_vertices_batch(v=v,f=f)
    return torch.sum(torch.linalg.norm(torch.cross(v1-v2,v1-v3,dim=2),dim=2,ord=2)/2,dim=1)

def get_bounding_boxes_points(v:torch.Tensor)->torch.Tensor:
    """get_bounding_boxes_points
    get a tensor of bounding box for each mesh on a batch.
    this function is calculating this quantety on the whole batch.
    Args:
        v (torch.Tensor): v the input batch to calculate the bounding box for. dims:[batch_size x n_vertices x 3]

    Returns:
        torch.Tensor: the bounding box for the input batch. the i'th entry represent the bounding box of the i'th model in the batch [batch_size x n_vertices_in_bounding_box x 3]
        when n_vertices_in_bounding_box is 8
    """
    #TODO test this function

    v_x=v[:,:,0] #[batch_size x n_vertices]
    v_y=v[:,:,1] #[batch_size x n_vertices]
    v_z=v[:,:,2] #[batch_size x n_vertices]

    min_x=torch.min(v_x,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]
    min_y=torch.min(v_y,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]
    min_z=torch.min(v_z,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]
    max_x=torch.max(v_x,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]
    max_y=torch.max(v_y,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]
    max_z=torch.max(v_z,dim=1).values.unsqueeze(1).unsqueeze(1) #[batch_size x 1 x 1]

    min_x_min_y_min_z=torch.cat((min_x,min_y,min_z),dim=2) #[batch_size x 1 x 3]
    min_x_min_y_max_z=torch.cat((min_x,min_y,max_z),dim=2) #[batch_size x 1 x 3]
    min_x_max_y_min_z=torch.cat((min_x,max_y,min_z),dim=2) #[batch_size x 1 x 3]
    min_x_max_y_max_z=torch.cat((min_x,max_y,max_z),dim=2) #[batch_size x 1 x 3]
    max_x_min_y_min_z=torch.cat((max_x,min_y,min_z),dim=2) #[batch_size x 1 x 3]
    max_x_min_y_max_z=torch.cat((max_x,min_y,max_z),dim=2) #[batch_size x 1 x 3]
    max_x_max_y_min_z=torch.cat((max_x,max_y,min_z),dim=2) #[batch_size x 1 x 3]
    max_x_max_y_max_z=torch.cat((max_x,max_y,max_z),dim=2) #[batch_size x 1 x 3]

    bounding_box=torch.cat((
        min_x_min_y_min_z,
        min_x_min_y_max_z,
        min_x_max_y_min_z,
        min_x_max_y_max_z,
        max_x_min_y_min_z,
        max_x_min_y_max_z,
        max_x_max_y_min_z,
        max_x_max_y_max_z),dim=1)

    return bounding_box

def get_a_b_c_lengths_from_bounding_box(bounding_box:torch.Tensor)->(torch.Tensor,torch.Tensor,torch.Tensor):
    """get_a_b_c_lengths_from_bounding_box.
    get the lengths for a bounding box
    this function is calculating this quantety on the whole batch.

    Args:
        bounding_boxe (torch.Tensor): bounding_box the input batch of the bounding_box to calculate the length for
                                    [batch_size x n_vertices_in_bounding_box x 3] ,when n_vertices_in_bounding_box is 8

    Returns:
        (torch.Tensor,torch.Tensor,torch.Tensor):each tensor represent a diamter length [batch_size].
    """
    min_x=bounding_box[:,0,0]
    min_y=bounding_box[:,0,1]
    min_z=bounding_box[:,0,2]

    max_x=bounding_box[:,-1,0]
    max_y=bounding_box[:,-1,1]
    max_z=bounding_box[:,-1,2]

    a=max_x-min_x
    b=max_y-min_y
    c=max_z-min_z

    return a,b,c

#NOTE the bounding box attr functions can be implemented bit better,if we would cache the bounding box points

def get_bounding_box_areas(v:torch.Tensor)->torch.Tensor:
    """get_bounding_boxes_points
    get a tensor of area bounding box for each mesh on a batch.
    this function is calculating this quantety on the whole batch.
    Args:
        v (torch.Tensor): v the input batch of the mesh to calculate the bounding box area for. dims:[batch_size x n_vertices x 3]

    Returns:
        torch.Tensor: the bounding box areas for the input batch. the i'th entry represent the area of the bounding box of the i'th model in the batch [batch_size ]
    """
    a,b,c=get_a_b_c_lengths_from_bounding_box(bounding_box=get_bounding_boxes_points(v=v)) #a,b,c dims are [batch_size]
    return 2*(a*b+a*c+b*c) # [batch_size]

def get_bounding_box_volumes(v:torch.Tensor)->torch.Tensor:
    """get_bounding_boxes_points
    get a tensor of volume bounding box for each mesh on a batch.
    this function is calculating this quantety on the whole batch.
    Args:
        v (torch.Tensor): v the input batch of the mesh to calculate the bounding box volumefor. dims:[batch_size x n_vertices x 3]

    Returns:
        torch.Tensor: the bounding box areas for the input batch. the i'th entry represent the volume of the bounding box of the i'th model in the batch [batch_size ]
    """
    a,b,c=get_a_b_c_lengths_from_bounding_box(bounding_box=get_bounding_boxes_points(v=v)) #a,b,c dims are [batch_size]
    return a*b*c # [batch_size]

