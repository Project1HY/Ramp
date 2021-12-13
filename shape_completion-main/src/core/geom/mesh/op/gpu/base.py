import torch
import cfg


# ---------------------------------------------------------------------------------------------------------------------#
#                                            Torch Singleton Computes
# ---------------------------------------------------------------------------------------------------------------------#

def torch_mesh_volume(v, f):
    # TODO: Does not work
    v1 = v[:, f[:, 0], :]
    v2 = v[:, f[:, 1], :]
    v3 = v[:, f[:, 2], :]
    a_vec = torch.cross(v2 - v1, v3 - v1, -1)
    center = (v1 + v2 + v3) / 3
    volume = torch.sum(a_vec * center / 6, dim=(1, 2))
    return volume


def torch_vf_adjacency(faces, n_faces, n_verts):
    # TODO: Does not work
    """
    :param faces: dim: [N_faces x 3]
    :param n_faces: number of faces
    :param n_verts: number of vertices
    :param device: device to place tensors
    :return: adjacency_vf: sparse integer adjacency matrix between vertices and faces, dim: [N_vertices x N_faces]
    """
    fvec = torch.arange(n_faces)
    i0 = torch.stack((faces[:, 0], fvec), dim=1)
    i1 = torch.stack((faces[:, 1], fvec), dim=1)
    i2 = torch.stack((faces[:, 2], fvec), dim=1)
    ind = torch.cat((i0, i1, i2), dim=0)
    ones_vec = torch.ones([3 * n_faces], dtype=torch.int8)
    adjacency_vf = torch.sparse.BoolTensor(ind.t(), ones_vec, torch.Size([n_verts, n_faces]))
    return adjacency_vf


# ---------------------------------------------------------------------------------------------------------------------#
#                                               Torch Batch Computes
# ---------------------------------------------------------------------------------------------------------------------#
def vertex_velocity(v_comp_t, v_ground_t):
    """
    Define a vertex velocity metric
    :param v_comp_t: [b, nv, 3] tensor that holds the vertices locations of the completion at a certain time
    :param v_comp_t_n: [b, nv, 3] tensor that holds the vertices locations of the completion at a consecutive time
    :param v_ground_t: [b, nv, 3] tensor that holds the vertices locations of the ground truth at a certain time
    :param v_ground_t_n: [b, nv, 3] tensor that holds the vertices locations of the ground truth at a consecutive time
    """
    shifted_v_comp = torch.cat((torch.zeros(1, *v_comp_t.shape[1:]), v_comp_t))[:-1, :, :]
    shifted_v_ground = torch.cat((torch.zeros(1, *v_ground_t.shape[1:]), v_comp_t))[:-1, :, :]

    velocity_comp = v_comp_t - shifted_v_comp
    velocity_ground = v_ground_t - shifted_v_ground
    diff = torch.norm(velocity_ground - velocity_comp)
    return diff


def batch_surface_area(vb, fb):
    """
    Compute the surface area of the batch of triangle meshes defined by v and f
    :param v: A [b, nv, 3] tensor where each [i, :, :] are the vertices of a mesh
    :param f: A [b, nf, 3] tensor where each [i, :, :] are the triangle indices into v[i, :, :] of the mesh
    :return: A tensor of shape [b, 1] with the surface area of each mesh
    """
    idx = torch.arange(vb.shape[0])
    tris = vb[:, fb, :][idx, idx, :, :]
    a = tris[:, :, 1, :] - tris[:, :, 0, :]
    b = tris[:, :, 2, :] - tris[:, :, 0, :]
    areas = torch.sum(torch.norm(torch.cross(a, b, dim=2), dim=2) / 2.0, dim=1)
    return areas


def batch_surface_volume(vb, fb):
    """
    Compute the surface area of the batch of triangle meshes defined by v and f
    :param v: A [b, nv, 3] tensor where each [i, :, :] are the vertices of a mesh
    :param f: A [b, nf, 3] tensor where each [i, :, :] are the triangle indices into v[i, :, :] of the mesh
    :return: A tensor of shape [b, 1] with the surface area of each mesh
    """
    idx = torch.arange(vb.shape[0])
    tris = vb[:, fb, :][idx, 0, :, :]
    volume = (tris[:, :, 0, :] * torch.cross(tris[:, :, 1, :], tris[:, :, 2, :], dim=-1)).sum(dim=-1) / 6.0
    return volume


def batch_moments(vb):
    # TODO - Implement
    # x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
    raise NotImplementedError


def batch_fnrmls_fareas(vb, f, return_normals=True):
    """ # TODO - Allow also [n_verts x 3]. Write another function called batch_fnrmls if we only need those
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix,we assume all the shapes have the same connectivity, dim: [n_faces x 3], dtype = torch.long
    :param return_normals : Whether to return the normals or not
    :return face_normals_b: batch of face normals, dim: [batch_size x n_faces x 3]
            face_areas_b: batch of face areas, dim: [batch_size x n_faces x 1]
            is_valid_fnb: boolean matrix indicating if the normal is valid,
            magnitude greater than zero [batch_size x n_faces].
            If the normal is not valid we return [0,0,0].
    """

    # calculate xyz coordinates for 1-3 vertices in each triangle
    v1 = vb[:, f[:, 0], :]  # dim: [batch_size x n_faces x 3]
    v2 = vb[:, f[:, 1], :]  # dim: [batch_size x n_faces x 3]
    v3 = vb[:, f[:, 2], :]  # dim: [batch_size x n_faces x 3]

    face_normals_b = torch.cross(v2 - v1, v3 - v2)
    face_areas_b = torch.norm(face_normals_b, dim=2, keepdim=True) / 2
    if not return_normals:
        return face_areas_b

    is_valid_fnb = (face_areas_b.squeeze(2) > (cfg.NORMAL_MAGNITUDE_THRESH / 2))
    fnb_out = torch.zeros_like(face_normals_b)
    fnb_out[is_valid_fnb, :] = face_normals_b[is_valid_fnb, :] / (2 * face_areas_b[is_valid_fnb, :])
    return fnb_out, face_areas_b, is_valid_fnb


def batch_vnrmls(vb, f, return_f_areas=False):
    """
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix, here we assume all the shapes have the same sonnectivity, dim: [n_faces x 3]
    :param return_f_areas: Whether to return the face areas or not
    :return: vnb:  batch of shape normals, per vertex, dim: [batch_size x n_vertices x 3]
    :return: is_valid_vnb: boolean matrix indicating if the normal is valid, magnitude greater than zero
    [batch_size x n_vertices].
    If the normal is not valid we return [0,0,0].
    :return face_areas_b (optional): a batch of face areas, dim: [batch_size x n_faces x 1]
    """

    n_faces = f.shape[0]
    n_batch = vb.shape[0]

    face_normals_b, face_areas_b, is_valid_fnb = batch_fnrmls_fareas(vb, f)
    # non valid face normals are: [0, 0, 0], due to batch_fnrmls_fareas
    face_normals_b *= face_areas_b  # weight each normal with the corresponding face area

    face_normals_b = face_normals_b.repeat(1, 3, 1)  # repeat face normals 3 times along the face dimension
    f = f.t().contiguous().view(3 * n_faces)  # dim: [n_faces x 3] --> [(3*n_faces)]
    f = f.expand(n_batch, -1)  # dim: [B x (3*n_faces)]
    f = f.unsqueeze(2).expand(n_batch, 3 * n_faces, 3)
    # dim: [B x (3*n_faces) x 3], last dimension (xyz dimension) is repeated

    # For each vertex, sum all the normals of the adjacent faces (weighted by their areas)
    vnb = torch.zeros_like(vb)  # dim: [batch_size x n_vertices x 3]
    vnb = vnb.scatter_add(1, f, face_normals_b)  # vb[b][f[b,f,xyz][xyz] = face_normals_b[b][f][xyz]

    magnitude = torch.norm(vnb, dim=2, keepdim=True)
    is_valid_vnb = (magnitude > cfg.NORMAL_MAGNITUDE_THRESH).squeeze(2)
    vnb_out = torch.zeros_like(vb)
    vnb_out[is_valid_vnb, :] = vnb[is_valid_vnb, :] / magnitude[is_valid_vnb, :]
    # check the sum of face normals is greater than zero

    return (vnb_out, is_valid_vnb, face_areas_b) if return_f_areas else (vnb_out, is_valid_vnb)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Test Suite
# ----------------------------------------------------------------------------------------------------------------------

def _bring_in_test_data():
    from data.sets import DatasetMenu
    from data.transforms import Center
    ds = DatasetMenu.order('FaustPyProj')
    samp = ds.sample(num_samples=5, transforms=[Center()], method='f2p')  # dim:
    vb = samp['gt'][:, :, :3]
    f = torch.from_numpy(ds.faces()).long()
    return vb, f


def _test_vnrmls_grad():
    vb, f = _bring_in_test_data()
    # N_faces = faces.shape[0]
    # N_vertices = batch_v.shape[1]
    # adjacency_vf = vf_adjacency(faces, N_faces, N_vertices)
    # This operation can be calculated once for the whole training

    from torch.autograd import gradcheck
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    x = (vb.requires_grad_(True).double(), f)
    test = gradcheck(batch_vnrmls, x, eps=1e-6, atol=1e-4, check_sparse_nnz=True)
    print(test)


def _test_vnrmls_visually():
    from geom.mesh.vis.base import plot_mesh
    vb, f = _bring_in_test_data()
    # adjacency_VF = vf_adjacency(faces, n_faces, n_verts)
    # This operation can be calculated once for the whole training
    vertex_normals, is_valid_vnb = batch_vnrmls(vb, f)
    # There exist 2 implementations for batch_vnrmls, batch_vnrmls_ uses adjacency_VF while batch_vnrmls doesn'r
    # magnitude = torch.norm(vertex_normals, dim=2)  # Debug: assert the values are equal to 1.000

    v = vb[4, :, :]
    n = vertex_normals[4, :, :]
    plot_mesh(v, f, n)


def _test_fnrmls_visually():
    from geom.mesh.vis.base import plot_mesh
    vb, f = _bring_in_test_data()
    fn, is_valid_fnb, face_areas_b = batch_fnrmls_fareas(vb, f)
    # magnitude = torch.norm(face_normals, dim=2)  # Debug: assert the values are equal to 1.000
    v = vb[4, :, :]
    n = fn[4, :, :]
    plot_mesh(v, f, n)


if __name__ == '__main__':
    _test_vnrmls_visually()
