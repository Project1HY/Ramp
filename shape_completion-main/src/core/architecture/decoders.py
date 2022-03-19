import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.base import BaseDecoder
from math import pi as PI


# import torchgeometry as tgm


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Decoders
# ----------------------------------------------------------------------------------------------------------------------
class BasicShapeDecoder(BaseDecoder):
    CCFG = [1, 1, 2, 4, 8, 16, 16]  # Enlarge this if you need more

    def __init__(self, code_size, out_channels, num_convl):
        super().__init__(code_size=code_size, out_channels=out_channels)

        if num_convl > len(self.CCFG):
            raise NotImplementedError("Please enlarge the Conv Config vector")

        self.thl = nn.Tanh()
        self.convls = []
        self.bnls = []
        for i in range(num_convl - 1):
            self.convls.append(nn.Conv1d(self.code_size // self.CCFG[i], self.code_size // self.CCFG[i + 1], 1))
            self.bnls.append(nn.BatchNorm1d(self.code_size // self.CCFG[i + 1]))
        self.convls.append(nn.Conv1d(self.code_size // self.CCFG[num_convl - 1], self.out_channels, 1))
        self.convls = nn.ModuleList(self.convls)
        self.bnls = nn.ModuleList(self.bnls)

    # noinspection PyUnresolvedReferences
    def forward(self, x):
        """
        :param x:  Point code for each point: [b x nv x pnt_code_size] pnt_code_size == in_channels + 2*shape_code
        :return: predicted coordinates for each point, after the deformation [B x nv x 3]
        """
        x = x.transpose(2, 1).contiguous()  # [b x nv x in_channels]
        for convl, bnl in zip(self.convls[:-1], self.bnls):
            x = F.relu(bnl(convl(x)))
        out = 2 * self.thl(self.convls[-1](x))  # TODO - Fix this constant - we need a global scale
        out = out.transpose(2, 1)
        return out


class LSTMDecoder(BaseDecoder):
    def __init__(self, code_size, out_channels, layer_count, bidirectional, dropout, n_verts, hidden_size, seq_len=3):
        super().__init__(code_size=code_size, out_channels=out_channels)
        self.n_verts = n_verts
        self.seq_len = seq_len
        self.thl = nn.Tanh()
        self.convolutions = nn.Sequential(*[
            nn.Conv1d(self.code_size, self.code_size // 4, 1),
            nn.BatchNorm1d(self.code_size // 4),
            nn.ReLU(),
            nn.Conv1d(self.code_size // 4, self.code_size // 16, 1),
            nn.BatchNorm1d(self.code_size // 16),
            nn.ReLU(),
            nn.Conv1d(self.code_size // 16, self.code_size // 32, 1),
            nn.BatchNorm1d(self.code_size // 32),
            nn.ReLU(),
            nn.Conv1d(self.code_size // 32, self.code_size // 256, 1),
            nn.BatchNorm1d(self.code_size // 256),
            nn.ReLU(),

        ])

        self.lstm = nn.LSTM(input_size=n_verts , hidden_size=hidden_size, dropout=dropout,
                            bidirectional=bidirectional, num_layers=layer_count,batch_first=True)
        D = 2 if bidirectional else 1
        self.reshape_matrix = nn.Sequential(nn.Linear(hidden_size * D, 1024), nn.ReLU(), nn.Linear(1024, 3 * n_verts))

    # noinspection PyUnresolvedReferences
    def forward(self, x):
        """
        :param x:  Point code for each point: [b x nv x pnt_code_size] pnt_code_size == in_channels + 2*shape_code
        :return: predicted coordinates for each point, after the deformation [B x nv x 3]
        """
        orig_shape = x.shape
        bs = x.size(0)
        window_size = x.size(1)
        x = x.reshape(bs , window_size, -1)
        assert False,f"shape is {x.shape}, bs {bs} window_size {window_size} orig shape {orig_shape}"
        x = x.transpose(2, 1).contiguous()  # [b x nv x in_channels]
        x = self.convolutions(x)
        x = x.reshape(bs, window_size, -1)
        out, _ = self.lstm(x)
        
        out = out.reshape(bs*window_size,-1)
        out = self.reshape_matrix(out).reshape(bs,window_size, self.n_verts, 3)
        # out = 2 * self.thl(out)
        return out

    # ----------------------------------------------------------------------------------------------------------------------
#                                                       Skinning Decoders
# ----------------------------------------------------------------------------------------------------------------------
# class SkinningDecoder(nn.Module):
#     def __init__(self, code_size, device, is_global=True, joints_parents=None, num_joints=52,
#                  hidden_sizes=[1024, 512, 256, 128, 64, 32, 16], s_rot=2 * PI, s_trans=2):
#         super().__init__()
#         self.code_size = code_size
#         self.num_joints = num_joints
#         self.output_size = self.num_joints * 6  # for translation vector and axis-angle vector
#         self.s_rot = s_rot
#         self.s_trans = s_trans
#         self.skin_net = self.SkinningModule(device=device, is_global=is_global, joints_parents=joints_parents)
#         self.joint_net = nn.Sequential()
#
#         layer_sizes = [self.code_size] + hidden_sizes
#         for i in range(len(layer_sizes) - 1):
#             self.joint_net.add_module(name=f'linear_{i}', module=nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
#             self.joint_net.add_module(name=f'bn_{i}', module=nn.BatchNorm1d(layer_sizes[i + 1]))
#             self.joint_net.add_module(name=f'relu_{i}', module=nn.ReLU())
#
#         self.joint_net.add_module(name='output_linear', module=nn.Linear(layer_sizes[-1], self.output_size))
#         self.joint_net.add_module(name='split_translation_rotation', module=self.SplitRotTrans(self.num_joints))
#         self.joint_net.add_module(name='output_tanh', module=self.Tanh())
#         self.joint_net.add_module(name='scale_rotation_translation',
#                                   module=self.ScaleRotTrans(s_rot=self.s_rot, s_trans=self.s_trans, device=device))
#         self.joint_net.add_module(name='rtvec_to_pose', module=self.RigidTransformation(num_joints=self.num_joints))
#
#     def forward(self, full, global_code, skinning_weights):
#         joint_transformations = self.joint_net(global_code)  # [B x J x 4 x 4]
#         deformed, global_joint_transformations = self.skin_net(full, joint_transformations, skinning_weights)
#         return deformed, global_joint_transformations
#
#     class Tanh(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#
#         def forward(self, x):
#             return torch.tanh(x)
#
#     class SplitRotTrans(torch.nn.Module):
#         # Split the last 6D channel into rotation channel and translation channel
#         def __init__(self, num_joints):
#             super().__init__()
#             self.num_joints = num_joints
#
#         def forward(self, x):
#             batch_size = x.shape[0]
#             return x.view(batch_size, self.num_joints, 2, 3)
#
#     class ScaleRotTrans(torch.nn.Module):
#         # Apply separate scale to the rotation channel and translation channel
#         def __init__(self, s_rot, s_trans, device):
#             super().__init__()
#             self.s_trans = s_trans
#             self.s_rot = s_rot
#             self.dev = device
#
#         def forward(self, x):
#             s = torch.tensor([self.s_rot, self.s_trans], device=self.dev)
#             s = s.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#             return s * x
#
#     class RigidTransformation(torch.nn.Module):
#         # convert [B x J x 2 x 3] rotation and translation vectors into [B x J x 4 x 4] homogeneous matrix
#         def __init__(self, num_joints):
#             super().__init__()
#             self.num_joints = num_joints
#
#         def forward(self, x):
#             # first, since pytroch-geometric support only [N x 6] tensors,
#             # we should convert the tensor dimensions [B x J x 2 x 3] --> [B*J, 6]
#             B = x.shape[0]
#             x = x.view(B * self.num_joints, 6)
#             x = tgm.rtvec_to_pose(x)
#             x = x.view(B, self.num_joints, 4, 4) #TODO: velidate view() is reversible
#             return x
#
#     class SkinningModule(torch.nn.Module):
#         def __init__(self, device, is_global=True, joints_parents=None):
#             super().__init__()
#             self.is_global = is_global
#             self.parents = joints_parents
#             self.dev = device
#
#             # if not is_global and not joints_parents:
#             #     raise TypeError(
#             #         f"If the skinning module uses local joint transformations, then the joint-tree must be supplied!")
#
#         def forward(self, v, T, skinning_weights):
#             # v: [B x N x 3]
#             # T: [B x J x 4 x 4]
#             # skinning_weights: [B x J x N]
#             v_homo = torch.cat((v,torch.ones(v.shape[0], v.shape[1], 1, device=self.dev)), 2) #[B x N x 4]
#
#             if not self.is_global:
#                 T = self.compute_global_transf(T)
#
#             v_new = torch.einsum('bjki,bni, bjn->bjnk', T[:, :, :3, :4], v_homo, skinning_weights) # [B x J x N x 3]
#             v_new = torch.sum(v_new, dim=1)  # [B x N x 3]
#
#             return v_new, T
#
#         def compute_global_transf(self, T):
#             global_T = [T[:, 0, :, :]]
#             for i in range(1, T.size()[1]):
#                 global_T += [torch.matmul(T[:, i, :, :], global_T[self.parents[i - 1]])]
#             return torch.stack(global_T, dim=1)
