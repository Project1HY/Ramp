import torch
from torch import nn as nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Abstract
# ----------------------------------------------------------------------------------------------------------------------

class BaseShapeModule(nn.Module):
    def init_weights(self):
        """
        Default init is Gaussian. Override to change behaviour
        """
        self._default_gaussian_weight_init()

    # noinspection PyUnresolvedReferences
    def _default_gaussian_weight_init(self, conv_mu=0.0, conv_sigma=0.02, bn_gamma_mu=1.0,
                                      bn_gamma_sigma=0.02, bn_betta_bias=0.0):
        # TODO - remove parameter hardcoding
        for m in self.modules():  # TODO - Should we init Linear layers here?
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=conv_mu, std=conv_sigma)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=bn_gamma_mu, std=bn_gamma_sigma)  # weight=gamma, bias=betta
                nn.init.constant_(m.bias, bn_betta_bias)


class BaseEncoder(BaseShapeModule):
    def __init__(self, code_size, in_channels):
        super().__init__()
        self.code_size, self.in_channels = code_size, in_channels

    def forward(self, x):
        """
        :param x: Batch of Point Clouds : [b x num_vertices X in_channels]
        :return: The latent encoding : [b x code_size]
        """
        raise NotImplementedError


class BaseDecoder(BaseShapeModule):
    def __init__(self, code_size, out_channels):
        super().__init__()
        self.code_size, self.out_channels = code_size, out_channels


# ----------------------------------------------------------------------------------------------------------------------
#                                               Regressors
# ----------------------------------------------------------------------------------------------------------------------

class Regressor(BaseShapeModule):
    # TODO: support external control on internal architecture
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.lin1 = nn.Linear(2 * self.code_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.code_size)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.code_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        return x


class Template:
    # TODO - Clean this up
    def __init__(self, in_channels, dev):
        from cfg import UNIVERSAL_PRECISION, SMPL_TEMPLATE_PATH
        from geom.mesh.io.base import read_ply
        from geom.mesh.op.gpu.base import batch_vnrmls

        vertices, faces, colors = read_ply(SMPL_TEMPLATE_PATH)
        self.vertices = torch.tensor(vertices, dtype=getattr(torch, UNIVERSAL_PRECISION)).unsqueeze(0)
        faces = torch.from_numpy(faces).long()
        self.in_channels = in_channels
        if self.in_channels == 6:
            self.normals, _ = batch_vnrmls(self.vertices, faces, return_f_areas=False)  # done on cpu (default)
            self.normals = self.normals.to(device=dev)  # transformed to requested device
        self.vertices = self.vertices.to(device=dev)  # transformed to requested device

        # self.colors = colors

    def get_template(self):
        if self.in_channels == 3:
            return self.vertices
        elif self.in_channels == 6:
            return torch.cat((self.vertices, self.normals), 2).contiguous()
        else:
            raise AssertionError
