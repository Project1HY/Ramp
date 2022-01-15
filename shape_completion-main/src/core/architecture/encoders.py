import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.base import BaseEncoder
from geom.mesh.op.gpu.dist import batch_knn
from architecture.pct import PointTransformerCls

# ----------------------------------------------------------------------------------------------------------------------
#                                                       Encoders
# ----------------------------------------------------------------------------------------------------------------------

class PointNetShapeEncoder(BaseEncoder):
    def __init__(self, code_size=1024, in_channels=3):
        super().__init__(code_size=code_size, in_channels=in_channels)

        self.graph = nn.Sequential(
            PointNetGlobalFeatures(self.code_size, self.in_channels),
            nn.Linear(self.code_size, self.code_size),
            nn.BatchNorm1d(self.code_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.graph(x)

class PCTShapeEncoder(BaseEncoder):
    def __init__(self, code_size=1024, in_channels=3):
        super().__init__(code_size=code_size, in_channels=in_channels)

        self.graph = nn.Sequential(
            PointTransformerCls(out_channels = self.code_size, input_dim = self.in_channels),
            nn.Linear(self.code_size, self.code_size),
            nn.BatchNorm1d(self.code_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.graph(x)



class PointNetGlobalFeatures(BaseEncoder):
    def __init__(self, code_size, in_channels):
        super().__init__(code_size=code_size, in_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, code_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(code_size)

    def forward(self, x):
        # Input: Batch of Point Clouds : [b x num_vertices X in_channels]
        # Output: The global feature vector : [b x code_size]
        x = x.transpose(2, 1).contiguous()  # [b x in_channels x num_vertices]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # [B x 128 x n]
        x = self.bn3(self.conv3(x))  # [B x code_size x n]
        x, _ = torch.max(x, 2)  # [B x code_size]
        return x


class DgcnnShapeEncoder(BaseEncoder):
    def __init__(self, k, device, in_channels=3, code_size=1024):
        super().__init__(code_size=code_size, in_channels=in_channels)
        self.k, self.dev = k, device

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(code_size)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, code_size, 1, bias=False), self.bn5, nn.LeakyReLU(0.2))
        self.linear1 = nn.Linear(code_size * 2, code_size, bias=False)
        self.bn6 = nn.BatchNorm1d(code_size)

    def init_weights(self):
        pass  # Use default torch init

    def graph_features(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        raise NotImplementedError('Look at me')
        if idx is None:
            # TODO - I think there is an error in batch_knn
            idx = batch_knn(x, k=self.k)  # (batch_size, num_points, k)
        idx_base = torch.arange(0, batch_size, device=self.dev).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        # batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        batch_size = x.size(0)
        x = self.graph_features(x)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.graph_features(x1)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.graph_features(x2)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.graph_features(x3)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)  # [B x 2 * code_size]
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # [B x code_size]
        return x
