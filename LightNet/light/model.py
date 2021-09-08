from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl


class LightSTNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv_layers = [nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()]
        self.fc_layers = [nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                          nn.Linear(256, k * k)]
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.k = k

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = self.fc_layers(x)
        iden = torch.eye(self.k, self.k).view(1, -1).float().to(device=x.device)
        x = x + iden

        x = x.view(-1, self.k, self.k)
        return x


class LightNetModel(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super().__init__()
        self.stn = LightSTNkd(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = LightSTNkd(k=64)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = x @ trans
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = x @ trans_feat
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device=trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class LightNetCls(pl.LightningModule):
    def __init__(self, k=2, feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = LightNetModel(global_feat=True, feature_transform=feature_transform)
        self.fc_layers = [nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 256), nn.Dropout(p=0.3), nn.BatchNorm1d(256), nn.ReLU(),
                          nn.Linear(256, k), nn.LogSoftmax(dim=1)]
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x.to(device=self.device)
        x, trans, trans_feat = self.feat(x)
        x = self.fc_layers(x)
        return x, trans, trans_feat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        points, target = batch
        target = target[:, 0]
        points = points.transpose(2, 1)
        pred, trans, trans_feat = self.forward(points)
        loss = F.nll_loss(pred, target)
        if trans_feat is not None:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        self.log("performance", {'loss': loss, 'accuracy': float(correct) / len(target)}, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        # self.log("correct count", correct, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        points, target = batch
        target = target[:, 0]
        points = points.transpose(2, 1)
        pred, _, _ = self.forward(points)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        self.log("performance", {'loss': loss, 'accuracy': float(correct) / len(target)}, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss
