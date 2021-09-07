import LightNet.light.model as model
from LightNet.light.dataset import ShapeNetDataset
import torch

import pytorch_lightning as pl
        # d = ShapeNetDataset(root = datapath, classification = True)

dataset = ShapeNetDataset(
    root="enter_path",
    classification=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=int(2))

test_dataset = ShapeNetDataset(
    root="enter_path",
    classification=True,
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=int(2))

trainer = pl.Trainer()
model = model.LightNetCls(k=16,)

trainer.fit(model, dataloader)