import LightNet.light.model as model
from LightNet.light.dataset import ShapeNetDataset
import torch

import pytorch_lightning as pl
        # d = ShapeNetDataset(root = datapath, classification = True)

dataset = ShapeNetDataset(
    root="D:\\Project1MStart\\shapenetcore_partanno_segmentation_benchmark_v0",
    classification=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=int(0))

test_dataset = ShapeNetDataset(
    root="D:\\Project1MStart\\shapenetcore_partanno_segmentation_benchmark_v0",
    classification=True,
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=int(0))

trainer = pl.Trainer(gpus=1,max_epochs=10)
model = model.LightNetCls(k=16,feature_transform=True)

trainer.fit(model, dataloader)
trainer.test(model,testdataloader)