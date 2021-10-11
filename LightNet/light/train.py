import LightNet.light.model as model
from LightNet.light.dataset import ShapeNetDataset
import torch

import pytorch_lightning as pl
if __name__ == "__main__":
        # d = ShapeNetDataset(root = datapath, classification = True)
    root = "D:\\projectHY\\Ramp\\shapenetcore_partanno_segmentation_benchmark_v0"
    dataset = ShapeNetDataset(
        root = root,
        classification=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=int(4))

    val_dataset = ShapeNetDataset(
        root =  root,
        classification=True,
        split='val',
        data_augmentation=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=int(1))

    test_dataset = ShapeNetDataset(
        root=root,
        classification=True,
        split='test',
        data_augmentation=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=int(1))

    trainer = pl.Trainer(gpus=1,max_epochs=10)
    model = model.LightNetCls(k=16,feature_transform=False)

    trainer.fit(model, dataloader,val_dataloader)
    trainer.test(model,test_dataloader)