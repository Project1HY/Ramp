import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # , CosineAnnealingLR
import architecture.loss
# from util.mesh.ops import batch_vnrmls
from collections import defaultdict
from util.torch.nn import PytorchNet
from util.func import all_variables_by_module_name
from copy import deepcopy
import sys
import wandb
import tqdm
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        self.hp = self.hparams  # Aliasing
        self._append_config_args()  # Must be done here, seeing we need hp.dev

        # Bookeeping:
        self.assets = None  # Set by Trainer
        self.loss, self.opt = None, None
        self.min_losses = defaultdict(lambda: float('inf'))

        self._build_model()
        self.type(dst_type=getattr(torch, self.hparams.UNIVERSAL_PRECISION))  # Transfer to precision
        self._init_model()

    @staticmethod  # You need to override this method
    def add_model_specific_args(parent_parser):
        return parent_parser

    def forward(self, input_dict):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def complete(self, input_dict):

        output_dict = self.forward(input_dict)
        # TODO - Implement Generic Loss and fix this function
        # if self.hparams.compute_output_normals: # TODO - Return this when we have the generic loss function
        #     vnb, vnb_is_valid = batch_vnrmls(output_dict['completion_xyz'], self.assets.data.torch_faces(),
        #                                      return_f_areas=False)
        #     output_dict['completion_vnb'] = vnb
        #     output_dict['completion_vnb_is_valid'] = vnb_is_valid

        return output_dict

    def configure_optimizers(self):
        loss_cls = getattr(architecture.loss, self.hp.loss_class)
        self.loss = loss_cls(hp=self.hp, f=self.assets.data.faces())
        self.opt = torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)

        if self.hp.plateau_patience is not None:
            sched = ReduceLROnPlateau(self.opt, mode='min', patience=self.hp.plateau_patience, verbose=True,
                                      cooldown=self.hp.DEF_LR_SCHED_COOLDOWN, eps=self.hp.DEF_MINIMAL_LR,
                                      factor=0.5)
            # Options: factor=0.1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
            return [self.opt], [sched]
        else:
            return [self.opt]

    def training_step(self, b, batch_idx):
        completion = self.complete(b)
        loss_dict = self.loss.compute(b, completion)
        loss_dict = {f'{k}_train': v for k, v in loss_dict.items()}  # make different logs for train, test, validation
        train_loss = loss_dict['total_loss_train']

        if self.assets.plt is not None and batch_idx == 0:  # On first batch
            self.assets.plt.cache(self.assets.plt.prepare_plotter_dict(b, completion))  # New tensors, without grad

        return {
            'loss': train_loss,  # Must use 'loss' instead of 'train_loss' due to old_lightning framework
            'log': loss_dict
        }

    def validation_step(self, b, batch_idx, set_id=0):
        pred = self.complete(b)
        batch_validation_mesh = pred['completion_xyz']
        if batch_idx == 0 and set_id == 0:
            batch_validation_mesh = torch.index_select(pred['completion_xyz'].cpu().detach(), 1,
                                                       torch.LongTensor([2, 0, 1]))

            batch_validation_mesh = batch_validation_mesh.numpy()[-1]
            wandb.log({"point_cloud": wandb.Object3D(batch_validation_mesh)})
            if self.assets.saver is not None:  # TODO - Generalize this
                images = self.assets.saver.get_completions_as_pil(pred, b)
                wandb.log({"completions": [wandb.Image(image) for image in images]})

        if batch_idx == 0 and set_id == 0 and self.assets.plt is not None and self.assets.plt.cache_is_filled():
            # On first batch, of first dataset, only if plotter exists and only if training step has been activated
            # before (last case does not happen if we run in dev mode).
            new_data = (self.assets.plt.uncache(), self.assets.plt.prepare_plotter_dict(b, pred))
            self.assets.plt.push(new_data=new_data, new_epoch=self.current_epoch)

        return self.loss.compute(b, pred)

    def validation_end(self, output_per_dset):

        if self.assets.data.num_vald_loaders() == 1:
            output_per_dset = [output_per_dset]  # Incase singleton case, due to PL default behaviour
        log_dict, progbar_dict, avg_val_loss = {}, {}, 0
        for i in range(len(output_per_dset)):  # Number of validation datasets
            ds_name = self.assets.data.index2validation_ds_name(i)
            for k in output_per_dset[i][0].keys():  # Compute validation loss per dataset
                log_dict[f'{k}_val_{ds_name}'] = torch.stack([x[k] for x in output_per_dset[i]]).mean()
            ds_val_loss = log_dict[f'total_loss_val_{ds_name}']
            prog_bar_name = f'val_loss_{ds_name}'
            progbar_dict[prog_bar_name] = ds_val_loss
            ds_val_loss_cpu = ds_val_loss.item()
            if ds_val_loss_cpu < self.min_losses[prog_bar_name]:
                self.min_losses[prog_bar_name] = ds_val_loss_cpu
            log_dict[f'{prog_bar_name}_min'] = self.min_losses[prog_bar_name]
            if i == 0:  # Always use the first dataset as the validation loss
                avg_val_loss = ds_val_loss
                progbar_dict['val_loss'] = avg_val_loss

        lr = self.learning_rate(self.opt)  # Also log learning rate
        progbar_dict['lr'], log_dict['lr'] = lr, lr

        # This must be kept as "val_loss" and not "avg_val_loss" due to old_lightning bug
        return {"val_loss": avg_val_loss,  # TODO - Remove double entry for val_koss
                "progress_bar": progbar_dict,
                "log": log_dict}

    def test_step(self, b, batch_idx, set_id=0):

        pred = self.complete(b)
        if self.assets.saver is not None:  # TODO - Generalize this
            self.assets.saver.save_completions_by_batch(pred, b, set_id)
        return self.loss.compute(b, pred)

    def test_end(self, outputs):
        if self.assets.data.num_test_loaders() == 1:
            outputs = [outputs]  # Incase singleton case
        if self.assets.saver is not None:  # TODO - Generalize this
            rows = []
            for completion_gif_path, completion, completion_name in tqdm.tqdm(self.assets.saver.load_completions()):
                wandb.log({"completion_video": wandb.Video(completion_gif_path, fps=60, format="gif")})
                completion = np.array(completion)
                completions_shifted = completion[1:]
                completion = completion[:-1]
                mean_velocity = np.mean(completions_shifted - completion)
                rows += [[completion_name, mean_velocity]]
            columns = ["completion subject and pose", "mean velocity"]
            wandb.log({"completion temporal metrics": wandb.Table(columns=columns, data=rows)})
            log_dict, progbar_dict = {}, {}
        avg_test_loss = 0

        for i in range(len(outputs)):  # Number of test datasets
            ds_name = self.assets.data.index2test_ds_name(i)
            for k in outputs[i][0].keys():
                log_dict[f'{k}_test_{ds_name}'] = torch.stack([x[k] for x in outputs[i]]).mean()
            ds_test_loss = log_dict[f'total_loss_test_{ds_name}']
            progbar_dict[f'test_loss_{ds_name}'] = ds_test_loss
            if i == 0:  # Always use the first dataset as the test loss
                avg_test_loss = ds_test_loss
                progbar_dict['test_loss'] = avg_test_loss

        return {"test_loss": avg_test_loss,
                "progress_bar": progbar_dict,
                "log": log_dict}

    def hyper_params(self):
        return deepcopy(self.hp)

    def _append_config_args(self):

        for k, v in all_variables_by_module_name('cfg').items():  # Only import non-class/module types
            setattr(self.hp, k, v)

        # Architecture name:
        setattr(self.hp, 'arch', self.family_name())

        if hasattr(self.hp, 'gpus'):  # This is here to allow init of lightning with only model params (no argin)

            # Device - TODO - Does this support multiple GPU ?
            dev = torch.device('cpu') if self.hp.gpus is None else torch.device('cuda', torch.cuda.current_device())
            setattr(self.hp, 'dev', dev)

            # Experiment:
            if self.hp.exp_name is None or not self.hp.exp_name:
                self.hp.exp_name = 'default_exp'

            # Epochs:
            if self.hp.max_epochs is None:
                self.hp.max_epochs = sys.maxsize

            # Correctness of config parameters:
            assert self.hp.VIS_N_MESH_SETS <= self.hp.batch_size, \
                f"Plotter needs requires batch size >= N_MESH_SETS={self.hp.VIS_N_MESH_SETS}"
