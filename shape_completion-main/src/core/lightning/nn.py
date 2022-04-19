import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # , CosineAnnealingLR
import architecture.loss
import geom.mesh.op.cpu
# from util.mesh.ops import batch_vnrmls
from collections import defaultdict
from util.torch.nn import PytorchNet
from util.func import all_variables_by_module_name
from copy import deepcopy
import numpy as np
import sys
import wandb
import tqdm
import numpy as np
import itertools
import gc
from visualize.get_objects_hardcoded_for_sets_base import get_segmentation_manger

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        self.hp = self.hparams  # Aliasing
        self._append_config_args()  # Must be done here, seeing we need hp.dev
        self.temp_data = []
        self.body_part_volume_weights = list(hp.body_part_volume_weights)

        # self.test_step_data = []

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


    def report_static_metrics(self,b,pred, stage):
        pass
        tp = b['tp']
        results = self.loss.compute_loss_end(b['gt_hi'], b['tp_hi'], b['gt'].cpu().detach().numpy(), b['gt_mask'], tp.cpu().detach().numpy(),pred['completion_xyz'].cpu().detach().numpy(), stage)
        # results = self.loss.compute_loss_end(b['gt_hi'], b['tp_hi'], b['gt'].cpu().detach().numpy(), b['gt_mask'], tp.cpu().detach().numpy(),pred['completion_xyz'].cpu().detach().numpy())

        results['gt_hi'] = b['gt_hi']
        results['tp_hi'] = b['tp_hi']

        results['gt_hi'] = list(['_'.join(str(x) for x in hi) for hi in results['gt_hi']])
        results['tp_hi'] = list(['_'.join(str(x) for x in hi) for hi in results['tp_hi']])

        # if self.assets.saver is not None:  # TODO - Generalize this
        #     images = self.assets.saver.get_completions_as_pil(pred, b)
        #     results['images']= [wandb.Image(image) for image in images]
        
        data = list(map(list, itertools.zip_longest(*results.values(),fillvalue=None)))
        keys = list(results.keys())

        wandb.log({f"static metrics {stage}": wandb.Table(columns=keys, data=data)})


    def training_step(self, b, batch_idx):
        completion = self.complete(b)
        loss_dict = self.loss.compute(b, completion)
        loss_dict = {f'{k}_train': v for k, v in loss_dict.items()}  # make different logs for train, test, validation
        train_loss = loss_dict['total_loss_train']
        if batch_idx == 0:
            self.report_static_metrics(b,completion,"train")
        
        if self.assets.plt is not None and batch_idx == 0:  # On first batch
            self.assets.plt.cache(self.assets.plt.prepare_plotter_dict(b, completion))  # New tensors, without grad

        return {
            'loss': train_loss,  # Must use 'loss' instead of 'train_loss' due to old_lightning framework
            'log': loss_dict
        }

    def training_end(self, output_per_dset):
        log_dict = {}
        best_stats = self.loss.return_best_stats('train')
        log_dict["best_mean_error_train"]=best_stats['Comp-GT Vertex L2'][2]
        log_dict["best_volume_error_train"]=best_stats['Comp-GT Volume L1'][2]
        log_dict["best_template_mean_error_train"]=best_stats['TP-GT Vertex L2'][2]

        worst_stats = self.loss.return_worst_stats('train')
        log_dict["worst_mean_error_train"]=worst_stats['Comp-GT Vertex L2'][2]
        log_dict["worst_volume_error_train"]=worst_stats['Comp-GT Volume L1'][2]
        log_dict["worst_template_mean_error_train"]=worst_stats['TP-GT Vertex L2'][2]

        mean_stats = self.loss.return_mean_stats('train')
        log_dict["mean_error_train"]=mean_stats['Comp-GT Vertex L2'][2]
        log_dict["mean_volume_error_train"]=mean_stats['Comp-GT Volume L1'][2]
        log_dict["template_mean_error_train"]=mean_stats['TP-GT Vertex L2'][2]

        wandb.run.summary['best_mean_error_train_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Vertex L2'][0])} {str(best_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['best_volume_error_train_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Volume L1'][0])} {str(best_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['best_template_mean_error_train_gt_hi_tp_hi'] = f"{str(best_stats['TP-GT Vertex L2'][0])} {str(best_stats['TP-GT Vertex L2'][1])}"
        wandb.run.summary['worst_mean_error_train_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Vertex L2'][0])} {str(worst_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['worst_volume_error_train_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Volume L1'][0])} {str(worst_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['worst_template_mean_error_train_gt_hi_tp_hi'] = f"{str(worst_stats['TP-GT Vertex L2'][0])} {str(worst_stats['TP-GT Vertex L2'][1])}"

        wandb.log(log_dict)
        return output_per_dset

    def on_validation_start(self):
        self.temp_data = []

    def validation_step(self, b, batch_idx, set_id=0):
        pred = self.complete(b)

        batch_validation_mesh = pred['completion_xyz']
        if batch_idx == 0 and set_id == 0:
            batch_validation_mesh = torch.index_select(pred['completion_xyz'].cpu().detach(), 1,
                                                       torch.LongTensor([2, 0, 1]))

            batch_validation_mesh = batch_validation_mesh.numpy()[-1]
            if self.assets.saver is not None:  # TODO - Generalize this
                images = self.assets.saver.get_completions_as_pil(pred, b)
                wandb.log({"completions": [wandb.Image(image) for image in images]})

        if batch_idx == 0 and set_id == 0 and self.assets.plt is not None and self.assets.plt.cache_is_filled():
            # On first batch, of first dataset, only if plotter exists and only if training step has been activated
            # before (last case does not happen if we run in dev mode).
            #new_data = (self.assets.plt.uncache(), self.assets.plt.prepare_plotter_dict(b, pred))
            new_data = self.assets.plt.prepare_plotter_dict(b, pred)
            if len(self.temp_data)==0:
                self.temp_data=new_data['gtrb']
            else:
                self.temp_data = np.concatenate((self.temp_data, new_data['gtrb']),axis=0)
            self.loss.compute_loss_start('validation')
            self.report_static_metrics(b,pred, "validation")

      
        return self.loss.compute(b, pred)

    def validation_end(self, output_per_dset):
        gc.collect()
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
        self.assets.plt.push(new_data=self.temp_data, new_epoch=self.current_epoch)

        lr = self.learning_rate(self.opt)  # Also log learning rate
        progbar_dict['lr'], log_dict['lr'] = lr, lr

        best_stats = self.loss.return_best_stats('validation')
        log_dict["best_mean_error_val"]=best_stats['Comp-GT Vertex L2'][2]
        log_dict["best_volume_error_val"]=best_stats['Comp-GT Volume L1'][2]
        log_dict["best_template_mean_error_val"]=best_stats['TP-GT Vertex L2'][2]

        worst_stats = self.loss.return_worst_stats('validation')
        log_dict["worst_mean_error_val"]=worst_stats['Comp-GT Vertex L2'][2]
        log_dict["worst_volume_error_val"]=worst_stats['Comp-GT Volume L1'][2]
        log_dict["worst_template_mean_error_val"]=worst_stats['TP-GT Vertex L2'][2]

        mean_stats = self.loss.return_mean_stats('validation')
        log_dict["mean_error_val"]=mean_stats['Comp-GT Vertex L2'][2]
        log_dict["mean_volume_error_val"]=mean_stats['Comp-GT Volume L1'][2]
        log_dict["template_mean_error_val"]=mean_stats['TP-GT Vertex L2'][2]

        wandb.run.summary['best_mean_error_val_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Vertex L2'][0])} {str(best_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['best_volume_error_val_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Volume L1'][0])} {str(best_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['best_template_mean_error_val_gt_hi_tp_hi'] = f"{str(best_stats['TP-GT Vertex L2'][0])} {str(best_stats['TP-GT Vertex L2'][1])}"
        wandb.run.summary['worst_mean_error_val_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Vertex L2'][0])} {str(worst_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['worst_volume_error_val_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Volume L1'][0])} {str(worst_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['worst_template_mean_error_val_gt_hi_tp_hi'] = f"{str(worst_stats['TP-GT Vertex L2'][0])} {str(worst_stats['TP-GT Vertex L2'][1])}"

        self.loss.compute_loss_start('validation')
        # This must be kept as "val_loss" and not "avg_val_loss" due to old_lightning bug
        return {"val_loss": avg_val_loss,  # TODO - Remove double entry for val_koss
                "progress_bar": progbar_dict,
                "log": log_dict}

    def test_step(self, b, batch_idx, set_id=0):
        pred = self.complete(b)

        if self.assets.saver is not None:  # TODO - Generalize this
            self.assets.saver.save_completions_by_batch(pred, b, set_id)
        
        if self.hp.visualization_run:
            self.organs_to_lambdas = {
                "RightArm": self.body_part_volume_weights[0],
                "LeftArm": self.body_part_volume_weights[1],
                "Head": self.body_part_volume_weights[2],
                "RightLeg": self.body_part_volume_weights[3],
                "LeftLeg": self.body_part_volume_weights[4],
                "Torso": self.body_part_volume_weights[5]
            }
            
            self.organs_to_lambdas = {k:v  for (k,v) in self.organs_to_lambdas.items() if v>0}
            segmentation_manger=get_segmentation_manger(organs=list(self.organs_to_lambdas.keys()))
            segmented_meshes_watertight = segmentation_manger.get_meshes_of_segments(pred,watertight_mesh=True,center=True)
            return
        tp = b['tp']
        results = self.loss.compute_loss_end(b['gt_hi'], b['tp_hi'], b['gt'].cpu().detach().numpy(), b['gt_mask'], tp.cpu().detach().numpy(),pred['completion_xyz'].cpu().detach().numpy(), 'test')

        results['gt_hi'] = b['gt_hi']
        results['tp_hi'] = b['tp_hi']

        results['gt_hi'] = list(['_'.join(str(x) for x in hi) for hi in results['gt_hi']])
        results['tp_hi'] = list(['_'.join(str(x) for x in hi) for hi in results['tp_hi']])

        if self.assets.saver is not None:  # TODO - Generalize this
            images = self.assets.saver.get_completions_as_pil(pred, b)
            results['images']= [wandb.Image(image) for image in images]

        data = list(map(list, itertools.zip_longest(*results.values(),fillvalue=None)))
        keys = list(results.keys())


        wandb.log({"static metrics": wandb.Table(columns=keys, data=data)})
        #new_test_data = self.assets.plt.prepare_plotter_dict(b, pred)
        # if len(self.test_step_data)==0:
        #     self.test_step_data=[results]
        # else:
        #     self.test_step_data += [results]
        
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
        table_dict = {}
        for key in log_dict:
            table_dict[key] = log_dict[key].item()    
        wandb.log({"completion test metrics":wandb.Table(columns=list(table_dict.keys()),data=[list(table_dict.values())])})
        
        best_stats = self.loss.return_best_stats('test')
        log_dict["best_mean_error_test"]=best_stats['Comp-GT Vertex L2'][2]
        log_dict["best_volume_error_test"]=best_stats['Comp-GT Volume L1'][2]
        log_dict["best_template_mean_error_test"]=best_stats['TP-GT Vertex L2'][2]
        
        worst_stats = self.loss.return_worst_stats('test')
        log_dict["worst_mean_error_test"]=worst_stats['Comp-GT Vertex L2'][2]
        log_dict["worst_volume_error_test"]=worst_stats['Comp-GT Volume L1'][2]
        log_dict["worst_template_mean_error_test"]=worst_stats['TP-GT Vertex L2'][2]

        mean_stats = self.loss.return_mean_stats('test')
        log_dict["mean_error_test"]=mean_stats['Comp-GT Vertex L2'][2]
        log_dict["mean_volume_error_test"]=mean_stats['Comp-GT Volume L1'][2]
        log_dict["template_mean_error_test"]=mean_stats['TP-GT Vertex L2'][2]

        wandb.run.summary['best_mean_error_test_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Vertex L2'][0])} {str(best_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['best_volume_error_test_gt_hi_tp_hi'] = f"{str(best_stats['Comp-GT Volume L1'][0])} {str(best_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['best_template_mean_error_test_gt_hi_tp_hi'] = f"{str(best_stats['TP-GT Vertex L2'][0])} {str(best_stats['TP-GT Vertex L2'][1])}"
        wandb.run.summary['worst_mean_error_test_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Vertex L2'][0])} {str(worst_stats['Comp-GT Vertex L2'][1])}"
        wandb.run.summary['worst_volume_error_test_gt_hi_tp_hi'] = f"{str(worst_stats['Comp-GT Volume L1'][0])} {str(worst_stats['Comp-GT Volume L1'][1])}"
        wandb.run.summary['worst_template_mean_error_test_gt_hi_tp_hi'] = f"{str(worst_stats['TP-GT Vertex L2'][0])} {str(worst_stats['TP-GT Vertex L2'][1])}"

        display_vals = [str(log_dict["best_mean_error_test"]), str(log_dict["best_volume_error_test"]), str(log_dict["best_template_mean_error_test"]),
        str(log_dict["worst_mean_error_test"]), str(log_dict["worst_volume_error_test"]), str(log_dict["worst_template_mean_error_test"]), str(log_dict["mean_error_test"]),
        str(log_dict["mean_volume_error_test"]), str(log_dict["template_mean_error_test"])]

        display_keys = ["best_mean_error_test", "best_volume_error_test", "best_template_mean_error_test", "worst_mean_error_test",
        "worst_volume_error_test", "worst_template_mean_error_test", "mean_error_test", "mean_volume_error_test", "template_mean_error_test"]
        result_dict = {k:v for k,v in zip(display_keys,display_vals)}
        wandb.log({"total test results": wandb.Table(columns=list(result_dict.keys()), data=[list(result_dict.values())])})

        #self.loss.compute_loss_start()
        # This must be kept as "val_loss" and not "avg_val_loss" due to old_lightning bug
        wandb.log(log_dict)
        return {"test_loss": avg_test_loss,  # TODO - Remove double entry for val_koss
                "progress_bar": progbar_dict,
                "log": log_dict}
        
        
        # best_metrics = self.loss.return_best_stats() 
        # best_metrics = ','.join(best_metrics)
        # vals = ["mean", "volume", "temp"]
        # #assert False, f"best metrics: {best_metrics}"
        # wandb.log({"best metrics test": wandb.Table(columns=vals, data=[list(best_metrics)])})
        # worst_metrics = self.loss.return_worst_stats() 
        # vals2 = ["mean", "volume", "temp"]
        # wandb.log({"worst metrics test": wandb.Table(columns=vals2, data=[list(worst_metrics)])})
        # mean_metrics = self.loss.return_mean_stats() 
        # vals3 = ["mean", "volume", "temp"]
        # wandb.log({"mean metrics test": wandb.Table(columns=vals3, data=[list(mean_metrics)])})
        # return {"test_loss": avg_test_loss,
        #                 "progress_bar": progbar_dict,
        #                 "log": log_dict}
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
