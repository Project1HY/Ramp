from lightning.pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch_lightning.loggers import TestTubeLogger, WandbLogger
from lightning.pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from lightning.assets.completion_saver import CompletionSaver
from lightning.assets.emailer import TensorboardEmailer
import wandb
from util.container import to_list, first
import lightning.assets.plotter
from util.torch.nn import TensorboardSupervisor
from util.strings import banner, white_banner
from pathlib import Path
import os
import torch
import logging as log
from visualize.get_objects_hardcoded_for_sets_base import get_segmentation_manger


#  This class adds additional functionality to the Lightning Trainer, wrapping it with a similar name
class LightningTrainer:
    def __init__(self, nn, loader_complex):
        # Link self to Neural Network
        self.nn = nn
        # Link Neural Network to assets:
        self.nn.assets = self
        # Link hp to self for quick access:
        self.hp = self.nn.hp

        self.data = ParametricData(loader_complex)
        self.hp = self.data.append_data_args(self.hp)

        # Training Asset place-holders:
        self.saver, self.early_stop = None, False  # Internal Trainer Assets are marked with False, not None
        # Testing Asset place-holders:
        self.plt, self.tb_sup, self.emailer = None, None, None
        # Additional Structures:
        self.trainer, self.exp_dp = None, None
        # Flags
        self.testing_only = True

    def train(self, debug_mode=False):
        white_banner('Training Init')
        if self.trainer is None:
            self._init_training_assets()
            # log.info(f'Training on dataset: {self.data.curr_trainset_name()}')
            self.testing_only = False
        self._trainer(debug_mode).fit(self.nn, self.data.train_ldr, self.data.vald_ldrs, self.data.test_ldrs)        # train_dataloader=None, val_dataloader=None, test_dataloader=None

    def test(self):
        white_banner('Testing')
        self._trainer().test(self.nn, self.data.test_ldrs)  # Sets the trainer

    def finalize(self):
        # Called after all epochs, for cleanup
        # If needed, send the final report via email:
        if self.emailer is not None:  # Long enough train, or test only
            if self.nn.current_epoch >= self.hp.MIN_EPOCHS_TO_SEND_EMAIL_RECORD or self.testing_only:
                log.info("Sending zip with experiment specs to configured inbox")
                self.emailer.send_report(self.trainer.final_result_str)
            else:
                log.info("Model has not been trained for enough epochs - skipping attachment send")

        if self.plt is not None and self.plt.is_alive():
            self.plt.finalize()
        if self.tb_sup is not None:
            self.tb_sup.finalize()

        log.info("Cleaning up GPU memory")
        torch.cuda.empty_cache()

    def _init_training_assets(self):

        # For internal trainer:
        self.early_stop = EarlyStopping(monitor='val_loss', patience=self.hp.early_stop_patience, verbose=True,
                                        mode='min')
        if self.hp.plotter_class is not None:
            plt_class = getattr(lightning.assets.plotter, self.hp.plotter_class)
            self.plt = plt_class(faces=self.data.faces(),
                                 n_verts=self.data.num_verts())  # TODO - Cannot currently train on Scans due to this:

    def _init_trainer(self, fast_dev_run):
        if self.hp.deterministic:
            seed_everything(42, workers=True)

        if self.hp.plotter_class is not None:
            plt_class = getattr(lightning.assets.plotter, self.hp.plotter_class)
            self.plt = plt_class(faces=self.data.faces(),
                                 n_verts=self.data.num_verts())  # TODO - Cannot currently train on Scans due to this:
        # Checkpointing and Logging:
        tb_log = TestTubeLogger(save_dir=self.hp.PRIMARY_RESULTS_DIR, description=f"{self.hp.exp_name} Experiment",
                                name=self.hp.exp_name, version=self.hp.version)
        wandb_log = WandbLogger(project="my-test-project", entity="temporal_shape_recon",name=self.hp.exp_name, id=f"{self.hp.exp_name}{self.hp.version}")
        # wandb_logger.experiment.config["counts"] = self.hp.counts
        # wandb.config.update(allow_val_change=True)
        # wandb_log.experiment.config.update({'counts':self.hp.counts},allow_val_change=True)
        self.exp_dp = Path(os.path.dirname(tb_log.experiment.log_dir)).resolve()  # Extract experiment path
        checkpoint = ModelCheckpoint(filepath=self.exp_dp / 'checkpoints', save_top_k=1, verbose=True,
                                     prefix='weight', monitor='val_loss', mode='min', period=1)

        # Support for Auto-Tensorboard:
        # if self.hp.use_auto_tensorboard > 0:
        #     self.tb_sup = TensorboardSupervisor(mode=self.hp.use_auto_tensorboard)

        # Support for Completion Save:
        if self.hp.save_completions > 0 and self.data.num_test_loaders() > 0:
            # TODO - Generalize to different save methods
            seg_manager = get_segmentation_manger()
            self.saver = CompletionSaver(exp_dir=self.exp_dp, testset_names=self.data.testset_names(),
                                         extended_save=(self.hp.save_completions == 3),
                                         f=self.data.faces() if self.hp.save_completions > 1 else None,centralize_mesh_clouds = self.hp.centralize_mesh_clouds,segmentation_manager =seg_manager)

        if self.hp.email_report:
            if self.hp.GCREDS_PATH.is_file():
                self.emailer = TensorboardEmailer(exp_dp=self.exp_dp)
            else:
                log.warning("Could not find GMAIl credentials file - Skipping Emailer class init")
        # callbacks = [self.early_stop]
        callbacks = []
        # assert False,f"resume_cfg {self.hp.resume_cfg}"
        self.trainer = Trainer(fast_dev_run=fast_dev_run, num_sanity_val_steps=0, weights_summary=None,
                               gpus=self.hp.gpus, distributed_backend=self.hp.distributed_backend,
                               # val_percent_check = 0.2,
                               # accelerator="cpu",
                               early_stop_callback=self.early_stop, checkpoint_callback=checkpoint,                               
                               logger=wandb_log,
                            #    logger=tb_log,
                               min_epochs=self.hp.force_train_epoches,
                                max_epochs=self.hp.max_epochs,
                                #max_epochs = 1,
                               print_nan_grads=False,
                                   resume_cfg=self.hp.resume_cfg,
                            #    resume_from_checkpoint=self.hp.resume_cfg,
                               accumulate_grad_batches=self.hp.accumulate_grad_batches,
                            #    profiler=profiler
                                )
        # log_gpu_memory = 'min_max' or 'all'  # How to log the GPU memory
        # track_grad_norm = 2  # Track L2 norm of the gradient # Track the Gradient Norm
        # log_save_interval = 100
        # weights_summary = 'full', 'top', None
        # accumulate_grad_batches = 1
        log.info(f'Current run directory: {str(self.exp_dp)}')
        white_banner("Training Started")

    def _trainer(self, fast_dev_run=False):
        if self.trainer is None:
            self._init_trainer(fast_dev_run)
        return self.trainer


class ParametricData:
    def __init__(self, loader_complex):
        self.train_ldr = loader_complex[0]  # TODO - fix this for multi-trainsets
        if isinstance(self.train_ldr, (list, tuple)):
            self.train_ldr = self.train_ldr[0]
        self.vald_ldrs = to_list(loader_complex[1], encapsulate_none=False)
        self.test_ldrs = to_list(loader_complex[2], encapsulate_none=False)

        # Presuming all loaders stem from the SAME parametric model
        self.rep_ldr = first([self.train_ldr] + self.vald_ldrs + self.test_ldrs,
                             lambda x: x is not None)  # TODO - fix this for scans
        # torch_faces cache
        self.torch_f = None  # TODO - fix this for scans

        self.vald_set_names = self._prepare_name_list(self.vald_ldrs)
        self.test_set_names = self._prepare_name_list(self.test_ldrs)
        self.report_loaders()

    def report_loaders(self):
        white_banner('Dataset Config')
        print(f'Training Sets:')
        for i, ldr in enumerate([self.train_ldr], 1):  # TODO - edit for multi-train
            print(f'\t{i}. \033[95m{self.curr_trainset_name()}\033[0m {ldr}')
        print(f'Validation Sets:')
        for i, (name, ldr) in enumerate(zip(self.vald_set_names, self.vald_ldrs), 1):
            print(f'\t{i}. \033[95m{name}\033[0m {ldr}')
        print(f'Test Sets:')
        for i, (name, ldr) in enumerate(zip(self.test_set_names, self.test_ldrs), 1):
            print(f'\t{i}. \033[95m{name}\033[0m {ldr}')

    @staticmethod
    def _prepare_name_list(ldrs):
        set_names = []
        suffixes = []
        for ldr in ldrs:
            name = ldr.set_name()
            cnt = set_names.count(name)
            if cnt >= 1:
                suffixes.append(f"_{cnt}")
            else:
                suffixes.append('')
            set_names.append(name)
        for i in range(len(set_names)):
            set_names[i] += suffixes[i]
        return tuple(set_names)

    def testset_names(self):
        return self.test_set_names

    def valdset_names(self):
        return self.vald_set_names

    def curr_trainset_name(self):
        if self.num_train_loaders() == 0:  # TODO - fix this for multi-trainsets
            return None
        else:
            return self.train_ldr.set_name()

    def num_train_loaders(self):
        return 1 if self.train_ldr else 0  # TODO - fix this for multi-trainsets

    def num_vald_loaders(self):
        return len(self.vald_ldrs)

    def num_test_loaders(self):
        return len(self.test_ldrs)

    def index2validation_ds_name(self, set_id):
        return self.vald_set_names[set_id]

    def index2test_ds_name(self, set_id):
        return self.test_set_names[set_id]

    def faces(self):
        return self.rep_ldr.faces()  # TODO - fix this for scans

    def torch_faces(self):
        assert self.torch_f is not None  # TODO - fix this for scans
        return self.torch_f

    def num_verts(self):
        return self.rep_ldr.num_verts()  # TODO - fix this for scans

    def num_faces(self):
        return self.rep_ldr.num_faces()  # TODO - fix this for scans

    def append_data_args(self, hp):

        if self.train_ldr is not None:
            setattr(hp, f'train_ds', self.train_ldr.recon_table())  # TODO - fix this
        for i in range(self.num_vald_loaders()):
            setattr(hp, f'vald_ds_{i}', self.vald_ldrs[i].recon_table())
        for i in range(self.num_test_loaders()):
            setattr(hp, f'test_ds_{i}', self.test_ldrs[i].recon_table())

        setattr(hp, 'compute_output_normals', hp.VIS_SHOW_NORMALS or  # TODO - fix this
                hp.lambdas[1] > 0 or hp.lambdas[4] > 0 or hp.lambdas[5] > 0)

        if hp.compute_output_normals:
            assert hp.in_channels >= 6, "In channels not aligned to loss/plot config"
            # self.torch_f = torch.from_numpy(self.faces()).long().to(device=hp.dev, non_blocking=hp.NON_BLOCKING)
            # TODO - Return this when we have the generic loss

        return hp
