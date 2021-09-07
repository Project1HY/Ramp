from lightning.trainer import LightningTrainer
from util.torch.nn import set_determinsitic_run
from util.torch.data import none_or_int, none_or_str
from util.strings import white_banner, set_logging_to_stdout
from data.loaders import *
from architecture.f2p import *
from architecture.inverse_skining import *

set_logging_to_stdout()
set_determinsitic_run()  # Set a universal random seed


# ----------------------------------------------------------------------------------------------------------------------
#                                               Main Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parser(exp_name='Skinning_Single_Subject_Overfit_Exp', version=None, is_global=True):
    p = HyperOptArgumentParser(strategy='random_search')

    # Check-pointing
    p.add_argument('--exp_name', type=str, default=exp_name,
                   # TODO - Don't forget to change me!
                   help='The experiment name. Leave empty for default')
    p.add_argument('--version', type=none_or_int, default=version,
                   help='Weights will be saved at weight_dir=exp_name/version_{version}. '
                        'Use None to automatically choose an unused version')
    p.add_argument('--resume_cfg', nargs=2, type=bool, default=(False, True),
                   help='Only works if version != None and and weight_dir exists. '
                        '1st Bool: Whether to attempt restore of early stopping callback. '
                        '2nd Bool: Whether to attempt restore learning rate scheduler')
    p.add_argument('--save_completions', type=int, choices=[0, 1, 2, 3], default=3,
                   help='Use 0 for no save. Use 1 for vertex only save in obj file. Use 2 for a full mesh save (v&f). '
                        'Use 3 for gt,tp,gt_part,tp_part save as well.')

    # Dataset Config:
    # NOTE: A well known ML rule: double the learning rate if you double the batch size.
    p.add_argument('--batch_size', type=int, default=10, help='SGD batch size')
    # TODO: This parameter applies for P & Q, however it can be overridden is some architecture
    p.add_argument('--in_channels', choices=[3, 6, 12], default=6,
                   help='Number of input channels')

    # Train Config:
    p.add_argument('--force_train_epoches', type=int, default=1,
                   help="Force train for this amount. Usually we'd early stop using the callback. Use 1 to disable")
    p.add_argument('--max_epochs', type=int, default=None,  # Must be over 1
                   help='Maximum epochs to train for. Use None for close to infinite epochs')
    p.add_argument('--lr', type=float, default=0.003, help='The learning step to use')
    p.add_argument('--counts', nargs=3, type=none_or_int, default=(10000, 1000, 1000),
                   help='The default train,validation and test counts. Recommended [8000-20000, 500-1000, 500-1000]. '
                        'Use None to take all examples in the partition - '
                        'for big datasets, this could blow up the epoch')

    # Optimizer
    p.add_argument("--weight_decay", type=float, default=0, help="Adam's weight decay - usually use 1e-4")
    p.add_argument("--plateau_patience", type=none_or_int, default=30,
                   help="Number of epoches to wait on learning plateau before reducing step size. Use None to shut off")
    p.add_argument("--early_stop_patience", type=int, default=80,  # TODO - Remember to setup resume_cfg correctly
                   help="Number of epoches to wait on learning plateau before stopping train")
    p.add_argument('--accumulate_grad_batches', type=int, default=5,
                   help='Number of batches to accumulate gradients. Use 1 for no accumulation')
    # Without early stop callback, we'll train for cfg.MAX_EPOCHS

    # L2 Losses: Use 0 to ignore, >0 to lightning
    p.add_argument('--lambdas', nargs=7, type=float, default=(1, 0.01, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap,FaceAreas,Volume]'
                        'loss multiplication modifiers')
    p.add_argument('--mask_penalties', nargs=7, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap,FaceAreas,Volume]'
                        'increased weight on mask vertices. Use val <= 1 to disable')
    p.add_argument('--dist_v_penalties', nargs=7, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap, FaceAreas, Volume]'
                        'increased weight on distant vertices. Use val <= 1 to disable')
    p.add_argument('--loss_class', type=str, choices=['BasicLoss', 'SkepticLoss'], default='BasicLoss',
                   help='The loss class')  # TODO - generalize this

    # Computation
    p.add_argument('--gpus', type=none_or_int, default=-1, help='Use -1 to use all available. Use None to run on CPU')
    p.add_argument('--distributed_backend', type=str, default='dp', help='supports three options dp, ddp, ddp2')
    # TODO - ddp2,ddp Untested. Multiple GPUS - not tested

    # Visualization
    p.add_argument('--use_auto_tensorboard', type=bool, default=3,
                   help='Mode: 0 - Does nothing. 1 - Opens up only server. 2 - Opens up only chrome. 3- Opens up both '
                        'chrome and server')
    p.add_argument('--plotter_class', type=none_or_str, choices=[None, 'SkinningPlotter', 'DenoisingPlotter'],
                   default='SkinningPlotter',
                   help='The plotter class or None for no plot')  # TODO - generalize this

    # Completion Report
    p.add_argument('--email_report', type=bool, default=True,
                   help='Email basic tensorboard dir if True')

    #global/local joint transformation
    p.add_argument('--is_global', type=bool, default=is_global,
                   help='global local joint transformation if true')

    return [p]


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Mains
# ----------------------------------------------------------------------------------------------------------------------
def train_main():
    white_banner('Network Init')
    nn = F2PEncoderDecoderSkinning(parser())
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MixamoSkinnedGaon9Proj', need_part=True)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train(debug_mode=True)
    trainer.test()
    trainer.finalize()

# f2f on miniMixamo with one subject (000)
def train_f2f_MiniMixamo_000_main():
    white_banner('F2F Network Init on Mini Mixamo with 000 subject')
    nn = F2FEncoderDecoderSkinning(parser())
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MiniMixamoSkinnedSingleSubject', need_part=False)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


# f2f on miniMixamo with all subjects
def train_f2f_MiniMixamo_main():
    white_banner('F2F Network Init on Mini Mixamo')
    nn = F2FEncoderDecoderSkinning(parser())
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MiniMixamoSkinned', need_part=False)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


def train_f2p_MiniMixamo_000_main():
    white_banner('F2P Network Init on Mini Mixamo with 000 subject')
    nn = F2PEncoderDecoderSkinning(parser(exp_name='F2P Skinning Network EXP on Mini Mixamo with 000 subject'))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MiniMixamoSkinnedSingleSubject', need_part=True)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()

# f2f on miniMixamo with all subjects
def train_f2p_MiniMixamo_main():
    white_banner('F2P Network Init on Mini Mixamo')
    nn = F2PEncoderDecoderSkinning(parser(exp_name='F2P Skinning Network EXP on Mini Mixamo'))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MiniMixamoSkinned', need_part=True)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()

###
# f2f on miniMixamo with one subject (000)
def train_f2f_Mixamo_000_main():
    white_banner('F2F Network Init on Mixamo with 000 subject')
    nn = F2FEncoderDecoderSkinning(parser(exp_name='F2F Skinning Network EXP on Mixamo with 000 subject'))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MixamoSkinnedSingleSubject', need_part=False)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


# f2f on miniMixamo with all subjects
def train_f2f_Mixamo_main():
    white_banner('F2F Network Init on Mixamo')
    nn = F2FEncoderDecoderSkinning(parser(exp_name='F2F Skinning Network EXP on Mixamo with local joints trans', is_global=False))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MixamoSkinned', need_part=False)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


def train_f2p_Mixamo_000_main():
    white_banner('F2P Network Init on Mixamo with 000 subject')
    nn = F2PEncoderDecoderSkinning(parser(exp_name='F2P Skinning Network EXP on Mixamo with 000 subject'))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MixamoSkinnedSingleSubject', need_part=True)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()

# f2f on miniMixamo with all subjects
def train_f2p_Mixamo_main(exp_name, is_gloabl):
    white_banner('F2P Network Init on Mixamo')
    nn = F2PEncoderDecoderSkinning(parser(exp_name=exp_name, is_global=is_gloabl))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MixamoSkinned', need_part=True)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()

#need_part false for f2f
def Mixmamo_trainer(encoder_decoder, exp_name, is_gloabl, need_part=True):
    nn = encoder_decoder(parser(exp_name=exp_name, is_global=is_gloabl))
    nn.identify_system()

    # Bring in data:
    ldrs = skinnng_loaders(nn.hp, ds_name='MiniMixamoSkinnedSingleSubject', need_part=need_part)

    # Commence Training
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    Mixmamo_trainer(F2FEncoderDecoderSkinning, exp_name='F2F Skinning Network EXP on single frame ,local joint trans from rest swapped recursion', is_gloabl=False, need_part=False)
    # Mixmamo_trainer(F2PEncoderDecoderSkinning, exp_name='F2P Skinning Network EXP on Mixamo with local joint trans with 50 BA', is_gloabl=False)
    # Mixmamo_trainer(F2FEncoderDecoderSkinning, exp_name='F2F Skinning Network EXP on Mixamo with global joint trans with 50 BA', is_gloabl=True, need_part=False)
    # Mixmamo_trainer(F2FEncoderDecoderSkinning, exp_name='F2F Skinning Network EXP on Mixamo with local joint trans with 50 BA', is_gloabl=False, need_part=False)

