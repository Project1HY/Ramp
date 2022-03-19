import os
import argparse
import human_mesh_utils
import cli_utils
import gui_utils
from gui_mesh_editor_standalone import gui_loop

base_dataset_dir = '/home/yiftach.ede/datasets/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplh_dir", help="path to the smplh directory", type=str, default=os.path.join(base_dataset_dir,'data/SLMPH/SLMPH/'))
    parser.add_argument("--dmpl_dir", help="path to the dmpl directory", type=str, default=os.path.join(base_dataset_dir,'data/DMPL_DIR/'))
    parser.add_argument("--num_betas", help="number of body parameters", type=int, default=16)
    parser.add_argument("--num_dmpls", help="number of DMPL parameters", type=int, default=8)
    parser.add_argument("-m","--initial_model_npz", help="path to the the initial model *.npz file path", type=str,
            default=os.path.join(base_dataset_dir,'data/amass_dir/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz'))
    parser.add_argument("--check_if_sure", help="check if the user is sure about all the arguments", type=cli_utils.str_to_bool, default=False)
    args = parser.parse_args(args)
    return args

def main(args):
    comp_device = "cpu" #TODO this time it's CPU and hardcoded,maybe in the futhure one can change it to choose gpu or cpu acurdingly
    amass_actor=gui_utils.AmassActor(initial_model_npz=args.initial_model_npz,smplh_dir=args.smplh_dir, dmpl_dir=args.dmpl_dir,num_betas=args.num_betas,num_dmpls=args.num_dmpls,comp_device=comp_device)
    gui_loop(amass_actor=amass_actor)

if __name__ == '__main__':
    args = parse_args()
    if args.check_if_sure and not cli_utils.userIsSure(args):
        exit()
    gui_utils.assertVaildInputGUI(args)
    main(args)
