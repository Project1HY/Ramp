import os
import argparse
import cli_utils
import human_mesh_utils

base_dataset_dir = '/home/yiftach.ede/datasets/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplh_dir", help="path to the smplh directory", type=str, default=os.path.join(base_dataset_dir,'data/SLMPH/SLMPH/'))
    parser.add_argument("--dmpl_dir", help="path to the dmpl directory", type=str, default=os.path.join(base_dataset_dir,'data/DMPL_DIR/'))

    parser.add_argument("--num_betas", help="number of body parameters", type=int, default=16)
    parser.add_argument("--num_dmpls", help="number of DMPL parameters", type=int, default=8)
    parser.add_argument("--force_gender", help="choose whether to force the gender of the model", type=str, default='no',choices={'no','female','male','neutral'})
    parser.add_argument("-m","--model_npz", help="path to the the model *.npz file path", type=str,
            default=os.path.join(base_dataset_dir,'data/amass_dir/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz'))

    parser.add_argument("--render_pose_body", help="choose whether to render the pose body of the model (true/false)", type=cli_utils.str_to_bool, default=True)
    parser.add_argument("--render_pose_hand", help="choose whether to render the pose hands of the model (true/false)", type=cli_utils.str_to_bool, default=True)
    parser.add_argument("--render_betas", help="choose whether to render the pose hands of the model (true/false)", type=cli_utils.str_to_bool, default=True)
    parser.add_argument("--render_dmpls", help="choose whether to render the dmpls of the model (true/false)", type=cli_utils.str_to_bool, default=True)
    parser.add_argument("--render_root_orient", help="choose whether to render the rotation orientation of the model (true/false)", type=cli_utils.str_to_bool, default=True)
    parser.add_argument("--render_trans", help="choose whether to render the translation of the model (true/false)", type=cli_utils.str_to_bool, default=False)

    parser.add_argument("--check_if_sure", help="check if the user is sure about all the arguments", type=cli_utils.str_to_bool, default=False)
    parser.add_argument("--output_dir", help="path to the the output directory", type=str, default=os.path.join(base_dataset_dir,'newResTmp'))

    #sub commands

    #command video single_frame_3D
    subparsers = parser.add_subparsers(title='commands', description='commands', help='select what you want to render', dest="command",required = True)

    singleFrame3D = subparsers.add_parser('single_frame_3D', help="renver single frame in 3D")
    singleFrame3D.add_argument("-f","--frame", help="frame number to render", type=int, default=0)
    singleFrame3D.add_argument("-s","--save_output", help="choose whether to save output (true/false)", type=cli_utils.str_to_bool, default=False)
    singleFrame3D.add_argument("-o","--show_output", help="choose whether to open the rendered model on separate window(true/false)", type=cli_utils.str_to_bool, default=True)
    #command render video
    video2D = subparsers.add_parser('video_2D', help="render video file 2D")
    video2D.add_argument("--fps", help="choose fps for the video", type=int, default=60)

    allFrame3D = subparsers.add_parser('all_frames_3D', help="render all frames in 3D")

    #allFrame3D_InerActive = subparsers.add_parser('gui_mode', help="render all frames in 3D")
    args = parser.parse_args(args)
    return args

def main(args):
    comp_device = "cpu"
    print(args)
    #interactive
    body_pose = human_mesh_utils.get_body_pose(model_npz=args.model_npz,num_betas=args.num_betas,num_dmpls=args.num_dmpls,force_gender=args.force_gender,smplh_dir=args.smplh_dir, dmpl_dir=args.dmpl_dir,rendering_params_list=human_mesh_utils.get_rendering_params_list(args),comp_device=comp_device)
    f_name = cli_utils.generate_file_name(args)
    print('output file name (or pattern):\n {}'.format(f_name))
    if args.command=='single_frame_3D':
        human_mesh_utils.render_body_mesh_single_frame_3D(body_pose=body_pose,frameID=args.frame,show_output=args.show_output,save_output=args.save_output,f_name=f_name)
    number_of_frames = human_mesh_utils.get_number_of_frames(args.model_npz)
    """
    if args.command=='gui_mode':
        initial_body_model = human_mesh_utils.get_body_model(model_npz=args.model_npz,smplh_dir=args.smplh_dir, dmpl_dir=args.dmpl_dir,num_betas=args.num_betas,num_dmpls=args.num_dmpls,gender=args.force_gender,comp_device=comp_device)
        initial_body_params = human_mesh_utils.get_body_params(model_npz=args.model_npz,num_betas=args.num_betas,num_dmpls=args.num_dmpls,comp_device=comp_device)
        update_body_model_with_body_params_func = human_mesh_utils.get_update_body_model_with_body_params_func(body_model=initial_body_model,rendering_params_list=human_mesh_utils.get_rendering_params_list(args))
        show_interactive_mode(initial_body_params=initial_body_params,number_of_frames=number_of_frames,update_body_model_with_body_params_func=update_body_model_with_body_params_func)
    """
    if args.command=='video_2D':
        human_mesh_utils.render_body_mesh_video(body_pose=body_pose,number_of_frames=number_of_frames,fps=args.fps,f_name=f_name)
    if args.command=='all_frames_3D':
        human_mesh_utils.render_body_mesh_all_frames_3D(body_pose=body_pose,number_of_frames=number_of_frames,f_name_pattern=f_name)

if __name__ == '__main__':
    args = parse_args()
    if args.check_if_sure and not cli_utils.userIsSure(args):
        exit()
    cli_utils.assertVaildInputCLI(args)
    main(args)
