import argparse
import os
import time
import cv2
import copy
import sys
import contextlib
import datetime
from multiprocessing import Pool
from tqdm import tqdm
import cli_visualizer
import cli_utils
import human_mesh_utils

def parse_args()->list:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", help="path to the npz root directory", type=str, default='/home/omer/PycharmProjects/shape_completion_dataset/data/amass_dir/BMLmovi')
    parser.add_argument("--smplh_dir", help="path to the smplh directory", type=str, default='/home/omer/PycharmProjects/shape_completion_dataset/data/SLMPH/SLMPH/')
    parser.add_argument("--dmpl_dir", help="path to the dmpl directory", type=str, default='/home/omer/PycharmProjects/shape_completion_dataset/data/DMPL_DIR/')
    parser.add_argument("--out_dir", help="path to the out root directory", type=str, default='/home/omer/PycharmProjects/shape_completion/src/visualize/tmpobj')
    parser.add_argument("--log_dir", help="path to the log directory", type=str, default='/home/omer/PycharmProjects/shape_completion/src/visualize/logs')
    parser.add_argument("-n","--number_of_threads", help="num of threads", type=int, default=1)
    subparsers = parser.add_subparsers(title='commands', description='commands', help='select what you want to render', dest="command",required = True)
    all_frames_3D = subparsers.add_parser('obj', help="render all frames frame in 3D a.k.a obj file")
    all_frames_2D = subparsers.add_parser('mp4', help="render all frames frame in 2D a.k.a mp4 file")
    count_frames = subparsers.add_parser('cnt', help="count the total number of frames within the out_dir")
    all_frames_2D.add_argument("--fps", help="choose fps for the video", type=int, default=60)
    args = parser.parse_args()
    return args

def run_main(vis_args)->None:
    if not os.path.exists(vis_args.log_dir):
        os.makedirs(vis_args.log_dir)
    sys.stdout = open(os.path.join(vis_args.log_dir,'stdout_process_{}.txt'.format(os.getpid())), 'a')
    sys.stderr = open(os.path.join(vis_args.log_dir,'stderr_process_{}.txt'.format(os.getpid())), 'a')
    cli_visualizer.main(vis_args)

def run_multiprocess_loop(args:list,get_list_f)->None:
    _list = get_list_f(args)
    print("there are reminded {} videos to render".format(len(_list)))
    #append the pbar
    if args.number_of_threads>1:
        with Pool(args.number_of_threads) as p:
            p.map(run_main,_list)
    else:
        main_pbar = tqdm(total=len(_list),desc='all samples')
        for vis_args in _list:
            run_main(vis_args)
            main_pbar.update()

    print("finish yay")

def parse_vis_from_command(args:list,command:str)->list:
    vis_args = cli_visualizer.parse_args([command])
    vis_args.log_dir = get_logs_dir(args)
    vis_args.check_if_sure = False
    vis_args.smplh_dir=args.smplh_dir
    vis_args.dmpl_dir=args.dmpl_dir
    return vis_args

def get_total_number_of_frames(args:list)->None:
    # TODO can be better.. and unify with "get_arg_list_for_video_output"
    res = 0
    for root, _, files in os.walk(args.npz_dir, topdown=False):
       for f_name in files:
          if f_name.endswith(('.npz')):
              model_npz=os.path.join(root, f_name)
              if not human_mesh_utils.is_valid_npz_file(model_npz):
                  #print("skipping {} becouse it's not a valid npz model file".format(vis_args.model_npz))
                  continue
              totalframecount_expected = cli_utils.get_number_of_frames(model_npz)
              #print("file {} contain {} number of frames".format(res))
              res += totalframecount_expected
    print("total number of frames is {}".format(res))

def get_arg_list_for_obj_output(args:list)->list:
    # TODO can be better.. and unify with "get_arg_list_for_video_output"
    res = []
    vis_args=parse_vis_from_command(args,'all_frames_3D')
    for root, _, files in os.walk(args.npz_dir, topdown=False):
       for f_name in files:
          if f_name.endswith(('.npz')):
              out_dir=os.path.join(args.out_dir,os.path.relpath(root,args.npz_dir))
              vis_args.output_dir=out_dir
              vis_args.model_npz=os.path.join(root, f_name)
              if not cli_visualizer.is_valid_npz_file(vis_args.model_npz):
                  #print("skipping {} becouse it's not a valid npz model file".format(vis_args.model_npz))
                  continue
              res.append(copy.deepcopy(vis_args))
    return res

def get_arg_list_for_video_output(args:list)->list:
    res = []
    #args = parse_args()
    vis_args = parse_vis_from_command(args,'video_2D')
    for root, _, files in os.walk(args.npz_dir, topdown=False):
       for f_name in files:
          if f_name.endswith(('.npz')):
              out_dir=os.path.join(args.out_dir,os.path.relpath(root,args.npz_dir))
              #if not os.path.exists(out_dir):
              #  os.makedirs(out_dir)
              vis_args.output_dir=out_dir
              vis_args.model_npz=os.path.join(root, f_name)
              if not cli_visualizer.is_valid_npz_file(vis_args.model_npz):
                  #print("skipping {} becouse it's not a valid npz model file".format(vis_args.model_npz))
                  continue
              exp_saved_file_name = os.path.join(out_dir,"{}".format(f_name.rstrip(".npz")),"{}.mp4".format(f_name.rstrip(".npz")))
              # skip on files we did and complete before
              if os.path.exists(exp_saved_file_name):
                  try:
                      cap= cv2.VideoCapture(exp_saved_file_name)
                      totalframecount_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                      totalframecount_expected = cli_visualizer.get_number_of_frames(vis_args.model_npz)
                      #print("actual {}".format(totalframecount_actual))
                      #print("expected {}".format(totalframecount_expected))
                      if totalframecount_actual==totalframecount_expected:
                          continue
                  except:
                      print("rewite curuppted video") # I guess I don't really need this part
                      print(vis_args)
              res.append(copy.deepcopy(vis_args))
    return res

def get_logs_dir(args:list)->str:
    now = datetime.datetime.now()
    log_dir_name = "date_y{}_m{}_d{}_h{}_m{}_s{}_task_{}_n_proccesses_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second,args.command,args.number_of_threads)
    log_dir_name=os.path.join(args.log_dir,log_dir_name)
    return log_dir_name
