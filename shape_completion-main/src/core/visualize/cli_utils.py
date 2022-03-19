import os
import pickle
from pathlib import Path
import human_mesh_utils

def str_to_bool(value) :
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def userIsSure(opts) -> bool:
        def yes_or_no_ans(question):
            while True:
                reply = str(input(question + ' (y/n): ')).lower().strip()
                if reply[0] == 'y':
                    return True
                if reply[0] == 'n':
                    return False

        print("____________________________________")
        print("parsed params:")
        print(str(opts).split("(")[1].replace(",", "\n").split(")")[0])
        print("____________________________________")
        return yes_or_no_ans("are you sure you want to continue?")

def generate_file_name(args)->str:
    # I can make this code cleaner
    def get_file_name_without_extension(full_filename:str)->str:
        return os.path.splitext(os.path.basename(full_filename))[0]
    f_name = get_file_name_without_extension(args.model_npz)
    d_name = os.path.join(args.output_dir,f_name)
    if not os.path.exists(d_name):
        os.makedirs(d_name)
    f_name=os.path.join(d_name,f_name)
    if args.command=='all_frames_3D':
        return '{}_frame_X.obj'.format(f_name)
    if args.command=='single_frame_3D':
        return '{}_frame_{}.obj'.format(f_name,args.frame)
    if args.command=='video_2D':
        return '{}.mp4'.format(f_name)

def assertVaildInputCLI(args)->None:
    human_mesh_utils.assertVaildEnviroment()
    errStr = ''
    # make sure we have something to render
    if not human_mesh_utils.is_valid_render_list(args):
        errStr+='nothing to render.\n'
    # make sure skmplh and dmpl paths are ok
    valid_prior_dirs,err_of_render_list = human_mesh_utils.is_valid_skmpl_and_dmpl_dirs(args)#TODO make it look better
    if not valid_prior_dirs:
        errStr+=err_of_render_list
    if not os.path.exists(args.model_npz):
                errStr+='model file {} not exists'.format(args.model_npz)
    if errStr == '' and not human_mesh_utils.is_valid_npz_file(args.model_npz):
                errStr+='model file {} exists, but not valid'.format(args.model_npz)
    if not os.path.exists(args.output_dir):
                errStr+='output_dir {} not exists'.format(args.output_dir)
    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)
