import argparse
import os
import torch
from sys import platform
import cli_utils
import sampling
import human_mesh_utils

#defult values
def get_defult_values():
    res=human_mesh_utils.get_defult_dir_values()
    num_of_samples=50
    res['num_of_samples']=num_of_samples
    res['overwrite_cache']=False

    return res

def create_objects_add_arguments(create_obj_subparser:argparse.ArgumentParser,default_out_dir:str,default_out_filename:str,add_amass_dir_arguments:bool=False)->None:
    create_obj_subparser.add_argument("-d","--output_object_dir", help="output object file dir", type=str, default=default_out_dir)
    create_obj_subparser.add_argument("-f","--output_object_filename", help="output object file name", type=str, default=default_out_filename)
    if add_amass_dir_arguments:
        default_value=os.path.join(get_defult_values()['cached_amass_object_dir'],'amass_obj.pkl') # not so buitiful I know
        create_obj_subparser.add_argument("--input_amass_obj_full_filepath", help="input_amass_full_filepath", type=str, default=default_value)
        create_obj_subparser.add_argument("--add_fps_full_sampling", help="adding fps full sampling to obj", type=cli_utils.str_to_bool, default=False)

def sample_add_arguments(sample_subparser:argparse.ArgumentParser,default_num_of_samples:int,\
        default_output_sample_dir:str,default_dataset_object_dir:str,default_dataset_object_filename:str)->None:
    sample_subparser.add_argument("-o","--output_sample_dir", help="output sampling files root dir", type=str, default=default_output_sample_dir)
    sample_subparser.add_argument("-n","--num_of_samples", help="""number of samples.
            (-1 to sample all frames is dataset-permutation)""", type=int, default=default_num_of_samples)
    sample_subparser.add_argument("-d","--dataset_object_dir", help="the dir of the required dataset to sample from", type=str, default=default_dataset_object_dir)
    sample_subparser.add_argument("-f","--dataset_object_filename", help="the filename of the required dataset to sample from"\
            , type=str, default=default_dataset_object_filename,required=True)
    sample_subparser.add_argument("-p","--save_output_npz_file_for_sampling", help="save a pkl file for the sampling", type=cli_utils.str_to_bool, default=True)
    sample_subparser.add_argument("-k","--save_output_sampling_histogram_figure", help="save a histogram file for the sampling", type=cli_utils.str_to_bool, default=True)
    sample_subparser.add_argument("-s","--seed", help="choose the seed for the sampling", type=int, default=1)
    sample_subparser.add_argument("-r","--force_random_seed", help="force random seed for the sampling.\
            will take over the seed argument in case it's true", type=cli_utils.str_to_bool, default=False)
    sample_subparser.add_argument("-v","--show_histogram_figure", help="show histogram figure", type=cli_utils.str_to_bool, default=False)
    sample_subparser.add_argument("-n_bins","--n_bins_for_histogram_figure", help="show histogram figure", type=int, default=100)

def parse_args(args=None):
    default_values=get_defult_values()

    parser = argparse.ArgumentParser(add_help=True,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--overwrite_cache", help="over-write saved files?", type=cli_utils.str_to_bool, default=default_values['overwrite_cache'])
    parser.add_argument("--check_if_sure", help="check if the user is sure about all the arguments", type=cli_utils.str_to_bool, default=True)

    subparsers = parser.add_subparsers(title='commands', description='commands', help='select what you want to do', dest="command",required = True)

    #creating objects

    #create_amass_obj
    create_amass_obj = subparsers.add_parser('create_amass_obj', help="create object (dataset for sample or amass object)")
    create_amass_obj.add_argument("-a","--amass_dir", help="path to the input amass directory", type=str, default=default_values['amass_dir'])
    create_objects_add_arguments(create_obj_subparser=create_amass_obj,\
            default_out_dir=default_values['cached_amass_object_dir'],default_out_filename='amass_obj.pkl',add_amass_dir_arguments=False)

    #create_dataset_obj
    create_dataset_obj = subparsers.add_parser('create_dataset_obj', help="create dataset object")
    create_objects_add_arguments(create_obj_subparser=create_dataset_obj,\
            default_out_dir=default_values['cached_dataset_objects_dir'],default_out_filename='',add_amass_dir_arguments=True)
    create_dataset_obj.add_argument("-n","--mini_dataset_name", help="the name of the mini dataset to create from AMASS", type=str, default='',required=True)

    #create_mini amass dataset_obj
    create_mini_amass_dataset_obj = subparsers.add_parser('create_mini_amass_dataset_obj', help="create mini amass dataset object")
    create_objects_add_arguments(create_obj_subparser=create_mini_amass_dataset_obj,\
            default_out_dir=default_values['cached_dataset_objects_dir'],
            default_out_filename='mini_amass_dataset_obj.pkl',add_amass_dir_arguments=True)

    #sampling objects

    #sample_dataset_objects
    sample_random = subparsers.add_parser('sample_random', help="sample randomly from a given dataset object")
    sample_add_arguments(sample_subparser=sample_random,default_num_of_samples=default_values['num_of_samples'],
        default_output_sample_dir=default_values['sampling_dir'],default_dataset_object_dir=default_values['cached_dataset_objects_dir'],
        default_dataset_object_filename='')

    furtherst_sampling= subparsers.add_parser('sample_furtherst', help="sample with furtherst point sampling from a given dataset object")
    sample_add_arguments(sample_subparser=furtherst_sampling,default_num_of_samples=default_values['num_of_samples'],
        default_output_sample_dir=default_values['sampling_dir'],default_dataset_object_dir=default_values['cached_dataset_objects_dir'],
        default_dataset_object_filename='')
    furtherst_sampling.add_argument('--path_to_theta_vector',help="if activated use requested theta vector when sampling", type=str,default='')
    furtherst_sampling.add_argument('--comp_device',help='''use spesific comptional device (cpu,cuda)
            .if default is remaning will choose cuda only if it is is_available''',type=str,default='default')
    furtherst_sampling.add_argument("--iterations_per_sample",help="""number of iterations for each sample.
            make this param bigger in case of memory errors""", type=int,default=1)
    furtherst_sampling.add_argument('--only_allow_samples_that_diveded_by',help="we sample only from every i'th sample (1 is for all)", type=int,default=1)
    furtherst_sampling.add_argument('--farthest_sampling_method',
            help="""method to use for executing farthest point sampling
            can be ('vectors_loop','distance_matrix' )""", type=str,default='vectors_loop')

    """
    dataset_fps_full = subparsers.add_parser('create_dataset_with_full_fps', help="save furtherst point sampling into the dataset objects.will overwriteCache by defult")
    create_objects_add_arguments(create_obj_subparser=dataset_fps_full,\
            default_out_dir=default_values['cached_dataset_objects_dir'],default_out_filename='',add_amass_dir_arguments=False)
    """

    args=parser.parse_args()
    return args

def assertVaildInput(args):
    errStr = ''
    path_arguments=['amass_dir','output_object_dir','output_sample_dir','dataset_object_dir','input_amass_obj_full_filepath']
    amass_datasets_lists=[ 'ACCAD','BMLmovi','BMLrub','CMU','DFaust67',\
            'EKUT','EyesJapanDataset','HumanEva','KIT','MPIHDM05','MPILimits',\
            'MPImosh','SFU','SSMsynced','TCDhandMocap','TotalCapture','Transitionsmocap']
    valid_comp_devices=['cpu','cuda','default']
    valid_farthest_sampling_methods=['vectors_loop','distance_matrix']
    for arg_name,arg_value in args.__dict__.items():
        if arg_name in path_arguments and not os.path.exists(arg_value):
            errStr+='worng argument {}:directory {} not exists.\n'.format(arg_name,arg_value)
        elif arg_name == 'mini_dataset_name' and arg_value not in amass_datasets_lists:
            errStr+='worng argument {}:dataset name {} not exists in amass.\n - please use one of the following {}.\n'.format(arg_name,arg_value,amass_datasets_lists)
        elif arg_name == 'comp_device':
            if arg_value not in valid_comp_devices:
                errStr+='worng argument {}:comp device {} is not valid.\n'.format(arg_name,arg_value)
            else:
                if not torch.cuda.is_available() and arg_value=='cuda':
                    errStr+='worng argument {}:cuda not available.\n'.format(arg_name)
        elif arg_name=='command' and arg_value in ['create_dataset_obj','create_mini_amass_dataset_obj']:
            if not os.path.exists(args.input_amass_obj_full_filepath):
                    errStr+='worng argument {}:amass object {} do not found.\n'.format(arg_name,arg_value)
        elif arg_name=='command' and arg_value=='farthest_sampling' and args.farthest_sampling_method not in valid_farthest_sampling_methods:
             errStr+='''worng argument {}:farthest_sampling_method {} not found.
                        please use one of the following methods\n'''.format(
                                arg_name,arg_value,valid_farthest_sampling_methods)
    if errStr != '':
        raise Exception("unvalid input:\n"+errStr)

def main(args):
    print('executing {}'.format(args.command))
    if args.command=='create_amass_obj':
        out_file=os.path.join(args.output_object_dir,args.output_object_filename)
        sampling.save_amass_object(input_amass_dir=args.amass_dir,output_amass_obj_full_filepath=out_file,overwriteCache=args.overwrite_cache)
    elif args.command=='create_dataset_obj':
        if args.output_object_filename=='':
            args.output_object_filename='{}_dataset_obj.pkl'.format(args.mini_dataset_name)
        out_file=os.path.join(args.output_object_dir,args.output_object_filename)
        sampling.save_internal_dataset_in_amass_obj(input_amass_obj_full_filepath=args.input_amass_obj_full_filepath,\
                req_dataset_name=args.mini_dataset_name,
                output_dataset_full_filepath=out_file,overwriteCache=args.overwrite_cache,
                add_fps_full_sampling=args.add_fps_full_sampling)
    elif args.command=='create_mini_amass_dataset_obj':
        out_file=os.path.join(args.output_object_dir,args.output_object_filename)
        sampling.save_mini_amass_obj(input_amass_obj_full_filepath=args.input_amass_obj_full_filepath,\
                output_mini_amass_full_filepath=out_file,overwriteCache=args.overwrite_cache,add_fps_full_sampling=args.add_fps_full_sampling)
    elif args.command=='sample_random':
        dataset_object_full_filepath=os.path.join(args.dataset_object_dir,args.dataset_object_filename)
        sampling.sample(method=args.command,num_of_frames_to_sample=args.num_of_samples,\
                dataset_object_full_filepath=dataset_object_full_filepath,\
                output_sample_dir=args.output_sample_dir,\
                save_npz_visualization_file=args.save_output_npz_file_for_sampling,\
                save_sampling_histogram_figure=args.save_output_sampling_histogram_figure,
                hist_n_bins=args.n_bins_for_histogram_figure,show_fig=args.show_histogram_figure,
                seed=args.seed)
    elif args.command=='sample_furtherst':
        dataset_object_full_filepath=os.path.join(args.dataset_object_dir,args.dataset_object_filename)
        if args.comp_device=='default':
            args.comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sampling.sample(method=args.command,num_of_frames_to_sample=args.num_of_samples,\
                dataset_object_full_filepath=dataset_object_full_filepath,\
                output_sample_dir=args.output_sample_dir,\
                save_npz_visualization_file=args.save_output_npz_file_for_sampling,\
                save_sampling_histogram_figure=args.save_output_sampling_histogram_figure,
                hist_n_bins=args.n_bins_for_histogram_figure,show_fig=args.show_histogram_figure,
                seed=args.seed,
                \
                comp_device=args.comp_device,iterations_per_sample=args.iterations_per_sample,
                only_allow_samples_that_diveded_by=args.only_allow_samples_that_diveded_by,
                path_to_theta_vector=args.path_to_theta_vector,farthest_sampling_method=args.farthest_sampling_method)
        print('tmp')

if __name__=="__main__":
    args = parse_args()
    print(args)
    assertVaildInput(args)
    if args.check_if_sure and not cli_utils.userIsSure(args):
        exit()
    main(args)
