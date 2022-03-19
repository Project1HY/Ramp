import mapping_utils
import argparse
import cli_utils

base_dataset_dir = '/home/yiftach.ede/datasets/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_dir", help="path to the amss directory", type=str, default=folder3)
    parser.add_argument("--overwrite_cache", help="do we overwrite cache", type=cli_utils.str_to_bool, default=False)
    args = parser.parse_args(args)
    return args

def main(args):
    #mapping_utils.sample_mini_amass(root_dataset_dir=args.amass_dir,method='farthest',num_of_frames_to_sample=700,overwriteCache=args.overwrite_cache)
    #mapping_utils.sample_mini_amass(root_dataset_dir=args.amass_dir,method='farthest',num_of_frames_to_sample=20000,overwriteCache=args.overwrite_cache)
    amass = mapping_utils.get_amass_obj(root_dataset_dir=args.amass_dir,overwriteCache=args.overwrite_cache)
    res=amass.get_dataset_name_and_actor_name_for_actor_with_the_least_frames()
    print(res)
    print('a')
    #mapping_utils.sample_one_of_amass_datasets(root_dataset_dir=args.amass_dir,req_dataset_name='HumanEva',method='farthest',num_of_frames_to_sample=100,overwriteCache=args.overwrite_cache)

if __name__=="__main__":
    #mapping_utils.print_statistics(root_dataset_dir=folder3,overwriteCache=False)
    #mapping_utils.sample_mini_amass(root_dataset_dir=folder3,overwriteCache=False)
    args = parse_args()
    main(args)

