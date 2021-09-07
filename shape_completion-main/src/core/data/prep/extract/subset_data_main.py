import os
from pathlib import Path
from shutil import copy, copytree
from data.index import HierarchicalIndexTree
from util.time import timestamp, progress
from util.structs.custom_dicts import vdict
from util.container import deep_dict_convert
from util.strings import banner
from util.fs import pkl_dump

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

TGT_DIR = (Path(__file__).parents[0]).resolve()
# TODO - Correct Filepaths
SRC_DIR = Path(r'Z:\ShapeCompletion\Mixamo')
SRC_FBX_DIR = SRC_DIR / 'spares' / 'collaterals' / 'fbx_sequences' / 'MPI-FAUST'
SRC_FULL_DIR = SRC_DIR / 'full'
SRC_2ND_FULL_DIR = SRC_DIR / 'full_from_converted_fbx_to_collada'
SRC_PROJ_DIR = SRC_DIR / 'projections'
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

SEQUENCES = ['Air Squat', 'Arm Stretching', 'Bartending', 'Sprint Turn', 'Standing', 'Stomping',
             'Talking At Watercooler', 'Throwing', 'Turn 90 Left', 'Turn 90 Right', 'Victory', 'Walking',
             'Waving']
N_PROJECTIONS = 2


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def main():
    assert TGT_DIR.is_dir(), "Invalid Target Directory"
    assert SRC_DIR.is_dir(), "Invalid Source Directory"
    dump_dir: Path = TGT_DIR / f'MiniMixamoPyProj_{timestamp(for_fs=True)}'
    dump_dir.mkdir(parents=False, exist_ok=False)
    print(f'Created dump directory {dump_dir}')

    subject_names = os.listdir(SRC_FBX_DIR)
    collect_fbx(dump_dir=dump_dir, subject_names=subject_names)
    collect_mesh_and_projections(dump_dir=dump_dir, subject_names=subject_names)
    write_hit(dump_dir, subject_names)


def collect_fbx(dump_dir, subject_names):
    for name in progress(subject_names, desc='FBX Move'):
        src_subject_fbx_dir = SRC_FBX_DIR / name
        tgt_subject_fbx_dir = dump_dir / 'collaterals' / 'fbx' / name
        tgt_subject_fbx_dir.mkdir(parents=True, exist_ok=False)
        for seq_name in SEQUENCES:
            tgt_file = src_subject_fbx_dir / f'{seq_name}.fbx'
            assert tgt_file.is_file(), f"Could not find {tgt_file}"
            copy(src=tgt_file, dst=tgt_subject_fbx_dir)
    banner('Finished copying FBX files')


def collect_mesh_and_projections(dump_dir, subject_names):
    for sub_name in subject_names:
        src_subject_full_dir = SRC_FULL_DIR / sub_name
        src_subject_2nd_full_dir = SRC_2ND_FULL_DIR / sub_name
        assert src_subject_full_dir.is_dir()
        tgt_subject_full_dir = dump_dir / 'full' / sub_name
        for seq_name in progress(SEQUENCES, desc=f'Subject {sub_name} Move'):
            src_dir = src_subject_full_dir / seq_name
            sec_src_dir = src_subject_2nd_full_dir / seq_name
            tgt_dir = tgt_subject_full_dir / seq_name
            tgt_dir.mkdir(parents=True, exist_ok=False)
            obj_files = list(src_dir.glob('*.obj'))
            assert len(obj_files) > 0, "Empty directory"
            for obj_file in obj_files:
                pkl_file = src_dir / f'{obj_file.stem}.pkl'
                if not pkl_file.is_file():
                    pkl_file = sec_src_dir / f'{obj_file.stem}.pkl'
                    assert pkl_file.is_file()
                copy(src=obj_file, dst=tgt_dir)
                copy(src=pkl_file, dst=tgt_dir)

            # Handle relevant projections
            src_proj_dir = SRC_PROJ_DIR / sub_name / seq_name
            tgt_proj_dir = dump_dir / 'projections' / sub_name / seq_name
            # Assuming projections are OK - do not need validation
            copytree(src=src_proj_dir, dst=tgt_proj_dir)
    banner('Finished copying mesh, joints and projection files')


def write_hit(dump_dir, subject_names):
    hit = vdict()
    for sub_name in progress(subject_names, desc='Subjects'):
        for seq_name in SEQUENCES:
            src_seq_dir = SRC_FULL_DIR / sub_name / seq_name  # dump_dir / 'full'  # Can also run over the completions
            obj_files = list(src_seq_dir.glob('*.obj'))
            assert len(obj_files) > 0
            for obj_file in obj_files:
                hit[sub_name][seq_name][obj_file.stem] = N_PROJECTIONS

    hit = HierarchicalIndexTree(deep_dict_convert(hit, dict), in_memory=True)
    print(hit)
    tgt_fp = dump_dir / f'mini_mixamo_hit.pkl'
    pkl_dump(tgt_fp, hit)
    print(f'Created index at {tgt_fp.resolve()}')


if __name__ == '__main__':
    main()
