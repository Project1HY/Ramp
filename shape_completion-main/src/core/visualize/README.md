intresting scripts (there are more that not on this README):
1.gui mesh editor launcher
2.mapping
3.sampling
# prerequests:
## my linux machine
`cd` to correct folder and change to right env
```
cd 
conda activate py37
```
## remote windows machine
`cd` to correct folder and change to right env
```
cd D:\Users\Omer\git\shape_completion\src\visuallise
conda activate DeepShape_1
set LANG="en_us"
set smplh_dir="D:\Users\Omer\data\prior_dirs\SLMPH\smplh"
set dmpl_dir="D:\Users\Omer\data\prior_dirs\DMPL\dmpls"
set initial_model_npz="D:\Users\Omer\data\datasets\AMASS\tar\amass_dir\ACCAD\Female1General_c3d\A1 - Stand_poses.npz"
set amass_dir="D:\Users\Omer\data\datasets\AMASS\tar\amass_dir"

set PRIOR_DIR_FLAGS=^
--smplh_dir=%smplh_dir% ^
--dmpl_dir=%dmpl_dir% ^
--initial_model_npz=%initial_model_npz%

set INITIAL_MODEL_NPZ_FLAG=^
--initial_model_npz=%initial_model_npz%

set AMASS_DIR_FLAG=^
--amass_dir=%amass_dir%

```

# 1.gui mesh editor launcher

## my linux machine

## remote windows machine
python gui_mesh_editor_launcher.py %PRIOR_DIR_FLAGS% %INITIAL_MODEL_NPZ_FLAG%

# 2.mapping

## remote windows machine
python mapping_main.py %AMASS_DIR_FLAG%

# 3.sampling
