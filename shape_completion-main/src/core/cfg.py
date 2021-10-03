import pathlib  # PyCharm tip : Use CTRL + SHIFT + f in to detect where these are used

# ----------------------------------------------------------------------------------------------------------------------
#                                           ADDITIONAL LEARNING PARAMS
# ----------------------------------------------------------------------------------------------------------------------
UNIVERSAL_RAND_SEED = 2147483647  # The random seed. Use datetime.now() For a truly random seed
UNIVERSAL_PRECISION = 'float32'  # float64,float32 or float16. PyTorch defaults to float32.
# TODO - This does not propagate to faces. Is this a problem ?
# TODO - VTK does not work with float16 - should we transform the CPU tensors before plot?

NORMAL_MAGNITUDE_THRESH = 10 ** (-6)  # The minimal norm allowed for vertex normals to decide that they are too small
DEF_LR_SCHED_COOLDOWN = 5  # Number of epoches to wait after reducing the step-size. Works only if LR sched is enabled
DEF_MINIMAL_LR = 1e-6  # The smallest learning step allowed with LR sched. Works only if LR sched is enabled
# ----------------------------------------------------------------------------------------------------------------------
#                                                    COMPLEXITY
# ----------------------------------------------------------------------------------------------------------------------
NON_BLOCKING = True  # Transfer to GPU in a non-blocking method
# ----------------------------------------------------------------------------------------------------------------------
#                                                      ERROR
# ----------------------------------------------------------------------------------------------------------------------
SUPPORTED_IN_CHANNELS = (3, 6, 12)  # The possible supported input channels - either 3, 6 or 12
DANGEROUS_MASK_THRESH = 100  # The minimal length allowed for mask vertex indices.
# ----------------------------------------------------------------------------------------------------------------------
#                                                   FILE SYSTEM
# ----------------------------------------------------------------------------------------------------------------------
ORIGINAL_SOURCE_CODE_DIR: pathlib.Path = pathlib.Path(r"D:\projectHY\Ramp")
SOURCE_CODE_DIR: pathlib.Path = pathlib.Path(__file__).parents[0]
PRIMARY_RESULTS_DIR: pathlib.Path = SOURCE_CODE_DIR / 'results'
PRIMARY_DATA_DIR: pathlib.Path = pathlib.Path(r"R:\Mano\data\DFaust\DFaust")
PRIMARY_EXAMPLE_DIR: pathlib.Path = ORIGINAL_SOURCE_CODE_DIR / 'exp' / 'test' / 'examples'
TEST_MESH_HUMAN_PATH: pathlib.Path = PRIMARY_EXAMPLE_DIR / 'fat_man.off'
TEST_MESH_HAND_PATH: pathlib.Path = PRIMARY_EXAMPLE_DIR / 'hand1_8k.off'
TEST_CAMERA_PATH: pathlib.Path = PRIMARY_EXAMPLE_DIR / 'camera.ply'
TEST_SCAN_PATH: pathlib.Path = PRIMARY_EXAMPLE_DIR / 'test_scan_006.off'
SMPL_TEMPLATE_PATH: pathlib.Path = PRIMARY_DATA_DIR / 'templates' / 'template_color.ply'

GCREDS_PATH: pathlib.Path = PRIMARY_DATA_DIR / 'collaterals' / 'gmail_credentials.txt'
SAVE_MESH_AS = 'ply'  # Currently implemented - ['obj','off','ply'] # For lightning.assets.completion_saver

MIXAMO_PATH_GIP = pathlib.Path(r"Z:\ShapeCompletion\synthetic\MixamoSkinned")
MIXAMO_PATH_GAON9 = pathlib.Path(r"R:\MixamoSkinned")
MINI_MIXAMO_PATH_GAON9 = pathlib.Path(r"R:\MiniMixamoSkinned13")
BROKEN_ANIMATION = pathlib.Path(r"R:\MixamoSkinned\broken_animation")
# ----------------------------------------------------------------------------------------------------------------------
#                                                      RECORD
# ----------------------------------------------------------------------------------------------------------------------
MIN_EPOCHS_TO_SEND_EMAIL_RECORD = 1  # We supress any results that have trained for less than 200 epochs
# ----------------------------------------------------------------------------------------------------------------------
#                                  VISUALIZATION - For lightning.assets.plotter
# ----------------------------------------------------------------------------------------------------------------------
VIS_N_MESH_SETS = 2  # Parallel plot will plot 8 meshes for each mesh set - 4 from train, 4 from vald
VIS_STRATEGY = 'cloud'  # spheres,cloud,mesh  - Choose how to display the meshes
VIS_CMAP = 'summer'  # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# We use two colors: one for the mask verts [Right end of the spectrum] and one for the rest [Left end of the spectrum].
VIS_SHOW_GRID = False  # Visualzie with grid?
VIS_SMOOTH_SHADING = False  # Smooth out the mesh before visualization?  Applicable only for 'mesh' method
VIS_SHOW_EDGES = False  # Visualize with edges? Applicable only for 'mesh' method
VIS_SHOW_NORMALS = False  # TODO - Removed support for this for the meantime, to save memory

# C:\Users\oshri.halimi\AppData\Local\Continuum\anaconda3\envs\DeepShape\python.exe
PC_NAME_MAPPING = \
    {
        'GIP-1080-3': '0Right-20Core-8GB',
        'GIP-2018-2': '1Right-28Core-11GB',
        'GIP-2018-3': '2Right-28Core-11GB',
        'GIP-I7X-1080TI1': '2Left-28Core-11GB',
        'GIP-2018-4': '1Left-28Core-11GB',
        'GIP-I7X-2080TI2': '0RightUbuntu-20Core-8GB',
        'GIP183': 'OfficeMor-24Core-12GB',
        'GIP-2018-1': 'LeftDoor-28Core-11GB',
        'GIP-i7x-2080ti2': 'RightDoor-68Core-11GB',
        'GIP-2018-5': '0Left-28Core-11GB',
        'GIP-1080-2': 'OfficeLoser-20Core-8GB',
        'DESKTOP-40EQETC': "Mano's PC - Debug",
        'smosesli-gpu': "Moshe's Computer"  # TODO
    }

# OSHRI PC
# 0LeftUbuntu-28Core-11GB
# 2RightUbuntu-28Core-11GB
# 0RightUbuntu-20Core-8GB
