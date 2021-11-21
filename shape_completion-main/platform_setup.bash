#!/bin/bash
#---------------------------------------------------------------------------------------------#
#                                       		Prep
#---------------------------------------------------------------------------------------------#
# [1] Download and install Anaconda + PyCharm/other IDEs
#     * If anaconda is already installed, make sure it is in the PATH variable 
#       - in some versions, add only condabin 
#       - in some versions, add both conda bin and Scripts in the anaconda directory 
# [2] For Windows: Install Git Bash 
# [3] Pull the relevant code from Github 
# [4] Run this script by opening up the GitBash shell in the code and running: 
#     Command: bash ./requirements.txt
# [5] Install support for \\gip-main\data on Z: and \\132.68.39.11\gipfs on R: 
#---------------------------------------------------------------------------------------------#
#                                       	
#---------------------------------------------------------------------------------------------#
# Opens and activates a conda env for Python 3.8.0 named DeepShape
# git config --global user.name "Ido Imanuel"
# git config --global user.email "ido.imanuel@gmail.com"
# git remote set-url origin https://github.com/iimanu/Deep-Shape.git

# eval "$(conda shell.bash hook)"
# conda update -y -n base -c defaults conda
# conda create -y -n ProjectHY python=3.8.11 
conda activate ProjectHY
#---------------------------------------------------------------------------------------------#
#                                     	Primary Dependencies
#---------------------------------------------------------------------------------------------#

# Install GPU Computing modules 
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tensorboard
pip install pytorch-lightning
pip install test-tube
pip install decorator
# Primary 3D Geometry Modules:  
pip install trimesh
pip install scikit_learn
pip install scipy 
pip install --user open3d

# Primary Visualizers:
pip install pyvista==0.26.1
pip install matplotlib
pip install seaborn

# Utilities: 
pip install better_exceptions
pip install pycollada
pip install psutil
pip install yagmail
pip install meshio
pip install plyfile # Consider removing
pip install matplotlib
pip install scipy
pip install scikit_learn
pip install npzviewer

# TO REMOVE: 
# gdist
# pip install probreg
# 
# pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# pip install torch-geometric
# conda install -y -c conda-forge point_cloud_utils # Optional 
# pip install tensorflow # Optional 
#---------------------------------------------------------------------------------------------#
#                            			Collaterals
#---------------------------------------------------------------------------------------------#
# [*] Optional: If you want to support full tensoboard features, install tensorflow via:
#     Command: pip install tensorflow
#
# [*] Optional: Some very specific features require point_cloud_utils by fwilliams: 
#      pip install git+git://github.com/fwilliams/point-cloud-utils
#      or conda install -c conda-forge point_cloud_utils
#
# [*] GMAIl Credentials - We use the yagmail package to send the Tensorboard logs to
#      a shared email account. You can configure the email account by placing a txt
#      file under data/collaterals/gmail_credentials.txt with the following information:
#      user=yourmail@gmail.com
#      pass=yourpassword
#---------------------------------------------------------------------------------------------#
#                            	Collaterals - Windows only
#---------------------------------------------------------------------------------------------#
# [*]  Optional: In order to support nn.visualize(), please install Graphviz
#  	   *  Surf to: https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi
#      *  Install the application to C:\Program Files (x86)\Graphviz2.38
#      *  Add to your PATH variable: "C:\Program Files (x86)\Graphviz2.38\bin"
#---------------------------------------------------------------------------------------------#
#                			Important Notes (to save painful debugging)
#---------------------------------------------------------------------------------------------#
# [*]  PyRender support for the Azimuthal Projections: 
#	   *  Location: src/data/prep
#      *  Usage: Only under Linux. Please see prep_main.py for more details
#      *  Compilation: executable is supplied, but you might need to compile it again on 
#        a different machine. Remember to fix the GLM include path in compile.sh
#        CUDA runtime 10.2 must be installed to enable compilation. We recommend the 
#        deb(local) setting. 
#         
# [*] PyCharm Users: Make sure that 'src/core' is enabled as the Source Root:
#---------------------------------------------------------------------------------------------#
#
#---------------------------------------------------------------------------------------------#
