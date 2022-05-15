#!/bin/bash

###
# CS236781: Deep Learning
# jupyter-lab.sh
#
# This script is intended to help you run jupyter lab on the course servers.
#
# Example usage:
#
# To run on the gateway machine (limited resources, no GPU):
# ./jupyter-lab.sh
#
# To run on a compute node:
# srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
#

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=ProjectHY

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
export PYTHONPATH=/home/yiftach.ede/Ramp/shape_completion-main/src/core/visualize:$PYTHONPATH.
# mkdir ~/gipfs
# mount -o ro //132.68.39.206/gipfs ~/gipfs
# jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
wandb login 19e347e092a58ca11a380ad43bd1fd5103f4d14a
export WANDB_MODE="offline"
#xvfb-run -a -s "-screen 0 1440x900x24" python src/core/main.py --exp_name baseline_full_volume_gipdeep_fixed_run --baseline --body_part_volume_weights 1 1 1 1 1 1 --version 2 --visualization_run
xvfb-run -a -s "-screen 0 1440x900x24" python src/core/main.py --exp_name LSTM_12_window_size --run_windowed_lstm_decoder --batch_size 12 --lr 0.003 --window_size 12 --stride 16 --version 4 --body_part_volume_weights 0 0 0 0 0 0 
