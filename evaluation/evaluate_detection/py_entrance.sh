#!/bin/bash
#export CONDA_DIR="/home/emotion/conda"
#export PATH="/home/emotion/conda/bin:/home/emotion/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
script_full_path=$(dirname "$0")
cd $script_full_path
python evaluate.py $1 $2
