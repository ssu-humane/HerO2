#!/bin/bash

# Initialize conda for script usage
source ~/miniconda3/etc/profile.d/conda.sh


# Create and activate environment
conda create -n hero2 python=3.12 -y  # Added -y for non-interactive
sleep 2  # Increased sleep time for readability

source activate hero2
sleep 2

conda info --envs
sleep 2

# Install packages
python3 -m pip install vllm==0.8.5
python3 -m pip install nltk
python3 -m pip install -r requirements.txt