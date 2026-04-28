#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=install_req
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_req_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1


# activate env
source activate adaworld_elyanne

# downgrade setuptools — version 82 breaks pkg_resources with lightning==2.0.8
pip install "setuptools==69.5.1"

# install PyTorch with CUDA 11.8 first (not in requirements.txt)
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# install remaining requirements
pip install -r requirements.txt