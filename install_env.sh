#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=install_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_env_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

create env
conda create -n adaworld_elyanne python=3.10 -y

# activate env
source activate adaworld_elyanne

# install requirements
# pip install -r requirements.txt