#!/bin/bash

#SBATCH --partition=gpu_mig
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

# install requirements
pip install -r requirements.txt