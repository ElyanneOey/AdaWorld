#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=check_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_env_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

echo "===== Active environment ====="
conda info --envs

echo ""
echo "===== Python location ====="
which python
python --version

echo ""
echo "===== Key packages ====="
pip show torch | grep -E "Name|Version"
pip show torchvision | grep -E "Name|Version"
pip show lightning | grep -E "Name|Version"
pip show opencv-python | grep -E "Name|Version"
pip show setuptools | grep -E "Name|Version"

echo ""
echo "===== Full package list ====="
pip list
