#!/bin/bash

module purge
module load 2025
module load Anaconda3/2025.06-1
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
