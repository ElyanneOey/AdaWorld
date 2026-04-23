#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=get_checkpoint
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/get_checkpoint_%A.out

# No GPU or conda needed — this is just a file download

# SLURM_SUBMIT_DIR is the directory where you ran sbatch from
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

mkdir -p "${SCRIPT_DIR}/checkpoints"

wget --show-progress \
    https://huggingface.co/Little-Podi/AdaWorld/resolve/main/lam.ckpt \
    -O "${SCRIPT_DIR}/checkpoints/lam.ckpt"
