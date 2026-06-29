#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=list_actions_skipped
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/list_actions_skipped_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/extract_jump_frames.py \
    --video-dir /scratch-shared/scur0531/skipped_frames_v0.0.0_noeffect \
    --list-actions
