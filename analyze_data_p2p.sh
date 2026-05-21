#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=analyze_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_data_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/analyze_data.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source adaworld \
    --out-dir ./plots/analysis/p2p \
    --results-dir ./results \
    --video-dir /scratch-shared/FoMo-Atomic-Actions/open-p2p-subset
