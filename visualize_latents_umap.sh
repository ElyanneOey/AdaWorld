#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=visualize_umap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/visualize_umap_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/visualize_latents.py \
    --dump-dir ./latent_actions_dump \
    --out-dir ./plots \
    --max-samples 10000 \
    --method umap \
    --umap-3d
