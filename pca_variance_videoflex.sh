#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=pca_variance_videoflex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/pca_variance_videoflex_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/visualize_latents.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_videoflextok \
    --no-source \
    --out-dir ./plots/pca_variance_videoflex \
    --max-samples 10000 \
    --method pca \
    --filter-actions "right,left,crouch,jump"
