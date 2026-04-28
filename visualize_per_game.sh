#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=per_game_viz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/per_game_viz_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

# Runs PCA for every game separately, colored by action.
# Plots saved to ./plots/per_game/<game_name>_pca.png
python New_stuff/visualize_latents.py \
    --dump-dir ./latent_actions_dump \
    --out-dir ./plots \
    --max-samples 999999 \
    --method pca \
    --per-game \
    --per-game-method pca \
    --min-samples 20
