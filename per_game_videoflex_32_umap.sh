#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=per_game_videoflex_32_umap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/per_game_videoflex_32_umap_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/visualize_latents.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_videoflextok \
    --no-source \
    --out-dir ./plots/per_game_videoflex_32 \
    --max-samples 10000 \
    --method umap \
    --per-game \
    --per-game-method umap \
    --min-samples 20 \
    --filter-actions "right,left,crouch,jump" \
    --max-features 192 \
    --dot-size 50 \
    --legend-fontsize 16 \
    --fig-width 20 \
    --fig-height 12 \
    --title-fontsize 20 \
    --clean-game-names
