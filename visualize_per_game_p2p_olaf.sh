#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=per_game_viz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/per_game_viz_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

# Runs per-game action visualizations for all three methods.
# Output structure:
#   plots/per_game/pca/<game>_pca.png
#   plots/per_game/tsne/<game>_tsne.png
#   plots/per_game/umap/<game>_umap.png

BASE_ARGS="--dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source olafworld \
    --out-dir ./plots/p2p/olafworld \
    --max-samples 999999 \
    --per-game \
    --min-samples 20

echo "=== PCA per game ==="
python New_stuff/visualize_latents.py $BASE_ARGS \
    --method pca \
    --per-game-method pca

echo "=== UMAP per game ==="
python New_stuff/visualize_latents.py $BASE_ARGS \
    --method umap \
    --per-game-method umap

echo "=== t-SNE per game ==="
python New_stuff/visualize_latents.py $BASE_ARGS \
    --method tsne \
    --per-game-method tsne
