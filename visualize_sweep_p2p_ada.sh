#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=viz_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/viz_sweep_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

# Runs parameter sweep for all three methods, colored by both action and game.
# Output structure:
#   plots/sweep/pca/   -- PC1v2, PC1v3, PC2v3, PC1v4 x {action, game}
#   plots/sweep/tsne/  -- perplexity 5, 30, 50, 100   x {action, game}
#   plots/sweep/umap/  -- 3 n_neighbors x 3 min_dist  x {action, game}
python New_stuff/visualize_latents.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump2 \
    --source adaworld \
    --out-dir ./plots/p2p/adaworld \
    --max-samples 10000 \
    --method all \
    --sweep \
    --sweep-color both \
    --umap-n-neighbors 15,50,100 \
    --umap-min-dist 0.1,0.5,0.8 \
    --tsne-perplexity 5,30,50,100
