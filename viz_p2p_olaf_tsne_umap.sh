#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=viz_p2p_olaf_tsne_umap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_p2p_olaf_tsne_umap_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

BASE="--dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source olafworld \
    --max-samples 10000 \
    --filter-actions right,left,crouch,jump \
    --dot-size 50 \
    --legend-fontsize 16 \
    --fig-width 20 \
    --fig-height 12 \
    --title-fontsize 20 \
    --clean-game-names"

echo "=== t-SNE ==="
python New_stuff/visualize_latents.py $BASE \
    --out-dir ./plots/viz_p2p_olaf/tsne \
    --method tsne

echo "=== UMAP ==="
python New_stuff/visualize_latents.py $BASE \
    --out-dir ./plots/viz_p2p_olaf/umap \
    --method umap
