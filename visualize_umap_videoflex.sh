#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=viz_umap_videoflex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_umap_videoflex_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

BASE="--dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_videoflextok \
    --no-source \
    --max-samples 10000 \
    --max-games 10 \
    --games-seed 42 \
    --method umap \
    --filter-actions right,left,crouch,jump"

echo "=== small font ==="
python New_stuff/visualize_latents.py $BASE \
    --out-dir ./plots/videoflextok_umap/small \
    --dot-size 10 \
    --legend-fontsize 9

echo "=== medium font ==="
python New_stuff/visualize_latents.py $BASE \
    --out-dir ./plots/videoflextok_umap/medium \
    --dot-size 15 \
    --legend-fontsize 12

echo "=== large font ==="
python New_stuff/visualize_latents.py $BASE \
    --out-dir ./plots/videoflextok_umap/large \
    --dot-size 20 \
    --legend-fontsize 16
