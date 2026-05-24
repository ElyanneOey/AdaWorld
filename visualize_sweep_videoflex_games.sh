#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=viz_sweep_videoflex
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/viz_sweep_videoflex_games_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/visualize_latents.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_videoflextok \
    --no-source \
    --out-dir ./plots/videoflextok_games \
    --max-samples 10000 \
    --max-games 10 \
    --method all \
    --sweep \
    --sweep-color both \
    --umap-n-neighbors 15,50,100 \
    --umap-min-dist 0.1,0.5,0.8 \
    --tsne-perplexity 5,30,50,100 \
    --filter-actions "right,left,crouch,jump"
