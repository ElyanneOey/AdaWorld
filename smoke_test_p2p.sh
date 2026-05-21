#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=smoke_p2p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_p2p_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

# Smoke test: loads 200 samples, runs PCA only, no sweep.
# Check logs/smoke_p2p_<jobid>.out to see keys and action labels.
python New_stuff/visualize_latents.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source adaworld \
    --out-dir ./plots/smoke/p2p/adaworld \
    --max-samples 200 \
    --method pca
