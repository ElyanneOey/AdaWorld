#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_linear
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_linear_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

python New_stuff/train_linear.py \
    --epochs 100 \
    --batch_size 256 \
    --action_hidden_layers 0 \
    --game_hidden_layers 0 \
    --dataset olafworld \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump \
    --per_game \
    --out-csv ./results/per_game_action_accuracy_olaf_no_hidden.csv