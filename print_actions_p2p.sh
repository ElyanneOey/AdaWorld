#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=0
#SBATCH --job-name=print_actions_p2p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/print_actions_p2p_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate adaworld_elyanne

cd ${SLURM_SUBMIT_DIR}

echo "=== adaworld ==="
python new_stuff/print_actions.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source adaworld

echo "=== olafworld ==="
python new_stuff/print_actions.py \
    --dump-dir /gpfs/home3/scur0531/AdaWorld/latent_actions_dump_2 \
    --source olafworld
