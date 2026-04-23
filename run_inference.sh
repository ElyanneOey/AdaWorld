#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=lam_inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/inference_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate adaworld_elyanne

# SLURM_SUBMIT_DIR is the directory where you ran sbatch from
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# Hardcoded path to the mp4 file
MP4="${SCRIPT_DIR}/../../random_actions_data/dataset/retro_act_v0.0.0/retro_8eyes-nes_v0.0.0/000000/000000/frames.mp4"

# Checkpoint downloaded by get_checkpoint.sh
CHECKPOINT="${SCRIPT_DIR}/checkpoints/lam.ckpt"

# The dataset loader expects mp4 files inside a test/ subfolder.
# We create data/test/ inside the repo and symlink the mp4 there.
mkdir -p "${SCRIPT_DIR}/data/test"
ln -sf "${MP4}" "${SCRIPT_DIR}/data/test/frames.mp4"

# Run inference
cd "${SCRIPT_DIR}/lam"

python main.py test \
    --ckpt_path "${CHECKPOINT}" \
    --config config/lam.yaml \
    --data.data_root="${SCRIPT_DIR}" \
    --data.env_source="data" \
    --data.num_frames=2 \
    --data.batch_size=1 \
    --trainer.devices=1

# Results saved to: lam/exp_imgs/test_step000000.png
# (side-by-side: frame0 | ground truth frame1 | predicted frame1)
