#!/bin/bash
# prune.sh
#
# This version demonstrates running "prune.py" with a single GPU
# using torchrun (DDP environment with nproc_per_node=1).
# If you prefer no DDP at all, simply run `python prune.py ...` instead.

###########################################
# Argument Definitions
###########################################
PRUNE_MODE="pruneonly"   # Either "pruneonly" or "prunefinetune"
PRUNE_TYPE="localprune"     # Either "localprune" or "globalprune"
PRUNE_AMOUNT=0.5

CHECKPOINT_PATH="/mnt/SCRATCH/chadolor/Datasets/Projects/FakeImageDetector/checkpoints/mask_15/rn50ft_translatemask.pth"
SAVE_FOLDER="/mnt/SCRATCH/chadolor/Datasets/Projects/FakeImageDetector/checkpoints/ablation/mask_15_translate"
MODEL_NAME="RN50"           # E.g. "RN50"
BATCH_SIZE=64
LR=0.0001
FINETUNE_EPOCHS=5

MASK_TYPE="fourier"         # e.g., "nomask", "fourier", "pixel", "patch", etc.
BAND="all"                  # e.g., "all", "low", "mid", "high" ...
RATIO=15                    # Mask ratio / augmentation ratio

TRAIN_DATA_PATH="/mnt/SCRATCH/chadolor/Datasets/Datasets/Wang_CVPR2020/training"
VAL_DATA_PATH="/mnt/SCRATCH/chadolor/Datasets/Datasets/Wang_CVPR2020/validation"

###########################################
# Run prune.py under single-GPU DDP
###########################################
torchrun --nproc_per_node=1 prune.py \
  --prune_mode "${PRUNE_MODE}" \
  --prune_type "${PRUNE_TYPE}" \
  --prune_amount "${PRUNE_AMOUNT}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --save_folder "${SAVE_FOLDER}" \
  --model_name "${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --finetune_epochs "${FINETUNE_EPOCHS}" \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --mask_type "${MASK_TYPE}" \
  --ratio "${RATIO}" \
  --smalltrain \
  # You can add or remove other arguments as needed.

