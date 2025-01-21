#!/bin/bash
# testv2.sh
#
# Script to run testv2.py with distributed single-GPU execution.

###########################################
# Argument Definitions
###########################################
MODEL_NAME="RN50"             # Model type (e.g., RN50, RN50_mod, etc.)
BATCH_SIZE=64                 # Batch size for validation
CHECKPOINT_PATH="checkpoints/ablation/mask_0/RN50_pruneonly_globalprune_amount80.pth"  # Path to the checkpoint to evaluate
LOCAL_RANK=0                  # Local rank for distributed training (set to 0 for single GPU)

###########################################
# Run testv2.py
###########################################
python -m torch.distributed.launch --nproc_per_node=1 testv2.py \
    -- \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --local_rank ${LOCAL_RANK}
