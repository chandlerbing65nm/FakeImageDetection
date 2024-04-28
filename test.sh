#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Ojha_CVPR23"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50" # clip, RN50_mod or RN50
MASK_TYPE="nomask" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=0
BATCH_SIZE=64
SEED=44

CHECKPOINT="./checkpoints/pruning/ungrouped/lnstructured_20%data.pth"

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
  -- \
  --seed $SEED \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --checkpoint_path ${CHECKPOINT} \
