#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50" # clip, RN50_mod or RN50
BATCH_SIZE=64
pruned_folder='./pruned_ckpts'
pruned_filename='pruned_model_1_pruned:0.2'

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU permaprune.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --folder_path $pruned_folder \
  --filename $pruned_filename \
  --pruned_model \