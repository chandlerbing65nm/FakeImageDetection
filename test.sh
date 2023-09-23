#!/bin/bash

# Define the arguments for your test script
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50" # RN50_mod or RN50
MASK_TYPE="pixel" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=64
DEVICE="cuda:0"

# Run the test command
python test.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --device $DEVICE
