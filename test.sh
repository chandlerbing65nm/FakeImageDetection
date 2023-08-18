#!/bin/bash

# Define the arguments for your test script
DATA_TYPE="ForenSynths"  # GenImage or ForenSynths
MODEL_NAME="RN50"
MASK_TYPE="spectral"
RATIO=15
BATCH_SIZE=64
DEVICE="cuda:0"

# Run the test command
python test.py \
  --data_type $DATA_TYPE \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --device $DEVICE
