#!/bin/bash

# Define the arguments for your test script
DATA_TYPE="ForenSynths"  # GenImage or ForenSynths
RESNET_MODEL="RN50"
MASK_TYPE="edge"
RATIO=30
BATCH_SIZE=64
DEVICE="cuda:0"

# Run the test command
python test.py \
  --data_type $DATA_TYPE \
  --resnet_model $RESNET_MODEL \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --device $DEVICE
