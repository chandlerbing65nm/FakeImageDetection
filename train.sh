#!/bin/bash

# Define the arguments for your training script
NUM_GPU=2
NUM_EPOCHS=100
PROJECT_NAME="Masked-ResNet"
RESNET_MODEL="RN50"
MASK_TYPE="spectral"
RATIO=20
BATCH_SIZE=64

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --wandb_online \
  --project_name $PROJECT_NAME \
  --resnet_model $RESNET_MODEL \
  --early_stop \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE
