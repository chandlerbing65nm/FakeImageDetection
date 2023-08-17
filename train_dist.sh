#!/bin/bash

# Define the arguments for your training script
NUM_GPU=2
NUM_EPOCHS=300
WANDB_ONLINE=False
PROJECT_NAME="Masked-ResNet"
RESNET_MODEL="RN50"
EARLY_STOP=True
MASK_TYPE="edge"
RATIO=30

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --wandb_online $WANDB_ONLINE \
  --project_name $PROJECT_NAME \
  --resnet_model $RESNET_MODEL \
  --early_stop $EARLY_STOP \
  --mask_type $MASK_TYPE \
  --ratio $RATIO
