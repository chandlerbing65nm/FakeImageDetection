#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
NUM_GPU=2
NUM_EPOCHS=10000
PROJECT_NAME="Masked-ResNet"
MODEL_NAME="RN50"
MASK_TYPE="spectral"
RATIO=15
BATCH_SIZE=64
WANDB_ID="u336igyq"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --wandb_online \
  --pretrained \
  --resume_train \
  --wandb_run_id $WANDB_ID \
