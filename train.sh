#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
PROJECT_NAME="Frequency-Masking"
MODEL_NAME="RN50" # RN50_mod, RN50
MASK_TYPE="spectral" # nomask, spectral, patch
BAND="all" # all, low, mid, high
RATIO=70
BATCH_SIZE=128
WANDB_ID="gxm9g46z"
RESUME="from_last" # from_last or from_best

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  --wandb_online \
  --wandb_run_id $WANDB_ID \
  --resume_train $RESUME 
