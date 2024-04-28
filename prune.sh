#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
MODEL_NAME="RN50" # RN50_mod, RN50, clip_vitl14, clip_rn50
MASK_TYPE="nomask" # nomask, spectral, pixel, patch
BAND="low" # all, low, mid, high
RATIO=70
BATCH_SIZE=128
learning_rate=0.0002
SEED=44
SMALL_DATA="True"

# Define the arguments for pruning
CHECKPOINT="./checkpoints/mask_0/rn50ft.pth"
CONV_PRUNING_RATIO=0.0
PRUNING_ITER=1
DATASET="ForenSynths" # ForenSynths, LSUNbinary
PRUNING_TEST="False" # for pruning in eval mode without finetuning
PRUNING_FT="True"

CLIP_GRAD="False" # for pruning finetuned clip model
PRETRAINED="False" # if use ImageNet weights, setting pretrained=True
PRUNING_VALUES_FILE="./pruning_amounts.txt"

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU prune.py \
  -- \
  --seed $SEED \
  --num_epochs $NUM_EPOCHS \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --lr ${learning_rate} \
  --batch_size $BATCH_SIZE \
  --conv2d_prune_amount ${CONV_PRUNING_RATIO} \
  --pruning_rounds ${PRUNING_ITER} \
  --checkpoint_path ${CHECKPOINT} \
  --pruning_test ${PRUNING_TEST} \
  --pruning_ft ${PRUNING_FT} \
  --clip_grad ${CLIP_GRAD} \
  --dataset ${DATASET} \
  --pretrained ${PRETRAINED} \
  --smallset ${SMALL_DATA} \
  --conv2d_prune_amount_file ${PRUNING_VALUES_FILE} \