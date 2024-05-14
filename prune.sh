#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000
MODEL_NAME="clip_rn50" # clip_rn50, rn50, rn50_cifar10, vgg_cifar10
MASK_TYPE="spectral" # nomask, spectral, pixel, patch
BAND="low" # all, low, mid, high
RATIO=70
BATCH_SIZE=128
learning_rate=0.0002
SEED=44
SMALL_DATA="True"

# Define the arguments for pruning
CHECKPOINT="./checkpoints/mask_0/clip_rn50ft.pth"
CALIB_SPARSITY=0.99 # for testing
DESIRED_SPARSITY=0.6 # for finetuning
PRUNING_RNDS=10
DATASET="ForenSynths" # ForenSynths, LSUNbinary, CIFAR10
PRUNING_TEST="False" # for pruning in eval mode without finetuning
PRUNING_TEST_FT="True"
PRUNING_METHOD="rd" # ours, ours_lamp, ours_erk, ours_nomask, lamp_erk, rd, ours_rd

CLIP_GRAD=$( [[ "$CHECKPOINT" == *"clip"* ]] && echo "True" || echo "False" ) # for pruning finetuned clip model
PRETRAINED="False" # if use ImageNet weights, setting pretrained=True

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPU/s with ID/s: $GPUs"

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
  --calib_sparsity ${CALIB_SPARSITY} \
  --desired_sparsity ${DESIRED_SPARSITY} \
  --pruning_rounds ${PRUNING_RNDS} \
  --checkpoint_path ${CHECKPOINT} \
  --pruning_test ${PRUNING_TEST} \
  --clip_grad ${CLIP_GRAD} \
  --dataset ${DATASET} \
  --pretrained ${PRETRAINED} \
  --smallset ${SMALL_DATA} \
  --pruning_test_ft ${PRUNING_TEST_FT} \
  --pruning_method ${PRUNING_METHOD} \