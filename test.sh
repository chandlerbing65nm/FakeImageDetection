#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50" # clip, RN50_mod or RN50
MASK_TYPE="nomask" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=0
BATCH_SIZE=64
SEED=44

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# CHECKPOINT="./checkpoints/pruning/rn50/ln_strcd_ep0_rnd1.pth"
# # CHECKPOINT="./checkpoints/pruning/lamp/lamp_ep0_rnd9.pth"

# # Run the test command
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
#   -- \
#   --seed $SEED \
#   --data_type $DATA_TYPE \
#   --pretrained \
#   --model_name $MODEL_NAME \
#   --mask_type $MASK_TYPE \
#   --band $BAND \
#   --ratio $RATIO \
#   --batch_size $BATCH_SIZE \
#   --checkpoint_path ${CHECKPOINT} \

# Loop to run tests for checkpoints ln_strcd_ep0_rnd0 to ln_strcd_ep0_rnd9
for i in {0..9}
do
    # CHECKPOINT="./checkpoints/pruning/rn50/ln_strcd_ep0_rnd${i}.pth"
    CHECKPOINT="./checkpoints/pruning/lamp/lamp_ep0_rnd${i}.pth"
    echo "Testing with checkpoint: $CHECKPOINT"

    # Run the test command
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
      -- \
      --seed $SEED \
      --data_type $DATA_TYPE \
      --pretrained \
      --model_name $MODEL_NAME \
      --mask_type $MASK_TYPE \
      --band $BAND \
      --ratio $RATIO \
      --batch_size $BATCH_SIZE \
      --checkpoint_path ${CHECKPOINT} \
    
done