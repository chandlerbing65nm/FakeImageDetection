#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Ojha_CVPR23"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50_npr" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="fourier" # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high
RATIO=15 # automatically becomes RATIO=0 if MASK_TYPE="nomask"
BATCH_SIZE=64

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
