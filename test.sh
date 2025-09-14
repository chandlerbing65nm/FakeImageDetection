#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="both"  # both, or Wang_CVPR20, or Ojha_CVPR23
MODEL_NAME="RN50" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="fourier"   # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate, rotate_translate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high
RATIO=15 # automatically becomes RATIO=0 if MASK_TYPE="nomask"
BATCH_SIZE=64
MASK_CHANNEL="all"    # all, r, g, b, 0, 1, 2 (applies to fourier/cosine/wavelet)
COMBINE_AUG="translate"   # none, rotate, translate, rotate_translate (combine with frequency masking)

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Randomize master port between 29000 and 29999 to avoid clashes
MASTER_PORT=$((29000 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$MASTER_PORT test.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --mask_channel $MASK_CHANNEL \
  --combine_aug $COMBINE_AUG \
  --batch_size $BATCH_SIZE \
