#!/bin/bash

# Get the current date
current_date=$(date)
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10000

MODEL_NAME="RN50" # RN50_mod, RN50, clip
MASK_TYPE="cosine" # nomask, fourier, pixel, patch, cosine, wavelet
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high ##### add +prog if using progressive masking
RATIO=15

BATCH_SIZE=124
RESUME_PTH="./checkpoints/mask_15/rn50_modft_prog_spectralmask_last_ep48.pth" # always use from_last
LEARNING_RATE=0.0002
SEED=44

EARLY_STOP="True"
PRETRAINED="True"
TRAIN_CLIP="False"

FINETUNE_PTH="./checkpoints/mask_0/rn50ft.pth"
USE_SMALL_DATA="True"

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
  -- \
  --pretrained $PRETRAINED \
  --seed $SEED \
  --num_epochs $NUM_EPOCHS \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --lr $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --early_stop $EARLY_STOP \
  --trainable_clip $TRAIN_CLIP \
#   --resume_pth $RESUME_PTH \
  # --smallset $USE_SMALL_DATA \
  # --finetune_pth $FINETUNE_PTH \