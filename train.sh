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
MODEL_NAME="RN50" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="fourier" # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate, rotate_translate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high ##### add +prog if using progressive masking
RATIO=15
BATCH_SIZE=128
MASK_CHANNEL="all" # all, r, g, b, 0, 1, 2 (applies to fourier/cosine/wavelet)
COMBINE_AUG="rotate_translate" # none, rotate, translate, rotate_translate (combine with frequency masking)
WANDB_ID="2w0btkas"
RESUME="from_last" # from_last or from_best
learning_rate=0.0001 # 0.0001 * NUM_GPU

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Randomize master port between 29000 and 29999 to avoid clashes
MASTER_PORT=$((29000 + RANDOM % 1000))
echo "Using master port: $MASTER_PORT"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$MASTER_PORT train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --mask_channel $MASK_CHANNEL \
  --combine_aug $COMBINE_AUG \
  --lr ${learning_rate} \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  # --resume_train $RESUME \
  # --clip_grad \
  # --debug \