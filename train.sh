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
MODEL_NAME="RN50_npr" # RN50, RN50_mod, RN50_npr, CLIP_vitl14, MNv2, SWIN_t, VGG11
MASK_TYPE="fourier" # nomask, fourier, pixel, patch, cosine, wavelet, translate, rotate
BAND="all" # all, low, mid, high, low+mid, low+high, mid+high ##### add +prog if using progressive masking
RATIO=15
BATCH_SIZE=128
WANDB_ID="2w0btkas"
RESUME="from_last" # from_last or from_best
learning_rate=0.0001 # 0.0001 * NUM_GPU

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the distributed training command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29800 train.py \
  -- \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --lr ${learning_rate} \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained \
  # --resume_train $RESUME \
  # --clip_grad \
  # --debug \