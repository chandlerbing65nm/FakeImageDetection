#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
MODEL_NAME="rn50_mod" # clip_rn50, rn50, rn50_mod
BATCH_SIZE=16
SEED=44

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# CHECKPOINT="./checkpoints/mask_15/rn50_modft_prog_spectralmask.pth"
CHECKPOINT="./checkpoints/mask_15/rn50_modft_prog_spectralmask.pth"
RESULTS_FOLDER="./results/masking"

# Run the test command
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
  -- \
  --seed $SEED \
  --pretrained \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --checkpoint_path ${CHECKPOINT} \
  --result_folder ${RESULTS_FOLDER} \

# # Loop to run tests for checkpoints ln_strcd_ep0_rnd0 to ln_strcd_ep0_rnd9
# for i in {1..10}
# do
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/ours/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/ours_erk/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/ours_nomask/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/lamp_erk/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/ours_lamp/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/lamp/ep0_rnd${i}.pth"
#     # CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/erk/ep0_rnd${i}.pth"
#     CHECKPOINT="./checkpoints/pruning/${MODEL_NAME}/rd/ep0_rnd${i}.pth"
#     echo "Testing with checkpoint: $CHECKPOINT"

#     # Run the test command
#     python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
#       -- \
#       --seed $SEED \
#       --pretrained \
#       --model_name $MODEL_NAME \
#       --mask_type $MASK_TYPE \
#       --band $BAND \
#       --ratio $RATIO \
#       --batch_size $BATCH_SIZE \
#       --checkpoint_path ${CHECKPOINT} \
    
# done