#!/bin/bash

#SBATCH --job-name=deepfake         # Job name (optional)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPUs per task
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --mem=16G                  # Total memory allocation
#SBATCH --partition=gpu            # Partition to submit to
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --time=72:00:00            # Time limit (hh:mm:ss)
#SBATCH --output=logs/rn50_nprft_fouriermask.out

# Load necessary modules (if required)
conda init
conda activate fakeimagedetector

bash train.sh '0'