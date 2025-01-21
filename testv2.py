import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import timm
import argparse
import random

import torchvision.models as vis_models

from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN50', 'RN50_mod', 'RN50_npr', 'CLIP_vitl14', 'MNv2', 'SWIN_t', 'VGG11'
        ],
        help='Type of model to use; includes ResNet'
        )
    parser.add_argument(
        '--mask_type', 
        default='fourier', 
        choices=[
            'fourier',
            'cosine',
            'wavelet',
            'pixel', 
            'patch',
            'translate',
            'rotate',
            'nomask'], 
        help='Type of mask generator'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=[
            'all', 'low', 'mid', 'high', 'low+mid', 'low+high', 'mid+high',]
        )
    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        help='For pretraining'
        )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=50,
        help='Ratio of mask to apply'
        )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Batch Size'
        )
    parser.add_argument(
        '--data_type', 
        default="Wang_CVPR20", 
        type=str, 
        choices=['Wang_CVPR20', 'Ojha_CVPR23'], 
        help="Dataset Type"
        )
    parser.add_argument(
        '--checkpoint_path', 
        type=str,
        )
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')

    model_name = args.model_name.lower()


    # Define the path to the results file
    checkpoint_foldername = os.path.basename(os.path.dirname(args.checkpoint_path))
    filename = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    results_path = f'results/ablation/{checkpoint_foldername}_{filename}.txt'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.local_rank}")
    print(f"Model type: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Checkpoint Type: {args.checkpoint_path}")
    print(f"Results saved to: {results_path}")
    print("-" * 30, "\n")

    datasets = {
        'ProGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/progan',
        'CycleGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/cyclegan',
        'BigGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/biggan',
        'StyleGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/stylegan',
        'GauGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/gaugan',
        'StarGAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/stargan',
        'DeepFake': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/deepfake',
        'SITD': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/seeingdark',
        'SAN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/san',
        'CRN': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/crn',
        'IMLE': '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/testing/imle',
        'Guided': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/guided',
        'LDM_200': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/ldm_200',
        'LDM_200_cfg': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/ldm_200_cfg',
        'LDM_100': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/ldm_100',
        'Glide_100_27': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/glide_100_27',
        'Glide_50_27': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/glide_50_27',
        'Glide_100_10': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/glide_100_10',
        'DALL-E': '/mnt/SCRATCH/chadolor/Datasets/Ojha_CVPR2023/dalle',       
    }

    # Initialize a counter
    dataset_count = len(datasets)

    for dataset_name, dataset_path in datasets.items():
        if dist.get_rank() == 0:
            print(f"\nEvaluating {dataset_name}")

        avg_ap, avg_acc, auc = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            args.ratio/100,
            dataset_path,
            args.batch_size,
            args.checkpoint_path,
            device,
            args,
        )
        if dist.get_rank() == 0:
            # Write the results to the file
            with open(f'{results_path}', 'a') as file:
                if file.tell() == 0: # Check if the file is empty
                    file.write("Selected Configuration:\n")
                    file.write("-" * 28 + "\n")
                    file.write(f"Device: {args.local_rank}\n")
                    file.write(f"Model type: {args.model_name}\n")
                    file.write(f"Batch Size: {args.batch_size}\n")
                    file.write(f"Checkpoint Type: {args.checkpoint_path}\n")
                    file.write(f"Results saved to: {results_path}\n")
                    file.write("-" * 28 + "\n\n")
                    file.write("Dataset, Avg.Prec., Acc., AUC\n")
                    file.write("-" * 28)
                    file.write("\n")
                file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}, {auc:.3f}\n")

            # Decrement the counter
            dataset_count -= 1
            if dataset_count == 0:
                with open(f'{results_path}', 'a') as file:
                    file.write("-" * 28 + "\n")
                    file.write("\n")