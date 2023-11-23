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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'clip',
        ],
        help='Type of model to use; includes ResNet variants'
        )
    parser.add_argument(
        '--mask_type', 
        default='spectral', 
        choices=[
            'patch', 
            'spectral',
            'pixel', 
            'nomask'], 
        help='Type of mask generator'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=[
            'all', 'low', 'mid', 'high',]
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
        '--device', 
        default="cuda:0" if torch.cuda.is_available() else "cpu", 
        type=str, 
        help="Device to use (default: auto-detect)"
        )

    args = parser.parse_args()

    device = torch.device(args.device)
    model_name = args.model_name.lower()
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}_{band}{args.mask_type}mask.pth'
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}.pth'

    # Define the path to the results file
    results_path = f'results/{args.data_type.lower()}'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model_name}{finetune}_{band}{args.mask_type}mask{ratio}.txt'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.device}")
    print(f"Dataset Type: {args.data_type}")
    print(f"Model type: {args.model_name}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_path}/{filename}")
    print("-" * 30, "\n")

    if args.data_type == 'Wang_CVPR20':
        datasets = {
            'ProGAN': '../../Datasets/Wang_CVPR2020/testing/progan',
            'CycleGAN': '../../Datasets/Wang_CVPR2020/testing/cyclegan',
            'BigGAN': '../../Datasets/Wang_CVPR2020/testing/biggan',
            'StyleGAN': '../../Datasets/Wang_CVPR2020/testing/stylegan',
            'GauGAN': '../../Datasets/Wang_CVPR2020/testing/gaugan',
            'StarGAN': '../../Datasets/Wang_CVPR2020/testing/stargan',
            'DeepFake': '../../Datasets/Wang_CVPR2020/testing/deepfake',
            'SITD': '../../Datasets/Wang_CVPR2020/testing/seeingdark',
            'SAN': '../../Datasets/Wang_CVPR2020/testing/san',
            'CRN': '../../Datasets/Wang_CVPR2020/testing/crn',
            'IMLE': '../../Datasets/Wang_CVPR2020/testing/imle',
        }
    # elif args.data_type == 'GenImage':
    #     datasets = {
    #         'VQDM': '../../Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val',
    #         'Glide': '../../Datasets/GenImage/imagenet_glide/imagenet_glide/val',
    #     }
    elif args.data_type == 'Ojha_CVPR23':
        datasets = {
            'Guided': '../../Datasets/Ojha_CVPR23/guided',
            'LDM_200': '../../Datasets/Ojha_CVPR23/ldm_200',
            'LDM_200_cfg': '../../Datasets/Ojha_CVPR23/ldm_200_cfg',
            'LDM_100': '../../Datasets/Ojha_CVPR23/ldm_100',
            'Glide_100_27': '../../Datasets/Ojha_CVPR23/glide_100_27',
            'Glide_50_27': '../../Datasets/Ojha_CVPR23/glide_50_27',
            'Glide_100_10': '../../Datasets/Ojha_CVPR23/glide_100_10',
            'DALL-E': '../../Datasets/Ojha_CVPR23/dalle',       
        }
    else:
        raise ValueError("wrong dataset type")

    # Initialize a counter
    dataset_count = len(datasets)

    for dataset_name, dataset_path in datasets.items():
        print(f"\nEvaluating {dataset_name}")

        avg_ap, avg_acc, auc = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
            args,
        )

        # Write the results to the file
        with open(f'{results_path}/{filename}', 'a') as file:
            if file.tell() == 0: # Check if the file is empty
                file.write("Selected Configuration:\n")
                file.write("-" * 28 + "\n")
                file.write(f"Device: {args.device}\n")
                file.write(f"Dataset Type: {args.data_type}\n")
                file.write(f"Model type: {args.model_name}\n")
                file.write(f"Ratio of mask: {ratio}\n")
                file.write(f"Batch Size: {args.batch_size}\n")
                file.write(f"Mask Type: {args.mask_type}\n")
                file.write(f"Checkpoint Type: {checkpoint_path}\n")
                file.write(f"Results saved to: {results_path}/{filename}\n")
                file.write("-" * 28 + "\n\n")
                file.write("Dataset, Precision, Accuracy, AUC\n")
                file.write("-" * 28)
                file.write("\n")
            file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}, {auc:.3f}\n")

        # Decrement the counter
        dataset_count -= 1
        if dataset_count == 0:
            with open(f'{results_path}/{filename}', 'a') as file:
                file.write("-" * 28 + "\n")
                file.write("\n")