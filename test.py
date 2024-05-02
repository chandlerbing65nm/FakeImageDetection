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

from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from commons import get_model_flops, get_model_sparsity

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='rn50',
        type=str,
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
        help="Dataset Type"
        )
    parser.add_argument(
        '--other_model', 
        action='store_true', 
        help='if the model is from my own code'
        )
    parser.add_argument(
        '--conv_features', 
        action='store_true', 
        help='get the conv blocks features'
        )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=44, 
        help='seed'
        )
    parser.add_argument(
        '--checkpoint_path', 
        type=str,
        )
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    args = parser.parse_args()

    seed = args.seed
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
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band
    convfeat = '_convfeats' if args.conv_features else ''

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}_{band}{args.mask_type}mask.pth'
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}.pth'

    checkpoint_path = args.checkpoint_path

    # Define the path to the results file
    results_path = f'results/pruning'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model_name}.txt'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.local_rank}")
    print(f"Dataset Type: {args.data_type}")
    print(f"Model type: {args.model_name}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_path}/{filename}")
    print("-" * 30, "\n")

    datasets = {
        'ProGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/progan',
        'CycleGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/cyclegan',
        'BigGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/biggan',
        'StyleGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/stylegan',
        'GauGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/gaugan',
        'StarGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/stargan',
        'DeepFake': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/deepfake',
        'SITD': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/seeingdark',
        'SAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/san',
        'CRN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/crn',
        'IMLE': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/imle',
        'Guided': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/guided',
        'LDM_200': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200',
        'LDM_200_cfg': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200_cfg',
        'LDM_100': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_100',
        'Glide_100_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_27',
        'Glide_50_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_50_27',
        'Glide_100_10': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_10',
        'DALL-E': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/dalle',  
    }

    # datasets = {
    #     'ImageNet-mini': '/home/users/chandler_doloriel/scratch/Datasets/imagenet-mini/val',
    #     'Glide_100_10': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_10',
    #     'ProGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/progan',
    #     }

    # Initialize a counter
    dataset_count = len(datasets)
    total_ap = 0
    total_acc = 0
    total_auc = 0

    for dataset_name, dataset_path in datasets.items():
        # if dist.get_rank() == 0:
            # print(f"\nEvaluating {dataset_name}")
        avg_ap, avg_acc, auc, model = evaluate_model(
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
        total_ap += avg_ap
        total_acc += avg_acc
        total_auc += auc

        if dist.get_rank() == 0:
            # Write the results to the file
            with open(f'{results_path}/{filename}', 'a') as file:
                if file.tell() == 0: # Check if the file is empty
                    file.write("Selected Configuration:\n")
                    file.write("-" * 28 + "\n")
                    file.write(f"Cuda: {args.local_rank}\n")
                    file.write(f"Model type: {args.model_name}\n")
                    file.write(f"Ratio of mask: {ratio}\n")
                    file.write(f"Batch Size: {args.batch_size}\n")
                    file.write(f"Mask Type: {args.mask_type}\n")
                    file.write(f"Results saved to: {results_path}/{filename}\n")
                    file.write("-" * 28 + "\n\n")
                if 'progan' in dataset_path:
                    file.write(f"\nCheckpoint Path: {checkpoint_path}\n")
                    file.write("Dataset, Precision, Accuracy, AUC\n")
                    file.write("-" * 28)
                    file.write("\n")
                file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}, {auc:.3f}\n")

                # Decrement the counter
                dataset_count -= 1
                if dataset_count == 0:
                    # Compute averages
                    avg_total_ap = total_ap / len(datasets)
                    avg_total_acc = total_acc / len(datasets)
                    avg_total_auc = total_auc / len(datasets)
                    # Write the averages to the file
                    file.write("-" * 28 + "\n")
                    file.write(f"Average Precision: {avg_total_ap*100:.2f}%, Average Accuracy: {avg_total_acc*100:.2f}%, Average AUC: {avg_total_auc:.3f}\n")
                    file.write(f'Sparsity: {get_model_sparsity(model) * 100:.2f} (%)\n')
                    file.write(f"Remaining FLOPs: {get_model_flops(model) * 100:.2f} (%)\n")
                    file.write("\n")
        del model