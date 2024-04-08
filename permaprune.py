

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
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'clip',
        ],
        help='Type of model to use; includes ResNet variants'
        )
    parser.add_argument(
        '--mask_type', 
        default='nomask', 
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
        default=0,
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
        '--folder_path', 
        default="./pruned_ckpts", 
        type=str, 
        help="folder where pruned models were saved"
        )
    parser.add_argument(
        '--filename', 
        default="prune_model_18", 
        type=str, 
        help="filename without the .pth ext"
        )
    parser.add_argument(
        '--pruned_model', 
        action='store_true', 
        help='check if it is pruned model'
        )
    parser.add_argument(
        '--per_class', 
        action='store_true', 
        help='if you want per class results'
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
    checkpoint_path = os.path.join(args.folder_path, args.filename + '.pth')

    # Define the path to the results file
    results_path = f'./results/{args.data_type.lower()}'
    os.makedirs(results_path, exist_ok=True)
    txtname = f'{args.filename}.txt'


    if args.data_type == 'Wang_CVPR20':
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
        }
    elif args.data_type == 'Ojha_CVPR23':
        datasets = {
            'Guided': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/guided',
            'LDM_200': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200',
            'LDM_200_cfg': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200_cfg',
            'LDM_100': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_100',
            'Glide_100_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_27',
            'Glide_50_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_50_27',
            'Glide_100_10': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_10',
            'DALL-E': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/dalle',       
        }
    else:
        raise ValueError("wrong dataset type")

    # Initialize a counter
    dataset_count = len(datasets)

    for dataset_name, dataset_path in datasets.items():
        if dist.get_rank() == 0:
            print(f"\nEvaluating {dataset_name}")

        # Adjust the call to evaluate_model to handle both return types
        results = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            args.ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
            args,
            args.per_class,
        )

        if dist.get_rank() == 0:
            with open(f'{results_path}/{txtname}', 'a') as file:
                if file.tell() == 0:  # Header for the file, write only once
                    file.write("Selected Configuration:\n")
                    file.write("-" * 28 + "\n")
                    file.write(f"Device: {args.local_rank}\n")
                    file.write(f"Dataset Type: {args.data_type}\n")
                    file.write(f"Model type: {args.model_name}\n")
                    file.write(f"Ratio of mask: {args.ratio}\n")
                    file.write(f"Batch Size: {args.batch_size}\n")
                    file.write(f"Mask Type: {args.mask_type}\n")
                    file.write(f"Checkpoint Type: {checkpoint_path}\n")
                    file.write(f"Results saved to: {results_path}/{txtname}\n")
                    file.write("-" * 28 + "\n\n")
                    file.write("Dataset, Metrics, Values\n")
                    file.write("-" * 28 + "\n")

                # Check if per_class_metrics was requested and the results are in a dictionary
                if args.per_class:
                    avg_ap, avg_acc, auc = results['overall']['average_precision'], results['overall']['accuracy'], results['overall']['auc']
                    file.write(f"{dataset_name}, Overall, AP: {avg_ap*100:.2f}, Acc: {avg_acc*100:.2f}, AUC: {auc:.3f}\n")
                    for class_label, metrics in results['per_class'].items():
                        file.write(f"{dataset_name}, {class_label}, Precision: {metrics['precision']*100:.2f}, Recall: {metrics['recall']*100:.2f}, F1score: {metrics['f1-score']*100:.2f}\n")
                else:
                    # If per_class_metrics was not requested, the results are a tuple
                    avg_ap, avg_acc, auc = results
                    file.write(f"{dataset_name}, Overall, AP: {avg_ap*100:.2f}, Acc: {avg_acc*100:.2f}, AUC: {auc:.3f}\n")

                # Decrement the counter and write a footer if it's the last dataset
                dataset_count -= 1
                if dataset_count == 0:
                    file.write("-" * 28 + "\n\n")

