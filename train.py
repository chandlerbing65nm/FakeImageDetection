
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import argparse
import wandb
import torchvision
import re

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from dataset import ForenSynths
# from extract_features import *
from augment import ImageAugmentor
from mask import *
from earlystop import EarlyStopping
from utils import *
from networks.resnet import resnet50
from networks.resnet_mod import resnet50 as _resnet50, ChannelLinear

from networks.clip_models import CLIPModel
import os

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'


def main(
    local_rank=0,
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    ratio=50,
    batch_size=64,
    model_name='RN50',
    band='all',
    save_path=None,
    mask_type=None,
    pretrained=False,
    resume_pth=None,
    early_stop=True,
    args=None,
    ):

    # Set options for augmentation
    train_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [0.0, 3.0],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['cv2', 'pil'],
        'jpg_qual': [30, 100]
    }

    val_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [(0.0 + 3.0) / 2],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['pil'],
        'jpg_qual': [int((30 + 100) / 2)]
    }

    if ratio > 1.0 or ratio < 0.0:
        raise valueError(f"Invalid mask ratio {ratio}")
    else:
        # Create a MaskGenerator
        if mask_type == 'fourier':
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band, transform_type=mask_type)
        if mask_type == 'wavelet':
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band, transform_type=mask_type)
        if mask_type == 'cosine':
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band, transform_type=mask_type)
        elif mask_type == 'pixel':
            mask_generator = PixelMaskGenerator(ratio=ratio)
        elif mask_type == 'patch':
            mask_generator = PatchMaskGenerator(ratio=ratio)
        else:
            mask_generator = None

    train_transform = train_augment(ImageAugmentor(train_opt), mask_generator, args)
    val_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)

    # Creating training dataset from images
    train_data = ForenSynths(
        '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/training', 
        transform=train_transform, 
        )
    if args.smallset:
        subset_size = int(0.2 * len(train_data))
        subset_indices = random.sample(range(len(train_data)), subset_size)
        train_data = Subset(train_data, subset_indices)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    # Creating validation dataset from images
    val_data = ForenSynths(
        '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/validation', 
        transform=val_transform, 
        )
    val_sampler = DistributedSampler(val_data, shuffle=False, seed=seed)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    # Creating and training the binary classifier
    if model_name == 'RN50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=pretrained, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('clip'):
        clip_model_name = 'ViT-L/14'
        model = CLIPModel(clip_model_name, num_classes=1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    model = DistributedDataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1, last_epoch=-1) # not used

    # Load checkpoint if resuming
    if resume_pth is not None:
        checkpoint_path = resume_pth
        checkpoint = torch.load(checkpoint_path)

        if args.model_name == 'clip':
            model.module.fc.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Extract val_accuracy, counter and epoch from the checkpoint
        counter = checkpoint['counter']
        last_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score'] # val_accuracy or best_score

        if dist.get_rank() == 0:
            print(f"\nResuming training from epoch {last_epoch} using {checkpoint_path}")
            print(f"Resumed validation accuracy: {best_score}")
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Resume learning rate: {param_group['lr']}")
    # Load checkpoint if resuming
    elif args.finetune_pth is not None:
        checkpoint_path = args.finetune_pth
        checkpoint = torch.load(checkpoint_path)

        if args.model_name == 'clip':
            model.module.fc.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        if dist.get_rank() == 0:
            print(f"\nFinetune training using {checkpoint_path}")
        
        best_score = None
        counter=0
    else:
        best_score = None
        counter=0

    early_stopping = EarlyStopping(
        path=save_path, 
        patience=5, 
        verbose=True, 
        min_lr=(args.lr)**2,
        early_stopping_enabled=early_stop,
        best_score=best_score,
        counter=counter,
        args=args
        )

    resume_epoch = last_epoch + 1 if resume_pth else 0

    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler,
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        resume_epoch=resume_epoch,
        save_path=save_path,
        early_stopping=early_stopping,
        device=device,
        sampler=train_sampler,
        args=args,
        )
        

if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Your model description here")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs training')
    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'clip',
            # 'ViT_base_patch16_224', 'ViT_base_patch32_224',
            # 'ViT_large_patch16_224', 'ViT_large_patch32_224'
        ],
        help='Type of model to use; includes ResNet'
        )
    parser.add_argument(
        '--resume_pth', 
        default=None,
        type=str,
        help='what epoch to resume training'
        )
    parser.add_argument(
        '--finetune_pth', 
        default=None,
        type=str,
        help='finetune training'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        # choices=[
        #     'all', 'low', 'mid', 'high',
        # ]
        )
    parser.add_argument(
        '--pretrained', 
        type=str2bool,
        default=False,  # Optional: you can set a default value 
        help='For pretraining'
        )
    parser.add_argument(
        '--early_stop', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For early stopping'
        )
    parser.add_argument(
        '--smallset', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For using small subset of training set'
        )
    parser.add_argument(
        '--mask_type', 
        default='spectral', 
        choices=[
            'pixel', 
            'fourier',
            'wavelet',
            'cosine', 
            'patch',
            'nomask'], 
        help='Type of mask generator'
        )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Batch Size'
        )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=50, 
        help='Masking ratio'
        )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0001, 
        help='learning rate'
        )
    parser.add_argument(
        '--trainable_clip', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For training the clip model'
        )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=44, 
        help='seed number'
        )
    parser.add_argument(
        '--pruning_test_ft', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='if use adaptive layerwise pruning'
        )
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')

    model_name = args.model_name.lower().replace('/', '').replace('-', '')
    finetune = 'ft' if args.pretrained else ''
    if 'prog' in args.band:
        band = 'prog_' if args.band == 'all+prog' else args.band
    else:
        band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        # translearn_suffix = 'translearn' if args.finetune_pth is not None else ''
        save_path = f'{ckpt_folder}/{model_name}{finetune}_{band}{args.mask_type}mask'
    else:
        ratio = 0
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        # translearn_suffix = 'translearn' if args.finetune_pth is not None else ''
        save_path = f'{ckpt_folder}/{model_name}{finetune}'

    num_epochs = 100 if args.early_stop else args.num_epochs

    # # Retrieve resume path and epoch
    # resume_pth = f"{save_path}_epoch{args.resume_epoch}.pth" if args.resume_epoch > 0 else None
    
    torch.distributed.barrier()
    
    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Seed: {args.seed}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Early Stopping: {args.early_stop}")
    print(f"Mask Generator Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model Arch: {args.model_name}")
    print(f"Save path: {save_path}.pth")
    print(f"Resume training path: {args.resume_pth}")
    print(f"Finetune path: {args.finetune_pth}")
    print("-" * 30, "\n")
    
    torch.distributed.barrier()

    main(
        local_rank=args.local_rank,
        num_epochs=num_epochs,
        ratio=ratio/100,
        batch_size=args.batch_size,
        model_name=args.model_name,
        band=args.band,
        save_path=save_path, 
        mask_type=args.mask_type,
        pretrained=args.pretrained,
        resume_pth=args.resume_pth,
        early_stop=args.early_stop,
        args=args
    )


# How to run?
# python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse
