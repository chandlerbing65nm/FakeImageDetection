
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import argparse
import wandb
import torchvision.models as vis_models
import re

from torchvision.models import (
    mobilenet_v2, 
    MobileNet_V2_Weights,
    swin_t,
    Swin_T_Weights,
    vgg11,
    VGG11_Weights,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

from dataset import ForenSynths
# from extract_features import *
from augment import ImageAugmentor
from mask import *
from earlystop import EarlyStopping
from utils import *
from networks.resnet import resnet50
from networks.resnet_npr import resnet50 as resnet50_npr
from networks.resnet_mod import resnet50 as _resnet50, ChannelLinear

from networks.clip_models import CLIPModel
import os
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

os.environ['WANDB_CONFIG_DIR'] = './wandb'
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CACHE_DIR'] = './wandb'

def main(
    local_rank=0,
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    ratio=50,
    batch_size=64,
    wandb_run_id=None,
    model_name='RN50',
    band='all',
    wandb_name=None,
    project_name=None,
    save_path=None,
    mask_type=None,
    pretrained=False,
    resume_train=None,
    early_stop=True,
    wandb_online=False,
    args=None,
    ):

    seed = 44
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3600))

    wandb_resume = "allow" if resume_train else None

    if dist.get_rank() == 0:
        status = "online" if wandb_online else "offline"
        wandb.init(id=wandb_run_id, resume=wandb_resume, project=project_name, name=wandb_name, mode=status)
        wandb.config.update(args, allow_val_change=True)

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
        if mask_type in ['fourier', 'cosine', 'wavelet']:
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band, transform_type=mask_type)
        elif mask_type == 'pixel':
            mask_generator = PixelMaskGenerator(ratio=ratio)
        elif mask_type == 'patch':
            mask_generator = PatchMaskGenerator(ratio=ratio)
        elif mask_type == 'rotate':
            mask_generator = None
            args.ratio = (args.ratio / 100) * 180
        elif mask_type == 'translate':
            mask_generator = None
            args.ratio = args.ratio / 100
        elif mask_type == 'nomask':
            mask_generator = None
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")

    train_transform = train_augment(ImageAugmentor(train_opt), mask_generator, args)
    val_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)


    # Creating training dataset from images
    train_data = ForenSynths('/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/training', transform=train_transform)
    if args.debug:
        subset_size = int(0.02 * len(train_data))
        subset_indices = random.sample(range(len(train_data)), subset_size)
        train_data = Subset(train_data, subset_indices)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    # Creating validation dataset from images
    val_data = ForenSynths('/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/validation', transform=val_transform)
    # val_sampler = DistributedSampler(val_data, shuffle=False)
    # val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)


    # Creating and training the binary classifier
    if model_name == 'RN50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'RN50_npr':
        model = resnet50_npr(pretrained=pretrained) # always don't use pretrained weight here as per author mentioned https://github.com/chuangchuangtan/NPR-DeepfakeDetection/issues/1
        model.fc = nn.Linear(model.fc1.in_features, 1)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=pretrained, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name == 'CLIP_vitl14':
        clip_model_name = 'ViT-L/14'
        model = CLIPModel(clip_model_name, num_classes=1)
    elif model_name == 'MNv2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == 'VGG11':
        model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    elif model_name == 'SWIN_t':
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, 1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    # model = DistributedDataParallel(model)
    model = DistributedDataParallel(model, find_unused_parameters=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4) 

    # Load checkpoint if resuming
    if resume_train:
        resume_prefix = f'{save_path}_last_ep' if resume_train == 'from_last' else f'{save_path}_best_ep'
        
        # Separate the directory and base filename
        folder_path = os.path.dirname(save_path)
        base_filename = os.path.basename(save_path)
        
        checkpoint_files = os.listdir(folder_path)
        if checkpoint_files:
            resume_prefix_base = f'{base_filename}_last_ep' if resume_train == 'from_last' else f'{base_filename}_best_ep'
            ep_numbers = [int(re.search(f'{re.escape(resume_prefix_base)}(\d+)', f).group(1)) for f in checkpoint_files if f.startswith(resume_prefix_base) and f.endswith('.pth')]
            max_ep = max(ep_numbers)
            checkpoint_path = f'{resume_prefix}{max_ep}.pth'
        else:
            raise ValueError("No matching checkpoint files found.")

        checkpoint = torch.load(checkpoint_path)
        if 'CLIP' in args.model_name:
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
            print(f"Best validation accuracy: {best_score}")
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Resume learning rate: {param_group['lr']}")
    else:
        best_score = None
        counter=0

    early_stopping = EarlyStopping(
        path=save_path, 
        patience=5, 
        verbose=True, 
        min_lr=args.lr/100,
        early_stopping_enabled=early_stop,
        best_score=best_score,
        counter=counter,
        args=args
        )

    resume_epoch = last_epoch + 1 if resume_train else 0

    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
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
        
    if dist.get_rank() == 0:
        wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Your model description here")

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs training')
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
        '--wandb_online', 
        action='store_true', 
        help='Run wandb in offline mode'
        )
    parser.add_argument(
        '--project_name', 
        type=str, 
        default="Masked-ResNet",
        help='wandb project name'
        )
    parser.add_argument(
        '--wandb_run_id', 
        type=str, 
        default=None,
        help='wandb run id'
        )
    parser.add_argument(
        '--resume_train', 
        default=None,
        type=str,
        choices=[
            'from_last', 'from_best'
        ],
        help='what epoch to resume training'
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
        '--early_stop', 
        action='store_true', 
        help='For early stopping'
        )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='For debugging'
        )
    parser.add_argument(
        '--clip_grad', 
        action='store_true', 
        help='For finetuning clip model'
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

    args = parser.parse_args()
    model_name = args.model_name.lower().replace('/', '').replace('-', '')
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        save_path = f'{ckpt_folder}/{model_name}{finetune}_{band}{args.mask_type}mask'
        wandb_name = f"mask_{ratio}_{model_name}{finetune}_{band}{args.mask_type}"
    else:
        ratio = 0
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        save_path = f'{ckpt_folder}/{model_name}{finetune}'
        wandb_name = f"mask_{ratio}_{model_name}{finetune}"

    num_epochs = 100 if args.early_stop else args.num_epochs

    # # Retrieve resume path and epoch
    # resume_train = f"{save_path}_epoch{args.resume_epoch}.pth" if args.resume_epoch > 0 else None
    
    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Early Stopping: {args.early_stop}")
    print(f"Mask Generator Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"WandB run ID: {args.wandb_run_id}")
    print(f"WandB Project Name: {args.project_name}")
    print(f"WandB Instance Name: {wandb_name}")
    print(f"WandB Online: {args.wandb_online}")
    print(f"model type: {args.model_name}")
    print(f"Save path: {save_path}.pth")
    print(f"Resume training: {args.resume_train}")
    print("-" * 30, "\n")

    main(
        local_rank=args.local_rank,
        num_epochs=num_epochs,
        ratio=ratio/100,
        batch_size=args.batch_size,
        wandb_run_id=args.wandb_run_id,
        model_name=args.model_name,
        band=args.band,
        wandb_name=wandb_name,
        project_name=args.project_name,
        save_path=save_path, 
        mask_type=args.mask_type,
        pretrained=args.pretrained,
        resume_train=args.resume_train,
        early_stop=args.early_stop,
        wandb_online=args.wandb_online,
        args=args
    )


# How to run?
# python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse
