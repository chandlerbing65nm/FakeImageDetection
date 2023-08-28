
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
import torchvision.models as vis_models

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

import os
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

os.environ['WANDB_CONFIG_DIR'] = '/home/timm/chandler/Experiments/FakeDetection/wandb'
os.environ['WANDB_DIR'] = '/home/timm/chandler/Experiments/FakeDetection/wandb'
os.environ['WANDB_CACHE_DIR'] = '/home/timm/chandler/Experiments/FakeDetection/wandb'

def main(
    local_rank=0,
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    ratio=50,
    batch_size=64,
    wandb_run_id=None,
    model_name='RN50',
    wandb_name=None,
    project_name=None,
    save_path=None,
    mask_type=None,
    pretrained=False,
    resume_train=False,
    early_stop=True,
    wandb_online=False,
    ):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')

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

    # Depending on the mask_type, create the appropriate mask generator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    elif mask_type == 'zoom':
        mask_generator = ZoomBlockGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    elif mask_type == 'shiftedpatch':
        mask_generator = ShiftedPatchMaskGenerator(ratio=ratio)
    elif mask_type == 'invblock':
        mask_generator = InvBlockMaskGenerator(ratio=ratio)
    elif mask_type == 'edge':
        mask_generator = EdgeAwareMaskGenerator(ratio=ratio)
    elif mask_type == 'highfreq':
        mask_generator = HighFrequencyMaskGenerator()
    else:
        mask_generator = None

    train_transform = train_augment(ImageAugmentor(train_opt), mask_generator)
    val_transform = val_augment(ImageAugmentor(val_opt), mask_generator)

    # Creating training dataset from images
    train_data = ForenSynths('../../Datasets/Wang_CVPR2020/training', transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    # Creating validation dataset from images
    val_data = ForenSynths('../../Datasets/Wang_CVPR2020/validation', transform=val_transform)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    # Creating and training the binary classifier
    if model_name == 'RN50':
        # model = vis_models.resnet50(pretrained=pretrained)
        # model.fc = nn.Linear(model.fc.in_features, 1)
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=pretrained, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    model = DistributedDataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4) 

    # Load checkpoint if resuming
    if resume_train:
        checkpoint = torch.load(f'{save_path}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Extract val_accuracy, counter and epoch from the checkpoint
        counter = checkpoint['counter']
        last_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score'] # val_accuracy

        if dist.get_rank() == 0:
            print(f"\nResuming training from epoch {last_epoch} using {save_path}.pth")
            print(f"Resumed validation accuracy: {best_score}")
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Learning rate for param_group {i}: {param_group['lr']}")
    else:
        best_score = None
        counter=0

    early_stopping = EarlyStopping(
        path=save_path, 
        patience=5, 
        verbose=True, 
        early_stopping_enabled=early_stop,
        best_score=best_score,
        counter=counter
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
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Your model description here")

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs training')
    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'RN101', 'RN152',
            'ViT_base_patch16_224', 'ViT_base_patch32_224',
            'ViT_large_patch16_224', 'ViT_large_patch32_224'
        ],
        help='Type of model to use; includes ResNet and ViT variants'
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
        action='store_true', 
        help='Run wandb in offline mode'
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
        '--mask_type', 
        default='zoom', 
        choices=[
            'zoom', 
            'patch', 
            'spectral', 
            'shiftedpatch', 
            'invblock', 
            'edge',
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

    args = parser.parse_args()
    model_name = args.model_name.lower().replace('/', '').replace('-', '')
    finetune = 'ft' if args.pretrained else ''

    if args.mask_type != 'nomask':
        ratio = args.ratio
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        save_path = f'{ckpt_folder}/{model_name}{finetune}_{args.mask_type}mask'
    else:
        ratio = 0
        ckpt_folder = f'./checkpoints/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        save_path = f'{ckpt_folder}/{model_name}{finetune}'

    num_epochs = 100 if args.early_stop else args.num_epochs
    wandb_name = f"mask_{ratio}_{model_name}{finetune}_{args.mask_type}"

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
    print(f"Device: cuda:{args.local_rank}")
    print("-" * 30, "\n")

    main(
        local_rank=args.local_rank,
        num_epochs=num_epochs,
        ratio=ratio/100,
        batch_size=args.batch_size,
        wandb_run_id=args.wandb_run_id,
        model_name=args.model_name,
        wandb_name=wandb_name,
        project_name=args.project_name,
        save_path=save_path, 
        mask_type=args.mask_type,
        pretrained=args.pretrained,
        resume_train=args.resume_train,
        early_stop=args.early_stop,
        wandb_online=args.wandb_online,
    )


# How to run?
# python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse