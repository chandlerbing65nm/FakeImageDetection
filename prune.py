
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import argparse
import torchvision.models as vis_models
from torchvision.datasets import CIFAR10, FashionMNIST
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
from networks.resnet_cifar10 import resnet50_cifar10
from networks.vgg_cifar10 import vgg19_bn_cifar10

from networks.clip_models import CLIPModel
import os
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

os.environ['LOCAL_RANK']

import sys
sys.path.append('./RD_PRUNE')
from RD_PRUNE.tools import *

def main(
    local_rank=0,
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    ratio=50,
    batch_size=64,
    model_name='rn50',
    band='all',
    mask_type=None,
    pretrained=False,
    args=None,
    ):

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')


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
        if mask_type == 'spectral':
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band)
        elif mask_type == 'pixel':
            mask_generator = PixelMaskGenerator(ratio=ratio)
        elif mask_type == 'patch':
            mask_generator = PatchMaskGenerator(ratio=ratio)
        else:
            mask_generator = None

    train_transform = train_augment(ImageAugmentor(train_opt), None, args)
    val_transform = val_augment(ImageAugmentor(val_opt), None, args)

    def select_binary_classes(dataset, classes):
        # Get the indices of samples belonging to the specified classes
        indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
        return Subset(dataset, indices)

    if args.dataset == 'ForenSynths':
        train_data = ForenSynths('/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/training', transform=train_transform)
        val_data = ForenSynths('/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/validation', transform=val_transform)
    elif args.dataset == 'LSUNbinary':
        train_data = ForenSynths('/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/binary', transform=train_transform)
        val_data = ForenSynths('/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/binary', transform=val_transform)
    elif args.dataset == 'CIFAR10':
        # train_transform = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        #     ]
        # )
        # val_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        #     ]
        # )
        train_data = torchvision.datasets.CIFAR10(root='/home/users/chandler_doloriel/scratch/Datasets/cifar10', train=True, download=False, transform=train_transform)
        val_data = torchvision.datasets.CIFAR10(root='/home/users/chandler_doloriel/scratch/Datasets/cifar10', train=False, download=False, transform=val_transform)

        # Filter the datasets to only include classes 0 and 1
        train_data = select_binary_classes(train_data, [0, 1])
        val_data = select_binary_classes(val_data, [0, 1])

    if args.smallset:
        subset_size = int(0.02 * len(train_data))
        subset_indices = random.sample(range(len(train_data)), subset_size)
        train_data = Subset(train_data, subset_indices)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.pruning_test_ft or args.pruning_test:
        if args.pruning_method == 'ours_nomask':
            mask_transform = val_augment(ImageAugmentor(val_opt), None, args)
        else:
            mask_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)
            
        # mask_data = ForenSynths('/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/validation', transform=mask_transform)
        mask_data = torchvision.datasets.CIFAR10(root='/home/users/chandler_doloriel/scratch/Datasets/cifar10', train=False, download=False, transform=mask_transform)
        mask_data = select_binary_classes(mask_data, [0, 1])
        subset_size = int(0.02 * len(mask_data))
        subset_indices = random.sample(range(len(mask_data)), subset_size)
        mask_data = Subset(mask_data, subset_indices)
        sampler = DistributedSampler(mask_data, shuffle=False, seed=seed)
        mask_loader = DataLoader(mask_data, batch_size=batch_size, sampler=sampler, num_workers=8)
    else:
        mask_loader = None
        

    # Creating and training the binary classifier
    if model_name == 'rn50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'rn50_mod':
        model = _resnet50(pretrained=pretrained, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name == 'clip_vitl14':
        clip_model_name = 'clip_vitl14'
        model = CLIPModel(clip_model_name, num_classes=1)
    elif model_name == 'clip_rn50':
        clip_model_name = 'rn50'
        model = CLIPModel(clip_model_name, num_classes=1)
    elif model_name == 'rn50_cifar10':
        model = resnet50_cifar10(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'vgg_cifar10':
        model = vgg19_bn_cifar10(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    container = model # for rd_prune method
    model = model.to(device)
    model = DistributedDataParallel(model)
    calib_model = model

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1, last_epoch=-1)

    if args.pretrained == False:
        checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path)

        if 'clip' in args.model_name and args.trainable_clip == False: # if the clip model is finetuned from the backbone
            model.module.fc.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

    if 'rd' in args.pruning_method:
        data_dir = '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/validation'
        # calib_loader = get_calib_deepfake_dali_loader(args, data_dir, batch_size=args.batch_size, calib_size=256)
        calib_loader = mask_loader
        rd_dict = {
            'container': container,
            'calib_loader': calib_loader
        }
    else:
        rd_dict = None
    pruned_model = iterative_pruning_finetuning(
        model, 
        calib_model,
        criterion, 
        optimizer, 
        scheduler,
        train_loader, 
        val_loader, 
        mask_loader,
        device,
        args.lr, 
        args=args,
        rd_dict=rd_dict
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
        default='rn50',
        type=str,
        choices=[
            'rn50', 'rn50_mod', 'clip_vitl14', 'clip_rn50', 'rn50_cifar10', 'vgg_cifar10'
            # 'ViT_base_patch16_224', 'ViT_base_patch32_224',
            # 'ViT_large_patch16_224', 'ViT_large_patch32_224'
        ],
        help='Type of model to use; includes ResNet'
        )
    parser.add_argument(
        '--trainable_clip', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For loading finetuned clip model'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=[
            'all', 'low', 'mid', 'high',
        ]
        )
    parser.add_argument(
        '--pretrained', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='if use ImageNet weights'
        )
    parser.add_argument(
        '--pruning_test_ft', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='if use adaptive layerwise pruning'
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
            'spectral', 
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
        '--pruning_ft', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For finetuning after pruning'
        )
    parser.add_argument(
        '--pruning_test', 
        type=str2bool,
        default=False,  # Optional: you can set a default value
        help='For test after pruning'
        )
    parser.add_argument(
        '--pruning_method', 
        default='ours',
        type=str,
        choices=[
            'ours', 'ours_lamp', 'ours_erk', 'ours_nomask', 'lamp_erk', 'rd', 'ours_rd'
        ],
        help='if use ours, ours_lamp, ours_erk'
    )
    parser.add_argument(
        '--calib_sparsity', 
        type=float, 
        default=0.2, 
        help='amount to prune'
        )
    parser.add_argument(
        '--desired_sparsity', 
        type=float, 
        default=0.2, 
        help='amount to prune'
        )
    parser.add_argument(
        '--linear_prune_amount', 
        type=float, 
        default=0.1, 
        help='amount to prune'
        )
    parser.add_argument(
        '--pruning_rounds', 
        type=int, 
        default=1, 
        help='pruning iteration'
        )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=44, 
        help='seed number'
        )
    parser.add_argument(
        '--checkpoint_path', 
        default='./checkpoints/mask_0/rn50ft.pth',
        type=str,
        )
    parser.add_argument(
        '--dataset',
        default='ForenSynths',
        help='Dataset to use for training and validation'
    )
    parser.add_argument(
        '--Beta', 
        type=float, 
        default=100, 
        help='hyperparameter of the sensitivity'
        )

    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--weight_rewind", action="store_true")
    parser.add_argument("--worst_case_curve", "-wcc", action="store_true")
    parser.add_argument("--synth_data", action="store_true")
    parser.add_argument("--singlelayer", action="store_true")
    parser.add_argument("--iter_start", type=int, default=1, help="start iteration for pruning (set >1 for resume)")
    parser.add_argument("--iter_end", type=int, default=11, help="end iteration for pruning")
    parser.add_argument("--maxsps", type=int, default=4)
    parser.add_argument("--remask_per_iter", type=int, default=4000)
    parser.add_argument("--prune_mode", "-pm", type=str, default="unstructured", choices=["unstructured", "structured"])
    parser.add_argument("--ranking", type=str, default="l1")


    args = parser.parse_args()
    model_name = args.model_name.lower().replace('/', '').replace('-', '')
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        ckpt_folder = f'./checkpoints/mask_{ratio}'
    else:
        ratio = 0
        args.band = 'None'
        ckpt_folder = f'./checkpoints/mask_{ratio}'


    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Seed: {args.seed}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}")
    print(f"Mask Band: {args.band}")
    print(f"Model Arch: {args.model_name}")
    print(f"ImageNet Weights: {args.pretrained}")
    if not args.pretrained: print(f"Checkpoint: {args.checkpoint_path}")

    print(f"\n")
    if args.pruning_test_ft: print(f"Pruning Method: {args.pruning_method}")
    print(f"Calibration: {args.pruning_test}")
    print(f"Calibration with Finetuning: {args.pruning_test_ft}")
    print(f"Calibration Sparsity: {args.calib_sparsity}")
    print(f"Desired Sparsity: {args.desired_sparsity}")
    print("-" * 30, "\n")

    main(
        local_rank=args.local_rank,
        num_epochs=5,
        ratio=ratio/100,
        batch_size=args.batch_size,
        model_name=args.model_name,
        band=args.band,
        mask_type=args.mask_type,
        pretrained=args.pretrained,
        args=args
    )


# How to run?
# python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse
