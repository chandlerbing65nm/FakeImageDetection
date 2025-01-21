#!/usr/bin/env python
import argparse
import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score

from utils import *
from augment import ImageAugmentor
from mask import *
from dataset import ForenSynths

from networks.resnet import resnet50 as my_resnet50

import torch.nn.utils.prune as prune
import copy



def apply_pruning(model, prune_type='localprune', conv2d_prune_amount=0.2):
    if prune_type == 'globalprune':
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=conv2d_prune_amount,
        )
    # elif prune_type == 'localprune':
    #     for module_name, module in model.named_modules():
    #         if isinstance(module, torch.nn.Conv2d):
    #             prune.l1_unstructured(
    #                 module,
    #                 name="weight",
    #                 amount=conv2d_prune_amount
    #             )
    elif prune_type == 'localprune':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Example of Ln-structured pruning, removing entire filters (dim=0)
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=conv2d_prune_amount,  # fraction of filters to prune
                    n=1,                        # can be 1 (L1 norm) or 2 (L2 norm)
                    dim=0                       # "0" prunes filters, "1" prunes channels in each filter
                )
    else:
        raise ValueError("Invalid prune_type. Choose 'localprune' or 'globalprune'.")

# -----------
# Minimal training loop for fine-tuning 5 epochs
# -----------
def finetune_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=5):
    """
    A *minimal* training loop for demonstration purposes.
    No early stopping here, trains for exactly `epochs` epochs.
    Adjust logging or metrics as desired.
    """
    model.train()
    for epoch in range(epochs):
        # Training pass
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation pass (optional, for monitoring)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Val Loss: {val_loss:.4f}")
        model.train()

def remove_pruning_masks(model):
    """
    Utility function to 'remove' the pruning reparametrization
    so that the final weights are no longer masked in a param+mask structure.
    """
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # This removes the mask and the reparametrization
            prune.remove(module, 'weight')

# Dynamically construct the filename based on args
def construct_save_filename(args):
    """Constructs a filename based on pruning parameters."""
    filename_parts = [args.model_name, args.prune_mode]  # Start with model name and prune mode

    # Common for both 'pruneonly' and 'prunefinetune'
    if args.prune_mode in ['pruneonly', 'prunefinetune']:
        filename_parts.append(f"{args.prune_type}")
        filename_parts.append(f"amount{int(args.prune_amount * 100)}")  # Convert to percentage

    # Add fine-tuning details only for 'prunefinetune'
    if args.prune_mode == 'prunefinetune':
        filename_parts.append(f"epochs{args.finetune_epochs}")
        filename_parts.append(f"lr{args.lr:.1e}")  # Scientific notation for clarity

        # Add mask-related details only for 'prunefinetune'
        if args.mask_type != 'nomask':
            filename_parts.append(f"{args.mask_type}")
            filename_parts.append(f"ratio{args.ratio}")

        # Add a tag for small training datasets only for 'prunefinetune'
        if args.smalltrain:
            filename_parts.append("smalltrain")

    # Combine all parts into a single filename
    filename = "_".join(filename_parts) + ".pth"
    return filename


# -----------
# Main function
# -----------
def main():
    parser = argparse.ArgumentParser(description="Model Pruning Script")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    # Distinguish between "pruneonly" and "prunefinetune"
    parser.add_argument('--prune_mode', type=str, required=True,
                        choices=['pruneonly', 'prunefinetune'],
                        help="Choose pruning only or pruning + fine-tuning")

    parser.add_argument('--prune_type', type=str, default='localprune',
                        choices=['localprune','globalprune'],
                        help='Local or global pruning')
    parser.add_argument('--prune_amount', type=float, default=0.2,
                        help='Fraction of weights to prune in Conv2d layers')

    # Checkpoint loading
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the .pth checkpoint to load model weights')
    
    # Output saving
    parser.add_argument('--save_folder', type=str, default='./pruned_models',
                        help='Folder to save pruned (or pruned+finetuned) model')
    parser.add_argument('--save_filename', type=str, default='pruned_model.pth',
                        help='Filename for the saved model')

    # Basic training hyperparams for fine-tuning
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=5, help='Number of epochs for fine-tuning')

    # Example: choose which model to load
    parser.add_argument('--model_name', type=str, default='RN50',
                        choices=['RN50'],
                        help='Which model to instantiate')
    
    # (Optional) Example dataset paths for fine-tuning
    parser.add_argument('--train_data_path', type=str, default='/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/training',
                        help='Path to training data folder')
    parser.add_argument('--val_data_path', type=str, default='/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/validation',
                        help='Path to validation data folder')

    parser.add_argument('--mask_type', default='fourier', choices=['fourier', 'cosine', 'wavelet', 'pixel', 'patch', 'translate', 'rotate', 'nomask'], help='Type of mask generator')
    parser.add_argument('--ratio', type=int, default=50, help='Masking/Augmentation ratio')
    parser.add_argument('--smalltrain', action='store_true', help='For small training set')

    args = parser.parse_args()

    # -----------
    # Init distributed
    # -----------
    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # -----------
    # Build Model
    # -----------
    if args.model_name == 'RN50':
        original_model = my_resnet50(pretrained=False)
        original_model.fc = nn.Linear(original_model.fc.in_features, 1)
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    original_model = original_model.to(device)
    original_model = DDP(original_model, find_unused_parameters=True)

    # -----------
    # Load Checkpoint
    # -----------
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"[Rank {dist.get_rank()}] Loaded checkpoint from {args.checkpoint_path}")

    # -----------
    # Apply Pruning
    # (masks remain active during subsequent steps)
    # -----------
    print(f"[Rank {dist.get_rank()}] Applying {args.prune_type} pruning at {args.prune_amount*100:.1f}%")
    model = copy.deepcopy(original_model)
    apply_pruning(model, prune_type=args.prune_type, conv2d_prune_amount=args.prune_amount)

    # -----------
    # If we only want to prune and skip fine-tuning
    # -----------
    if args.prune_mode == 'pruneonly':
        remove_pruning_masks(model)

        # Finally, save the pruned model
        if dist.get_rank() == 0:
            os.makedirs(args.save_folder, exist_ok=True)
            args.save_filename = construct_save_filename(args)
            save_path = os.path.join(args.save_folder, args.save_filename)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'prune_type': args.prune_type,
                    'prune_amount': args.prune_amount
                },
                save_path
            )
            print(f"[Rank 0] Saved pruned model (no finetuning) to {save_path}")

    # -----------
    # If we want to prune + finetune
    # -----------
    elif args.prune_mode == 'prunefinetune':
        # Build data loaders for your fine-tuning
        train_opt = {'rz_interp': ['bilinear'], 'loadSize': 256, 'blur_prob': 0.1, 'blur_sig': [0.0, 3.0], 'jpg_prob': 0.1, 'jpg_method': ['cv2', 'pil'], 'jpg_qual': [30, 100]}
        val_opt = {'rz_interp': ['bilinear'], 'loadSize': 256, 'blur_prob': 0.1, 'blur_sig': [(0.0 + 3.0) / 2], 'jpg_prob': 0.1, 'jpg_method': ['pil'], 'jpg_qual': [int((30 + 100) / 2)]}

        if ratio > 100.0 or ratio < 0.0:
            raise valueError(f"Invalid mask ratio {ratio}")
        else:
            # Create a MaskGenerator
            if args.mask_type in ['fourier', 'cosine', 'wavelet']:
                mask_generator = FrequencyMaskGenerator(ratio=args.ratio / 100, band=band, transform_type=args.mask_type)
            elif args.mask_type == 'pixel':
                mask_generator = PixelMaskGenerator(ratio=args.ratio / 100)
            elif args.mask_type == 'patch':
                mask_generator = PatchMaskGenerator(ratio=args.ratio / 100)
            elif args.mask_type == 'rotate':
                mask_generator = None
                args.ratio = (args.ratio / 100) * 180
            elif args.mask_type == 'translate':
                mask_generator = None
                args.ratio = args.ratio / 100
            elif args.mask_type == 'nomask':
                mask_generator = None
            else:
                raise ValueError(f"Unsupported mask type: {args.mask_type}")

        train_transform = train_augment(ImageAugmentor(train_opt), mask_generator, args)
        val_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)

        train_dataset = ForenSynths(args.train_data_path, transform=train_transform)
        val_dataset   = ForenSynths(args.val_data_path,   transform=val_transform)
        if args.smalltrain:
            subset_size = int(0.02 * len(train_dataset))
            subset_indices = random.sample(range(len(train_dataset)), subset_size)
            train_dataset = Subset(train_dataset, subset_indices)
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Setup training objects
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4) 

        # Fine-tune for a fixed number of epochs, with the pruning mask still in place
        finetune_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.finetune_epochs
        )

        # Only remove the masks after finishing fine-tuning
        remove_pruning_masks(model)

        # Save the pruned+finetuned model
        if dist.get_rank() == 0:
            os.makedirs(args.save_folder, exist_ok=True)
            # Dynamically generate save filename
            args.save_filename = construct_save_filename(args)
            save_path = os.path.join(args.save_folder, args.save_filename)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'prune_type': args.prune_type,
                    'prune_amount': args.prune_amount,
                    'finetuned': True
                },
                save_path
            )
            print(f"[Rank 0] Saved pruned+finetuned model to {save_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()