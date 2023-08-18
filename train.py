
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from clip import load
import numpy as np
import pickle
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import clip
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse
import wandb
from torchvision.models import resnet50, resnet101
import multiprocessing

import torch.distributed as dist
from tqdm.auto import tqdm as tqdm_auto
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from dataset import WangEtAlDataset, CorviEtAlDataset
# from extract_features import *
from wangetal_augment import ImageAugmentor
from utils import *

class EarlyStopping:
    def __init__(
        self, 
        path, 
        patience=7, 
        verbose=False, 
        delta=0, min_lr=1e-6, 
        factor=0.1, 
        early_stopping_enabled=True, 
        num_epochs=25
        ):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_lr = min_lr
        self.factor = factor
        self.path = path
        self.early_stopping_enabled = early_stopping_enabled
        self.last_epochs = []
        self.num_epochs = num_epochs

    def __call__(self, val_loss, model, optimizer, epoch):

        score = -val_loss

        if self.early_stopping_enabled:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if dist.get_rank() == 0:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] > self.min_lr:
                            if dist.get_rank() == 0:
                                print(f'Reducing learning rate from {param_group["lr"]} to {param_group["lr"] * self.factor}')
                            param_group['lr'] *= self.factor
                            self.counter = 0  # reset the counter
                        else:
                            self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch)
                self.counter = 0
        else:
            self.last_epochs.append((val_loss, model.state_dict()))
            if len(self.last_epochs) > 1:
                self.last_epochs.pop(0)  # remove the oldest model if we have more than 3
            if epoch == self.num_epochs-1:  # if it's the last epoch
                for i, (val_loss, state_dict) in enumerate(self.last_epochs):
                    if dist.get_rank() == 0:
                        torch.save(state_dict, f"{self.path}_epoch{epoch-i}" + '.pth')
        
    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose and epoch % 1 == 0:
            if dist.get_rank() == 0:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), self.path + '.pth')  # change here to use self.path
        self.val_loss_min = val_loss


def train_model(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    val_loader, 
    num_epochs=25, 
    save_path='./', 
    early_stopping_enabled=True,
    device='cpu'
    ):

    early_stopping = EarlyStopping(
        path=save_path, 
        patience=5, 
        verbose=True, 
        early_stopping_enabled=early_stopping_enabled,
        num_epochs=num_epochs,
        )

    for epoch in range(num_epochs):
        if epoch % 1 == 0:  # Only print every 20 epochs
            if dist.get_rank() == 0:
                print('\n')
                print(f'Epoch {epoch+1}/{num_epochs}')
                print('-' * 10)

        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            y_true, y_pred = [], []

            disable_tqdm = dist.get_rank() != 0
            data_loader_with_tqdm = tqdm(data_loader, f"{phase}", disable=disable_tqdm)

            for inputs, labels in data_loader_with_tqdm: #tqdm(data_loader, f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs).view(-1).unsqueeze(1)
                    loss = criterion(outputs.squeeze(1), labels)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(data_loader.dataset)
            
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)
            
            if epoch % 1 == 0:  # Only print every 20 epochs
                if dist.get_rank() == 0:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

            # Early stopping
            if phase == 'Validation':
                if dist.get_rank() == 0:
                    wandb.log({"Validation Loss": epoch_loss, "Validation Acc": acc, "Validation AP": ap})
                early_stopping(epoch_loss, model, optimizer, epoch)
                if early_stopping.early_stop:
                    if dist.get_rank() == 0:
                        print("Early stopping")
                    return model
            else:
                if dist.get_rank() == 0:
                    wandb.log({"Training Loss": epoch_loss, "Training Acc": acc, "Training AP": ap})
        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model

def train_augment(augmentor, mask_generator):
    # Define the custom transform
    masking_transform = MaskingTransform(mask_generator)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        masking_transform,
    ])
    return transform

def val_augment(mask_generator):
    # Define the custom transform
    masking_transform = MaskingTransform(mask_generator)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        masking_transform, # Add the custom masking transform here
    ])
    return transform

def main(
    local_rank=0,
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    ratio=50,
    batch_size=64,
    resnet_model='ViT-L/14',
    wandb_name=None,
    project_name=None,
    save_path=None,
    mask_type=None,
    early_stop=True,
    wandb_online=False,
    ):

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')

    if dist.get_rank() == 0:
        status = "online" if wandb_online else "offline"
        wandb.init(project=project_name, name=wandb_name, mode=status)
        wandb.config.update(args)  # Log all hyperparameters

    # Set options for augmentation
    opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [0.5],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['cv2'],
        'jpg_qual': [75]
    }

    # Depending on the mask_type, create the appropriate mask generator
    if mask_type == 'spectral':
        mask_generator = BalancedSpectralMaskGenerator(ratio=ratio, device=device)
    elif mask_type == 'zoom':
        mask_generator = ZoomBlockGenerator(ratio=ratio, device=device)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio, device=device)
    elif mask_type == 'shiftedpatch':
        mask_generator = ShiftedPatchMaskGenerator(ratio=ratio, device=device)
    elif mask_type == 'invblock':
        mask_generator = InvBlockMaskGenerator(ratio=ratio, device=device)
    elif mask_type == 'edge':
        mask_generator = EdgeAwareMaskGenerator(ratio=ratio)
    elif mask_type == 'highfreq':
        mask_generator = HighFrequencyMaskGenerator()
    else:
        mask_generator = None

    augmentor = ImageAugmentor(opt)
    train_transform = train_augment(augmentor, mask_generator)
    val_transform = val_augment(mask_generator)

    # # Defining a subset of the training dataset (e.g., using only the first 100 samples for debugging)
    # debug_size = 128*10
    # train_data = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/training', transform=train_transform)
    # train_data = Subset(train_data, range(debug_size))

    # Creating training dataset from images
    train_data = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/training', transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    # Creating validation dataset from images
    val_data = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/validation', transform=val_transform)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    # Creating and training the binary classifier
    if resnet_model == 'RN50':
        model = resnet50(pretrained=False)
    elif resnet_model == 'RN101':
        model = resnet101(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    model = DistributedDataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4) 


    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        save_path=save_path,
        early_stopping_enabled=early_stop,
        device=device,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Your model description here")

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs training')
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
        '--resnet_model', 
        default='RN50', 
        choices=['RN50', 'RN101'],
        help='Type of resnet_model visual model'
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
    resnet_model = args.resnet_model.lower().replace('/', '').replace('-', '')
    
    if args.mask_type != 'nomask':
        ratio = args.ratio
        save_path = f'checkpoints/mask_{ratio}/{resnet_model}_{args.mask_type}mask_best'
    else:
        ratio = 0
        save_path = f'checkpoints/mask_{ratio}/{resnet_model}_best'

    num_epochs = 300 if args.early_stop else args.num_epochs
    wandb_name = f"mask_{ratio}_{resnet_model}_{args.mask_type}"

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Early Stopping: {args.early_stop}")
    print(f"Mask Generator Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"WandB Project Name: {args.project_name}")
    print(f"WandB Instance Name: {wandb_name}")
    print(f"WandB Online: {args.wandb_online}")
    print(f"CLIP model type: {args.resnet_model}")
    print(f"Save path: {save_path}.pth")
    print(f"Device: cuda:{args.local_rank}")
    print("-" * 30, "\n")

    main(
        local_rank=args.local_rank,
        num_epochs=num_epochs,
        ratio=ratio/100,
        batch_size=args.batch_size,
        resnet_model=args.resnet_model,
        wandb_name=wandb_name,
        project_name=args.project_name,
        save_path=save_path, 
        mask_type=args.mask_type,
        early_stop=args.early_stop,
        wandb_online=args.wandb_online,
    )


# How to run?
# python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse