
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import torch.distributed as dist  # If distributed training is being used
import wandb  # If Weights & Biases is being used for logging

from torchvision import transforms
from torch.utils.data import DataLoader
import timm 

import torchvision.models as vis_models

from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *
from networks.resnet import resnet50
from networks.resnet_mod import resnet50 as _resnet50, ChannelLinear


def train_augment(augmentor, mask_generator=None):
    # Initialize an empty list to store transforms
    transform_list = []
    if mask_generator is not None:
        transform_list.append(transforms.Lambda(lambda img: mask_generator.transform(img)))  
    transform_list.extend([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Create the composed transform
    transform = transforms.Compose(transform_list)
    return transform

def val_augment(augmentor, mask_generator=None):
    # Initialize an empty list to store transforms
    transform_list = []
    if mask_generator is not None:
        transform_list.append(transforms.Lambda(lambda img: mask_generator.transform(img))) 
    transform_list.extend([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Create the composed transform
    transform = transforms.Compose(transform_list)
    return transform
    
def test_augment(augmentor, mask_generator=None):
    # Define the custom transform
    # masking_transform = MaskingTransform(mask_generator)

    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform


def train_model(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    val_loader, 
    num_epochs=25, 
    resume_epoch=0,
    save_path='./', 
    early_stopping=None,
    device='cpu'
    ):

    for epoch in range(resume_epoch, num_epochs):
        if epoch % 1 == 0:  # Only print every 20 epochs
            if dist.get_rank() == 0:
                print('\n')
                print(f'Epoch {epoch}/{num_epochs}')
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
                    wandb.log({"Validation Loss": epoch_loss, "Validation Acc": acc, "Validation AP": ap}, step=epoch)
                early_stopping(acc, model, optimizer, epoch)  # Pass the accuracy instead of loss
                if early_stopping.early_stop:
                    if dist.get_rank() == 0:
                        print("Early stopping")
                    return model
            else:
                if dist.get_rank() == 0:
                    wandb.log({"Training Loss": epoch_loss, "Training Acc": acc, "Training AP": ap}, step=epoch)

        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model


def evaluate_model(
    model_name,
    data_type,
    mask_type, 
    ratio,
    dataset_path, 
    batch_size,
    checkpoint_path, 
    device
    ):

    # Depending on the mask_type, create the appropriate mask generator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    else:
        mask_generator = None


    test_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,  # Set your value
        'blur_sig': [(0.0 + 3.0) / 2],
        'jpg_prob': 0.1,  # Set your value
        'jpg_method': ['pil'],
        'jpg_qual': [int((30 + 100) / 2)]
    }

    test_transform = test_augment(ImageAugmentor(test_opt), mask_generator)

    if data_type == 'GenImage':
        test_dataset = GenImage(dataset_path, transform=test_transform)
    elif data_type == 'Wang_CVPR20' :
        test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    elif data_type == 'Ojha_CVPR23' :
        test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    else:
        raise ValueError("wrong dataset input")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    if model_name == 'RN50':
        # model = vis_models.resnet50(pretrained=pretrained)
        # model.fc = nn.Linear(model.fc.in_features, 1)
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'RN50_mod':
        model = _resnet50(pretrained=False, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    model.eval() 

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, "Accessing test dataloader"):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs).view(-1).unsqueeze(1)
            y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print(f'Average Precision: {ap}')
    print(f'Accuracy: {acc}')
    print(f'ROC AUC Score: {auc}')

    return ap, acc, auc

