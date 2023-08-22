import torch
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
from wangetal_augment import ImageAugmentor
from utils import *

def test_augment(mask_generator):
    # Define the custom transform
    # masking_transform = MaskingTransform(mask_generator)

    transform = transforms.Compose([
        # transforms.Lambda(lambda img: mask_generator.transform(img)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform

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
        mask_generator = BalancedSpectralMaskGenerator(ratio=ratio)
    elif mask_type == 'zoom':
        mask_generator = ZoomBlockGenerator(ratio=ratio)
    elif mask_type == 'edge':
        mask_generator = EdgeAwareMaskGenerator(ratio=ratio)
    else:
        mask_generator = None

    test_transform = test_augment(mask_generator)

    if data_type == 'GenImage':
        test_dataset = GenImage(dataset_path, transform=test_transform)
    elif data_type == 'Wang_CVPR20' :
        test_dataset = Wang_CVPR20(dataset_path, transform=test_transform)
    elif data_type == 'Ojha_CVPR23' :
        test_dataset = OjhaCVPR23(dataset_path, transform=test_transform)
    else:
        raise ValueError("wrong dataset input")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_name == 'RN18':
        model = vis_models.resnet18(pretrained=False)
    elif model_name == 'RN34':
        model = vis_models.resnet34(pretrained=False)
    elif model_name == 'RN50':
        model = vis_models.resnet50(pretrained=False)
    elif model_name == 'RN101':
        model = vis_models.resnet101(pretrained=False)
    elif model_name == 'RN152':
        model = vis_models.resnet152(pretrained=False)
    elif model_name.startswith('ViT'):
        model_variant = model_name.split('_')[1] # Assuming the model name is like 'ViT_base_patch16_224'
        model = timm.create_model(model_variant, pretrained=False)
    else:
        raise ValueError(f"Model {model} not recognized!")
    
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(checkpoint_path))
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

    print(f'Average Precision: {ap}')
    print(f'Accuracy: {acc}')

    return ap, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN101', 'RN152',
            'ViT_base_patch16_224', 'ViT_base_patch32_224',
            'ViT_large_patch16_224', 'ViT_large_patch32_224'
        ],
        help='Type of model to use; includes ResNet and ViT variants'
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
        default="ForenSynths", 
        type=str, 
        choices=['GenImage', 'Wang_CVPR20', 'Ojha_CVPR23'], 
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

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}_{args.mask_type}mask_best.pth'
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}_best.pth'

    # Define the path to the results file
    results_path = f'results/{args.data_type.lower()}'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model_name}{finetune}_{args.mask_type}mask{ratio}.txt'

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
            'ProGAN': '../../Datasets/Wang_CVPR20/progan',
            'CycleGAN': '../../Datasets/Wang_CVPR20/cyclegan',
            'BigGAN': '../../Datasets/Wang_CVPR20/biggan',
            'StyleGAN': '../../Datasets/Wang_CVPR20/stylegan',
            'GauGAN': '../../Datasets/Wang_CVPR20/gaugan',
            'StarGAN': '../../Datasets/Wang_CVPR20/stargan',
            'DeepFake': '../../Datasets/Wang_CVPR20/deepfake',
            'SITD': '../../Datasets/Wang_CVPR20/seeingdark',
            'SAN': '../../Datasets/Wang_CVPR20/san',
            'CRN': '../../Datasets/Wang_CVPR20/crn',
            'IMLE': '../../Datasets/Wang_CVPR20/imle',
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

        avg_ap, avg_acc = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
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
                file.write("Dataset, Precision, Accuracy\n")
                file.write("-" * 28)
                file.write("\n")
            file.write(f"{dataset_name}, {avg_ap*100:.2f}, {avg_acc*100:.2f}\n")

        # Decrement the counter
        dataset_count -= 1
        if dataset_count == 0:
            with open(f'{results_path}/{filename}', 'a') as file:
                file.write("\n")