import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import argparse
import random

from torchvision.models import resnet50, resnet101

from dataset import *
from wangetal_augment import ImageAugmentor
from utils import *

def evaluate_model(
    resnet_model,
    data_type, 
    dataset_path, 
    batch_size,
    checkpoint_path, 
    device
    ):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if data_type == 'GenImage':
        test_dataset = GenImage(dataset_path, transform=transform)
    elif data_type == 'ForenSynths' :
        test_dataset = ForenSynths(dataset_path, transform=transform)
    else:
        raise ValueError("wrong dataset input")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if resnet_model == 'RN50':
        model = resnet50(pretrained=False)
    elif resnet_model == 'RN101':
        model = resnet101(pretrained=False)
    
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path))
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
        '--resnet_model',
        default='RN50',
        type=str,
        choices=['RN50', 'RN101'],
        help='Type of ResNet model'
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
        choices=['GenImage', 'ForenSynths'], 
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
    model = args.resnet_model.lower()

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model}_{args.mask_type}mask_best.pth'
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model}_best.pth'

    # Define the path to the results file
    results_path = f'results/{args.data_type.lower()}/'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model}_{args.mask_type}mask{ratio}.txt'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.device}")
    print(f"Dataset Type: {args.data_type}")
    print(f"ResNet model type: {args.resnet_model}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_path}/{filename}")
    print("-" * 30, "\n")

    if args.data_type == 'ForenSynths':
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
    elif args.data_type == 'GenImage':
        datasets = {
            'VQDM': '../../Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val',
            'Glide': '../../Datasets/GenImage/imagenet_glide/imagenet_glide/val',
        }
    else:
        raise ValueError("wrong dataset type")

    for dataset_name, dataset_path in datasets.items():
        print(f"\nEvaluating {dataset_name}")

        avg_ap, avg_acc = evaluate_model(
            args.resnet_model,
            args.data_type,
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
                file.write(f"ResNet model type: {args.resnet_model}\n")
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