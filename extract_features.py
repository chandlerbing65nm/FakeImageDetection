import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import clip
import pickle
import timm
import joblib
import h5py
import torch.nn as nn
import argparse

from dataset import WangEtAlDataset, CorviEtAlDataset
from wangetal_augment import ImageAugmentor
from utils import *


class CLIPFeatureExtractor:
    def __init__(self, model_name='ViT-B/16', mask_generator=None, use_masking=False, device="cpu"):  # ViT-L/14
        self.device = device
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()
        self.use_masking = use_masking
        self.mask_generator = mask_generator

    def extract_features(self, dataloader):
        real_embeddings = []
        fake_embeddings = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader):
                if self.use_masking:
                    imgs = self.mask_image(imgs)  # Apply mask if enabled
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                features = self.model.encode_image(imgs)
                features = features.cpu().numpy()

                for feature, label in zip(features, labels):
                    if label == 0:  # 'real'
                        real_embeddings.append(feature)
                    else:  # 'fake'
                        fake_embeddings.append(feature)

        return np.array(real_embeddings), np.array(fake_embeddings)

    def mask_image(self, imgs):
        # Apply mask to each image in the batch
        for i in range(len(imgs)):
            # Generate the mask on the device
            masked_img = self.mask_generator.transform(imgs[i].cpu())
            # Move the masked image back to the device
            imgs[i] = masked_img.to(self.device)
        return imgs

def create_transform(augmentor, mask_ratio=0.10):
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform

def extract_save_features(
    mask_generator_type, 
    clip_model,
    dataset_path, 
    save_path, 
    mask_ratio, 
    use_masking, 
    device):
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

    augmentor = ImageAugmentor(opt)
    transform = create_transform(augmentor)

    # Depending on the mask_generator_type create the appropriate mask generator
    if mask_generator_type == 'spectral':
        mask_generator = BalancedSpectralMaskGenerator(mask_ratio=mask_ratio, device=device)
    elif mask_generator_type == 'zoom':
        mask_generator = ZoomBlockGenerator(mask_ratio=mask_ratio, device=device)
    elif mask_generator_type == 'patch':
        mask_generator = PatchMaskGenerator(mask_ratio=mask_ratio, device=device)
    elif mask_generator_type == 'shiftedpatch':
        mask_generator = ShiftedPatchMaskGenerator(mask_ratio=mask_ratio, device=device)
    else:
        mask_generator = None
        # raise ValueError('Invalid mask_generator_type')

    # Define the dataset and dataloader
    dataset = WangEtAlDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define the feature extractor with the given mask generator and use_masking flag
    feature_extractor = CLIPFeatureExtractor(model_name=clip_model, mask_generator=mask_generator, use_masking=use_masking, device=device)

    # Extract the features
    real_embeddings, fake_embeddings = feature_extractor.extract_features(dataloader)

    # Save the embeddings
    with open(save_path, 'wb') as f:
        pickle.dump((real_embeddings, fake_embeddings), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features and save them")

    parser.add_argument('--mask_generator_type', default='nomask', 
                        choices=['zoom', 'patch', 'spectral', 'shiftedpatch', 'nomask'],
                        help='Type of mask generator')
    parser.add_argument('--clip_model', default='ViT-L/14', 
                        choices=['ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
                        help='Type of clip visual model')
    parser.add_argument('--dataset_path', default='../../Datasets/Wang_CVPR20/wang_et_al/training',
                        help='Path to the dataset')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        choices=['cuda:0', 'cpu'],
                        help='Computing device to use')
    parser.add_argument('--mask_ratio', type=int, default=50,
                        help='Ratio of mask to apply')

    args = parser.parse_args()
    clip_model = args.clip_model.lower().replace('/', '').replace('-', '')

    if args.mask_ratio != 0 and args.mask_ratio > 0 and args.mask_generator_type != 'nomask':
        mask_ratio = args.mask_ratio
        save_path = f'embeddings/masking/{clip_model}_{args.mask_generator_type}mask{mask_ratio}clip_embeddings.pkl'
        use_masking = True
    else:
        save_path = f'embeddings/{clip_model}_clip_embeddings.pkl'
        use_masking = False
        mask_ratio = 0

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Type of mask generator: {args.mask_generator_type}")
    print(f"Path to the dataset: {args.dataset_path}")
    print(f"CLIP model type: {args.clip_model}")
    print(f"Flag to use masking: {use_masking}")
    print(f"Ratio of mask to apply: {mask_ratio}")
    print(f"Embedding Path: {save_path}")
    print(f"Device: {args.device}")
    print("-" * 30, "\n")

    extract_save_features(
        args.mask_generator_type,
        args.clip_model, 
        args.dataset_path, 
        save_path, 
        mask_ratio/100, 
        use_masking, 
        args.device
        )