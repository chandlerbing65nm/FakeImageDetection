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
    def __init__(
        self, 
        model_name='ViT-L/14', 
        mask_generator=None, 
        use_masking=False, 
        full_clip=False, 
        device="cpu"
        ):

        self.device = device
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()
        self.use_masking = use_masking
        self.full_clip = full_clip
        self.mask_generator = mask_generator
        self.text_inputs = [
            "A visual depiction that may either represent an authentic capture of reality or a digitally manipulated construct, reflecting varying degrees of truthfulness or artificiality."
            ]

    def extract_features(self, dataloader):
        real_embeddings = []
        fake_embeddings = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader):
                if self.use_masking:
                    imgs = self.mask_image(imgs)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                if self.full_clip:
                    # text_inputs = [self.text_inputs[label] for label in labels]
                    text_inputs = clip.tokenize(self.text_inputs).to(self.device)
                    image_features = self.model.encode_image(imgs)
                    text_features = self.model.encode_text(text_inputs)

                    # Normalize the features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Compute the similarity as the dot product of normalized features
                    features = (image_features * text_features)
                else:
                    features = self.model.encode_image(imgs)

                features = features.cpu().numpy()

                for feature, label in zip(features, labels):
                    if label == 0:  # 'real'
                        real_embeddings.append(feature)
                    else:  # 'fake'
                        fake_embeddings.append(feature)

        return np.array(real_embeddings), np.array(fake_embeddings)

    def mask_image(self, imgs):
        for i in range(len(imgs)):
            masked_img = self.mask_generator.transform(imgs[i].cpu())
            imgs[i] = masked_img.to(self.device)
        return imgs

def create_transform(augmentor):
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
    mask_type, 
    clip_model,
    dataset_path, 
    save_path, 
    full_clip,
    ratio, 
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

    # Depending on the mask_type create the appropriate mask generator
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
        mask_generator = EdgeAwareMaskGenerator(ratio=ratio, device=device)
    elif mask_type == 'highfreq':
        mask_generator = HighFrequencyMaskGenerator(device=device)
    else:
        mask_generator = None
        # raise ValueError('Invalid mask_type')

    # Define the dataset and dataloader
    dataset = WangEtAlDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define the feature extractor with the given mask generator and use_masking flag
    feature_extractor = CLIPFeatureExtractor(
        model_name=clip_model, 
        mask_generator=mask_generator, 
        use_masking=use_masking,
        full_clip=full_clip, 
        device=device)

    # Extract the features
    real_embeddings, fake_embeddings = feature_extractor.extract_features(dataloader)

    # Save the embeddings
    with open(save_path, 'wb') as f:
        pickle.dump((real_embeddings, fake_embeddings), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features and save them")

    parser.add_argument(
        '--clip_model', 
        default='ViT-B/16', 
        choices=['ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
        help='Type of clip visual model'
        )
    parser.add_argument(
        '--dataset_path', 
        default='../../Datasets/Wang_CVPR20/wang_et_al/training',
        help='Path to the dataset'
        )
    parser.add_argument(
        '--full_clip', 
        action='store_true', 
        help='Use CLIP text encoder'
        )
    parser.add_argument(
        '--mask_type', 
        default='nomask', 
        choices=[
            'zoom', 
            'patch', 
            'spectral', 
            'shiftedpatch', 
            'invblock', 
            'edge',
            'highfreq',
            'nomask'
            ],
        help='Type of mask generator'
        )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=50,help='Ratio of mask to apply'
        )
    parser.add_argument(
        '--device', 
        default='cuda:0' if torch.cuda.is_available() else 'cpu', 
        help='Computing device to use'
        )

    args = parser.parse_args()
    clip_model = args.clip_model.lower().replace('/', '').replace('-', '')
    full_clip = '' if args.full_clip is False else 'full'

    if args.mask_type != 'nomask':
        ratio = args.ratio
        save_path = f'embeddings/masking/{clip_model}_{args.mask_type}mask{ratio}{full_clip}clip_embeddings.pkl'
        use_masking = True
    else:
        save_path = f'embeddings/{clip_model}_{full_clip}clip_embeddings.pkl'
        use_masking = False
        ratio = 0

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Type of mask generator: {args.mask_type}")
    print(f"Path to the dataset: {args.dataset_path}")
    print(f"CLIP model type: {args.clip_model}")
    print(f"Use text encoder: {args.full_clip}")
    print(f"Flag to use masking: {use_masking}")
    print(f"Ratio to apply operation: {ratio}")
    print(f"Embedding Path: {save_path}")
    print(f"Device: {args.device}")
    print("-" * 30, "\n")

    extract_save_features(
        args.mask_type,
        args.clip_model, 
        args.dataset_path, 
        save_path, 
        full_clip,
        ratio/100, 
        use_masking, 
        args.device
        )