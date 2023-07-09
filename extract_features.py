import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
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

from dataset import WangEtAlDataset, CorviEtAlDataset
from wangetal_augment import ImageAugmentor

class CLIPFeatureExtractor:
    def __init__(self, model_name='RN50'): # ViT-L/14
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()

    def extract_features(self, dataloader):
        real_embeddings = []
        fake_embeddings = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader):
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

class ImageNetFeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(self, dataloader):
        real_embeddings = []
        fake_embeddings = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                features = self.model(imgs)
                features = features.cpu().numpy()

                for feature, label in zip(features, labels):
                    if label == 0:  # 'real'
                        real_embeddings.append(feature)
                    else:  # 'fake'
                        fake_embeddings.append(feature)

        return np.array(real_embeddings), np.array(fake_embeddings)

def create_transform(augmentor):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # Normalize image data to [-1, 1]
    ])
    return transform

if __name__ == "__main__":
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
    # _, transform = clip.load("ViT-L/14", device="cuda:0")

    dataset = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/training', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # feature_extractor = CLIPFeatureExtractor()
    feature_extractor = ImageNetFeatureExtractor()
    real_embeddings, fake_embeddings = feature_extractor.extract_features(dataloader)

    with open('embeddings/rn50_imagenet_embeddings.pkl', 'wb') as f:
        pickle.dump((real_embeddings, fake_embeddings), f)
