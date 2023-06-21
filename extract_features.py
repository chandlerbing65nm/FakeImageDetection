import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from clip import load
import pickle
import timm

from dataset import WangEtAlDataset

class CLIPFeatureExtractor:
    def __init__(self, model_name='ViT-L/14'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = load(model_name, device=self.device, jit=False)
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
    def __init__(self, model_name='vit_base_patch16_224.augreg_in21k'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

    def extract_features(self, dataloader):
        real_embeddings = []
        fake_embeddings = []

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                features = self.model.forward_features(imgs)
                features = features.cpu().numpy()

                for feature, label in zip(features, labels):
                    if label == 0:  # 'real'
                        real_embeddings.append(feature)
                    else:  # 'fake'
                        fake_embeddings.append(feature)

        return np.array(real_embeddings), np.array(fake_embeddings)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = WangEtAlDataset('/home/paperspace/Documents/chandler/ForenSynths/wang_et_al/training', transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)

    feature_extractor = ImageNetFeatureExtractor()
    real_embeddings, fake_embeddings = feature_extractor.extract_features(dataloader)

    with open('in21k_embeddings.pkl', 'wb') as f:
        pickle.dump((real_embeddings, fake_embeddings), f)