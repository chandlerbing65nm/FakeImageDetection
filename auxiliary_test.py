
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from clip import load
import numpy as np
import pickle
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import clip
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse
from torch.utils.data import Subset
from torchvision.transforms import Resize, InterpolationMode

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import WangEtAlDataset, CorviEtAlDataset, ForenSynths, GenImage
from extract_features import FullCLIPFeatureExtractor, CLIPFeatureExtractor
from wangetal_augment import ImageAugmentor
from auxiliary_train import *


def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, "Evaluating"):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs).view(-1).unsqueeze(1)
            y_pred.extend(outputs.sigmoid().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)

    # Print results
    print(f'Average Precision: {ap}')
    print(f'Accuracy: {acc}')

    return acc, ap

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_dataset = GenImage('../../Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val', transform=transform)
    # test_dataset = ForenSynths('../../Datasets/Wang_CVPR20/deepfake', transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model_path = 'checkpoints/auxiliary/auxiliary_6400_samples_epoch29.pth'
    model = DeepfakeDetectionModel(mask=True, device=device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    evaluate_model(model, test_loader)


