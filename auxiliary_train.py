
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
import collections
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import WangEtAlDataset, CorviEtAlDataset
from extract_features import FullCLIPFeatureExtractor, CLIPFeatureExtractor
from wangetal_augment import ImageAugmentor

class EarlyStopping:
    def __init__(self, path, patience=7, verbose=False, delta=0, min_lr=1e-6, factor=0.1, early_stopping_enabled=True, num_epochs=25):

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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] > self.min_lr:
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
                    torch.save(state_dict, f"{self.path}_epoch{epoch-i}" + '.pth')
        
    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose and epoch % 1 == 0:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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
    early_stopping_enabled=True
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    early_stopping = EarlyStopping(
        path=save_path, 
        patience=5, 
        verbose=True, 
        early_stopping_enabled=early_stopping_enabled,
        num_epochs=num_epochs,
        )

    for epoch in range(num_epochs):
        if epoch % 1 == 0:  # Only print every 20 epochs
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

            for inputs, labels in tqdm(data_loader, f"{phase}"):
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
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

            # Early stopping
            if phase == 'Validation':
                early_stopping(epoch_loss, model, optimizer, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    return model
        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model

class ContinuousMaskingModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ContinuousMaskingModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, output_channels) # You can adjust the size here if needed
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, mask=False, device='cuda:0'):
        super(DeepfakeDetectionModel, self).__init__()
        self.clip_model, _ = clip.load('ViT-B/16')
        self.mask = mask
        self.device = device
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.masking_model = ContinuousMaskingModel(input_channels=3, output_channels=512).to(self.device)
        self.mlp_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

    def forward(self, x):
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)
        
        if self.mask is True:
            mask = self.masking_model(x)
            masked_features = clip_features * mask
        else:
            masked_features = clip_features

        output = self.mlp_encoder(masked_features)
        return output

def extract_clip_features(dataset, clip_model):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in tqdm(dataset, desc="Extracting Features"):
            inputs = inputs.unsqueeze(0).to(device)
            feature = clip_model.encode_image(inputs)
            features.append(feature.cpu())
            labels.append(label)

    features_tensor = torch.stack(features).squeeze()
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(features_tensor, labels_tensor)

def create_transform(augmentor, mask_ratio=0.10):
    # Create an instance of the mask generator
    # mask_generator = PatchMaskGenerator(mask_ratio=mask_ratio)

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        # ApplyMask(mask_generator), # Apply the mask after resizing
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform

def main(batch_size, mask, save_path, early_stop, epochs, subset_length):
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

    # Define the dataset
    train_dataset = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/training', transform=transform)

    # subset_length = batch_size * 200  # int(len(train_dataset) * 0.01)
    indices = torch.randperm(len(train_dataset))[:subset_length]
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Extracting features from validation set
    val_dataset = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/validation', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetectionModel(mask=mask, device=device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

    if mask:
        save_path = save_path + f'{int(subset_length)}_samples'

    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        val_loader, 
        num_epochs=epochs, 
        save_path=save_path,
        early_stopping_enabled=early_stop,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deepfake Detection Model')
    parser.add_argument('--early_stop', type=bool, default=False, help='For early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--subset', type=int, default=200, help='to multiply by batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--mask', type=bool, default=True, help='Whether to use mask in the model')
    parser.add_argument('--save_path', type=str, default='checkpoints/auxiliary/auxiliary_', help='Path to save the trained model')

    args = parser.parse_args()

    args.save_path = args.save_path if args.mask else 'checkpoints/mask_0/vitb16_clip_best_mlp'
    subset_length = args.batch_size * args.subset if args.subset != 0 else -1

    main(args.batch_size, args.mask, args.save_path, args.early_stop, args.epochs, subset_length)
