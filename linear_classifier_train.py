
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

from dataset import WangEtAlDataset, CorviEtAlDataset
from extract_features import CLIPFeatureExtractor, ImageNetFeatureExtractor
from wangetal_augment import ImageAugmentor

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.fc(x)
        # x = self.sigmoid(x)
        # x = x.squeeze()
        return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, min_lr=1e-6, factor=0.1):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_lr = min_lr
        self.factor = factor

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > self.min_lr:
                        print(f'Reducing learning rate from {param_group["lr"]} to {param_group["lr"] * self.factor}')
                        param_group['lr'] *= self.factor
                        self.counter = 0 # reset the counter
                    else:
                        self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoints/rn50_imagenet_best.pt')
        self.val_loss_min = val_loss


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            y_true, y_pred = [], []

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).view(-1).unsqueeze(1)
                    loss = criterion(outputs.squeeze(1), labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
                y_true.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(data_loader.dataset)
            
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

            # Early stopping
            if phase == 'val':
                early_stopping(epoch_loss, model, optimizer)
                if early_stopping.early_stop:
                    print("Early stopping")
                    return model
        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model

def create_transform(augmentor):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: augmentor.custom_resize(img)),
        transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # Normalize image data to [-1, 1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

    # Load embeddings
    with open('embeddings/rn50_imagenet_embeddings.pkl', 'rb') as f:
        real_embeddings, fake_embeddings = pickle.load(f)

    # Creating training dataset from embeddings
    embeddings = np.concatenate((real_embeddings, fake_embeddings), axis=0)
    labels = np.array([0] * len(real_embeddings) + [1] * len(fake_embeddings))
    train_data = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Extracting features from validation set
    val_dataset = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/validation', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_real_embeddings, val_fake_embeddings = ImageNetFeatureExtractor().extract_features(val_dataloader)

    # Creating validation dataset from embeddings
    val_embeddings = np.concatenate((val_real_embeddings, val_fake_embeddings), axis=0)
    val_labels = np.array([0] * len(val_real_embeddings) + [1] * len(val_fake_embeddings))
    val_data = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

    # Creating and training the binary classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_size = real_embeddings.shape[1] # Inspect the size of the embeddings
    model = BinaryClassifier(input_size=feature_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0) 
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.0, weight_decay=0)

    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10000)