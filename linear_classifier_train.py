
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from clip import load
import numpy as np
import pickle
from sklearn.metrics import average_precision_score

from dataset import WangEtAlDataset
from extract_features import CLIPFeatureExtractor, ImageNetFeatureExtractor

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoints/rn50_in1k_best_model.pt')
        self.val_loss_min = val_loss


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    early_stopping = EarlyStopping(patience=7, verbose=True)
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
            running_corrects = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Early stopping
            if phase == 'val':
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    return model
        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Load embeddings
    with open('embeddings/r50_in1k_embeddings.pkl', 'rb') as f:
        real_embeddings, fake_embeddings = pickle.load(f)

    # Creating training dataset from embeddings
    embeddings = np.concatenate((real_embeddings, fake_embeddings), axis=0)
    labels = np.array([0] * len(real_embeddings) + [1] * len(fake_embeddings))
    train_data = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Extracting features from validation set
    val_dataset = WangEtAlDataset('/home/paperspace/Documents/chandler/ForenSynths/wang_et_al/validation', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    val_real_embeddings, val_fake_embeddings = ImageNetFeatureExtractor().extract_features(val_dataloader)

    # Creating validation dataset from embeddings
    val_embeddings = np.concatenate((val_real_embeddings, val_fake_embeddings), axis=0)
    val_labels = np.array([0] * len(val_real_embeddings) + [1] * len(val_fake_embeddings))
    val_data = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

    # Creating and training the binary classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_size = real_embeddings.shape[1] # Inspect the size of the embeddings
    model = BinaryClassifier(input_size=feature_size).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)