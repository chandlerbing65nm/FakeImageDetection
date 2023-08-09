
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

from dataset import WangEtAlDataset, CorviEtAlDataset
from extract_features import FullCLIPFeatureExtractor, CLIPFeatureExtractor
from wangetal_augment import ImageAugmentor

class LinearClassifier(nn.Module):
    def __init__(self, input_size):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.fc(x)
        # x = self.sigmoid(x)
        # x = x.squeeze()
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = input_size // 2
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc3 = nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        torch.nn.init.normal_(self.fc1.weight.data, 0.0, 0.02)
        # torch.nn.init.normal_(self.fc2.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(self.fc3.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, nhead, num_layers):
        super(SelfAttentionClassifier, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = input_size
        self.nhead = nhead
        self.num_layers = num_layers

        # Define the transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.num_layers)

        # Define the final Linear layer
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size // 2)
        self.fc2 = nn.Linear(self.embedding_size // 2, 1)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x shape: batch_size x seq_len
        x = x.unsqueeze(1)  # x shape: batch_size x 1 x seq_len

        # Pass through transformer
        x = self.transformer_encoder(x)  # x shape: batch_size x 1 x embedding_size

        # Get the attention vector
        attn_vector = x[:, -1, :]  # x shape: batch_size x embedding_size

        # Pass through final MLP layer
        x = self.fc1(attn_vector)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# class EarlyStopping:
#     def __init__(self, path, patience=7, verbose=False, delta=0, min_lr=1e-6, factor=0.1):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.min_lr = min_lr
#         self.factor = factor
#         self.path = path  # add this line

#     def __call__(self, val_loss, model, optimizer, epoch):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, epoch)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 for param_group in optimizer.param_groups:
#                     if param_group['lr'] > self.min_lr:
#                         print(f'Reducing learning rate from {param_group["lr"]} to {param_group["lr"] * self.factor}')
#                         param_group['lr'] *= self.factor
#                         self.counter = 0  # reset the counter
#                     else:
#                         self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, epoch)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, epoch):
#         if self.verbose and epoch % 1 == 0:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), self.path)  # change here to use self.path
#         self.val_loss_min = val_loss


# def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, save_path='./'):

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     early_stopping = EarlyStopping(path=save_path, patience=5, verbose=True)
#     for epoch in range(num_epochs):
#         if epoch % 1 == 0:  # Only print every 20 epochs
#             print('\n')
#             print(f'Epoch {epoch+1}/{num_epochs}')
#             print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#                 data_loader = train_loader
#             else:
#                 model.eval()
#                 data_loader = val_loader

#             running_loss = 0.0
#             y_true, y_pred = [], []

#             for inputs, labels in data_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs).view(-1).unsqueeze(1)
#                     loss = criterion(outputs.squeeze(1), labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 y_pred.extend(outputs.sigmoid().detach().cpu().numpy())
#                 y_true.extend(labels.cpu().numpy())

#             epoch_loss = running_loss / len(data_loader.dataset)
            
#             y_true, y_pred = np.array(y_true), np.array(y_pred)
#             acc = accuracy_score(y_true, y_pred > 0.5)
#             ap = average_precision_score(y_true, y_pred)
            
#             if epoch % 1 == 0:  # Only print every 20 epochs
#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

#             # Early stopping
#             if phase == 'val':
#                 early_stopping(epoch_loss, model, optimizer, epoch)
#                 if early_stopping.early_stop:
#                     print("Early stopping")
#                     return model
        
#         # Save the model after every epoch
#         # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

#     return model

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


def main(
    nhead=8,
    num_layers=6,
    num_epochs=10000,
    embedding_path=None,
    model_type='attention', 
    save_path=None,
    early_stop=True,
    device="cpu",
    ):

    # Load embeddings
    with open(embedding_path, 'rb') as f:
        real_embeddings, fake_embeddings = pickle.load(f)

    # Creating training dataset from embeddings
    embeddings = np.concatenate((real_embeddings, fake_embeddings), axis=0)
    labels = np.array([0] * len(real_embeddings) + [1] * len(fake_embeddings))
    train_data = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Extracting features from validation set
    val_dataset = WangEtAlDataset('../../Datasets/Wang_CVPR20/wang_et_al/validation', transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    feature_extractor = CLIPFeatureExtractor(model_name='ViT-B/16', device=device)
    # Extract the features
    val_real_embeddings, val_fake_embeddings = feature_extractor.extract_features(val_dataloader)

    # Creating validation dataset from embeddings
    val_embeddings = np.concatenate((val_real_embeddings, val_fake_embeddings), axis=0)
    val_labels = np.array([0] * len(val_real_embeddings) + [1] * len(val_fake_embeddings))
    val_data = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Creating and training the binary classifier

    feature_size = real_embeddings.shape[1] # Inspect the size of the embeddings
    if model_type == 'linear':
        model = LinearClassifier(input_size=feature_size).to(device)
    elif model_type == 'mlp':
        model = MLPClassifier(input_size=feature_size).to(device)
    elif model_type == 'attention':
        model = SelfAttentionClassifier(input_size=feature_size, nhead=nhead, num_layers=num_layers).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0) 

    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        save_path=save_path,
        early_stopping_enabled=early_stop,
        )

# if __name__ == "__main__":
#     nhead=8
#     num_layers=6
#     mask_generator_type = 'spectral'
#     # mask_generator_type = None
#     mask_ratio=70
#     embedding_path=f'embeddings/masking/vitb16_{mask_generator_type}mask{mask_ratio}clip_embeddings.pkl'
#     model_type='attention'
    
#     if mask_generator_type in ['zoomblock', 'patch', 'spectral', 'shiftedpatch']:
#         save_path=f'checkpoints/mask_{mask_ratio}/vitb16_{mask_generator_type}maskclip_best_{model_type}.pt'
#     else:
#         save_path=f'checkpoints/vitb16_clip_best_{model_type}.pt'

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     main(
#         nhead=nhead, 
#         num_layers=num_layers, 
#         model_type=model_type, 
#         save_path=save_path, 
#         device=device
#         )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your model description here")

    parser.add_argument('--nhead', type=int, default=8, help='Number of heads for attention mechanism')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--early_stop', type=bool, default=False, help='For early stopping')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs training')
    parser.add_argument('--mask_generator_type', default='spectral', 
                        choices=['zoomblock', 'patch', 'spectral', 'shiftedpatch', 'None'], 
                        help='Type of mask generator')
    parser.add_argument('--mask_ratio', type=int, default=50, help='Masking ratio')
    parser.add_argument('--model_type', default='attention', 
                        choices=['attention', 'mlp', 'linear'], 
                        help='Type of model to be used')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        choices=['cuda:0', 'cpu'], help='Computing device to use')

    args = parser.parse_args()
    
    if args.mask_generator_type in ['zoomblock', 'patch', 'spectral', 'shiftedpatch']:
        mask_ratio = args.mask_ratio
        embedding_path = f'embeddings/masking/vitb16_{args.mask_generator_type}mask{mask_ratio}clip_embeddings.pkl'
        save_path = f'checkpoints/mask_{mask_ratio}/vitb16_{args.mask_generator_type}maskclip_best_{args.model_type}'
    else:
        mask_ratio = 0
        embedding_path = f'embeddings/vitb16_clip_embeddings.pkl'
        save_path = f'checkpoints/mask_{mask_ratio}/vitb16_clip_best_{args.model_type}'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Number of Heads: {args.nhead}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Early Stopping: {args.early_stop}")
    print(f"Mask Generator Type: {args.mask_generator_type}")
    print(f"Mask Ratio: {mask_ratio}")
    print(f"Model Type: {args.model_type}")
    print(f"Save path: {save_path}.pth") if args.early_stop else print(f"Save path: {save_path}_epoch{args.num_epochs-1}.pth")
    print(f"Embed path: {embedding_path}")
    print(f"Device: {args.device}")
    print("-" * 30, "\n")

    main(
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        num_epochs=args.num_epochs,
        embedding_path=embedding_path,
        model_type=args.model_type, 
        save_path=save_path, 
        early_stop=args.early_stop,
        device=args.device
    )