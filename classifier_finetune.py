
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

from dataset import WangEtAlDataset, CorviEtAlDataset, GenImage
from extract_features import CLIPFeatureExtractor, ImageNetFeatureExtractor, ViTFeatureExtractor
from wangetal_augment import ImageAugmentor
from classifier_train import MLPClassifier, SelfAttentionClassifier, LinearClassifier
from utils import *

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

    def __call__(self, val_loss, model, optimizer, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if epoch % 20 == 0:  # Only print every 20 epochs
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > self.min_lr:
                        # if epoch % 20 == 0:  # Only print every 20 epochs
                        print(f'Reducing learning rate from {param_group["lr"]} to {param_group["lr"] * self.factor}')
                        param_group['lr'] *= self.factor
                        self.counter = 0 # reset the counter
                    else:
                        self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose and epoch % 19 == 0:  # Only print every 20 epochs
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoints/mask_10/vitb16_firstpatch16maskclip_best_linearft_fs16.pt')
        self.val_loss_min = val_loss


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(num_epochs):
        if epoch % 19 == 0:  # Only print every 20 epochs
            print('\n')
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
            
            if epoch % 19 == 0:  # Only print every 20 epochs
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} AP: {ap:.4f}')

            # Early stopping
            if phase == 'val':
                early_stopping(epoch_loss, model, optimizer, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    return model
        
        # Save the model after every epoch
        # torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}.pth')

    return model

# def create_transform(augmentor):
#     transform = transforms.Compose([
#         transforms.Lambda(lambda img: augmentor.custom_resize(img)),
#         transforms.Lambda(lambda img: augmentor.data_augment(img)),  # Pass opt dictionary here
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # Normalize image data to [-1, 1]
#         # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     return transform

# if __name__ == "__main__":

#     # Set options for augmentation
#     opt = {
#         'rz_interp': ['bilinear'],
#         'loadSize': 256,
#         'blur_prob': 0.1,  # Set your value
#         'blur_sig': [0.5],
#         'jpg_prob': 0.1,  # Set your value
#         'jpg_method': ['cv2'],
#         'jpg_qual': [75]
#     }

class ApplyMask:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def __call__(self, img):
        # Convert PIL image to PyTorch tensor
        img_tensor = transforms.ToTensor()(img)
        # Apply the mask
        masked_img_tensor = self.mask_generator.transform(img_tensor)
        # Return as PIL image
        return transforms.ToPILImage()(masked_img_tensor)
        
def create_transform(augmentor, mask_ratio=0.10):
    # Create an instance of the mask generator
    mask_generator = PatchMaskGenerator(mask_ratio=mask_ratio)

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

    #--------------------------------------TRAINING-----------------------------------------------#
    #---------------------------------------------------------------------------------------------#
    # Extracting features from training set
    train_dataset = GenImage('../../Datasets/GenImage/imagenet_glide/imagenet_glide/train', transform=transform, sample_size=64*32)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    train_real_embeddings, train_fake_embeddings = CLIPFeatureExtractor(model_name='ViT-B/16', mask_images=True, mask_ratio=0.1).extract_features(train_dataloader)
    # train_real_embeddings, train_fake_embeddings = CLIPFeatureExtractor(model_name='ViT-B/16').extract_features(train_dataloader)
    
    # Creating training dataset from embeddings
    embeddings = np.concatenate((train_real_embeddings, train_fake_embeddings), axis=0)
    labels = np.array([0] * len(train_real_embeddings) + [1] * len(train_fake_embeddings))

    # Save embeddings and labels
    with open('few-shot/fstrain_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open('few-shot/fstrain_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    raise ValueError("meh")

    # Load embeddings and labels
    with open('few-shot/fstrain_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('few-shot/fstrain_labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # Sample few-shot data
    num_samples_per_class = 16

    # Assuming labels is a numpy array with sorted labels
    classes, counts = np.unique(labels, return_counts=True)
    indices = []
    for c in classes:
        # Get the indices of the first # occurrences of the class
        class_indices = np.where(labels == c)[0][:num_samples_per_class]
        indices.extend(class_indices)
    indices = np.array(indices)

    # Now use these indices to select the first 4 occurrences from the embeddings and labels
    few_shot_embeddings = embeddings[indices]
    few_shot_labels = labels[indices]

    # Creating few-shot training dataset from embeddings
    train_data = TensorDataset(torch.tensor(few_shot_embeddings, dtype=torch.float32), torch.tensor(few_shot_labels, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    #--------------------------------------VALIDATION-----------------------------------------------#
    #-----------------------------------------------------------------------------------------------#
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Extracting features from validation set
    val_dataset = GenImage('../../Datasets/GenImage/imagenet_glide/imagenet_glide/val', transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_real_embeddings, val_fake_embeddings = CLIPFeatureExtractor(model_name='ViT-B/16').extract_features(val_dataloader)

    # Creating validation dataset from embeddings
    val_embeddings = np.concatenate((val_real_embeddings, val_fake_embeddings), axis=0)
    val_labels = np.array([0] * len(val_real_embeddings) + [1] * len(val_fake_embeddings))
    
    # # Save embeddings and labels
    # with open('few-shot/fsval_embeddings.pkl', 'wb') as f:
    #     pickle.dump(val_embeddings, f)
    # with open('few-shot/fsval_labels.pkl', 'wb') as f:
    #     pickle.dump(val_labels, f)
    # raise ValueError("meh")

    # Load embeddings and labels
    with open('few-shot/fsval_embeddings.pkl', 'rb') as f:
        fsval_embeddings = pickle.load(f)
    with open('few-shot/fsval_labels.pkl', 'rb') as f:
        fsval_labels = pickle.load(f)

    # Sample few-shot data
    val_shot_list = {1: 1, 2: 2, 4: 4, 8: 4, 16: 4}
    val_num_shot = val_shot_list[num_samples_per_class]

    # Assuming labels is a numpy array with sorted labels
    val_classes, val_counts = np.unique(fsval_labels, return_counts=True)
    val_indices = []
    for c in val_classes:
        # Get the indices of the first 4 occurrences of the class
        val_class_indices = np.where(fsval_labels == c)[0][:val_num_shot]
        val_indices.extend(val_class_indices)
    val_indices = np.array(val_indices)

    # Now use these indices to select the first 4 occurrences from the embeddings and labels
    val_embeddings = fsval_embeddings[val_indices]
    val_labels = fsval_labels[val_indices]

    val_data = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)


    #--------------------------------------MODEL-----------------------------------------------#
    #------------------------------------------------------------------------------------------#
    # Creating and training the binary classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model weights
    pretrained_weights = torch.load('./checkpoints/mask_10/vitb16_firstpatch16maskclip_best_linear.pt')

    feature_size = val_real_embeddings.shape[1] # Inspect the size of the embeddings
    model = LinearClassifier(input_size=feature_size).to(device)
    # model = MLPClassifier(input_size=feature_size).to(device)
    # model = SelfAttentionClassifier(input_size=feature_size, nhead=4, num_layers=2).to(device)
    
    # Load the pre-trained model weights into your model
    model.load_state_dict(pretrained_weights)

    #--------------------------------------OPTIMIZATION-----------------------------------------------#
    #-------------------------------------------------------------------------------------------------#
    # Creating optimization
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0) 
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.0, weight_decay=0)

    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10000)