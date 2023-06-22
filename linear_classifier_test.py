import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
from clip import load
from tqdm import tqdm

from dataset import WangEtAlDataset, DiffusionDataset, ForenSynths
from extract_features import CLIPFeatureExtractor, ImageNetFeatureExtractor
from linear_classifier_train import BinaryClassifier

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # real_dir = '/home/paperspace/Documents/chandler/UniversalFakeDetection/DiffusionImages/REAL'
    # fake_dir = '/home/paperspace/Documents/chandler/UniversalFakeDetection/DiffusionImages/TAMING-TRANSFORMERS'
    # test_dataset = DiffusionDataset(fake_dir, real_dir, transform=transform)

    root_dir = '/home/paperspace/Documents/chandler/ForenSynths/imle'
    test_dataset = ForenSynths(root_dir, transform=transform)

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    test_real_embeddings, test_fake_embeddings = ImageNetFeatureExtractor().extract_features(test_dataloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model
    feature_size = test_fake_embeddings.shape[1] # Inspect the size of the embeddings
    model = BinaryClassifier(input_size=feature_size).to(device)
    model.load_state_dict(torch.load('checkpoints/rn50_in1k_best_model.pt'))

    # Handle the possibility that there might be only one class in the dataset
    if test_real_embeddings.size != 0 and test_fake_embeddings.size != 0:
        # Both 'real' and 'fake' classes are present
        test_embeddings = np.concatenate((test_real_embeddings, test_fake_embeddings), axis=0)
        test_labels = np.array([0] * len(test_real_embeddings) + [1] * len(test_fake_embeddings))
    else:
        # Only 'fake' class is present
        test_embeddings = test_fake_embeddings
        test_labels = np.array([1] * len(test_fake_embeddings))

    test_data = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()  # set model to evaluation mode
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            preds_list.extend(outputs.detach().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # # Compute mean average precision (mAP)
    # mAP = average_precision_score(labels_list, preds_list)
    # print(f'Test mAP: {mAP}')

    # Convert to numpy arrays
    preds_array = np.array(preds_list)
    labels_array = np.array(labels_list)

    # Apply threshold to predictions to get binary label
    preds_binary = (preds_array > 0.5).astype(int)

    # Calculate metrics for each class
    if np.any(labels_array == 0):
        precision_real = precision_score(labels_array, preds_binary, pos_label=0) # assuming 0 label represents 'real'
        recall_real = recall_score(labels_array, preds_binary, pos_label=0) # assuming 0 label represents 'real'
        print(f'Precision (Real): {precision_real}')
        print(f'Recall (Real): {recall_real}')

    if np.any(labels_array == 1):
        precision_fake = precision_score(labels_array, preds_binary, pos_label=1) # assuming 1 label represents 'fake'
        recall_fake = recall_score(labels_array, preds_binary, pos_label=1) # assuming 1 label represents 'fake'
        print(f'Precision (Fake): {precision_fake}')
        print(f'Recall (Fake): {recall_fake}')

    # Calculate accuracy
    accuracy = accuracy_score(labels_array, preds_binary)
    print(f'Accuracy: {accuracy}')