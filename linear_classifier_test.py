import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm

from dataset import WangEtAlDataset, CorviEtAlDataset, ForenSynths
from extract_features import CLIPFeatureExtractor, ImageNetFeatureExtractor
from linear_classifier_train import BinaryClassifier

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # _, transform = clip.load("ViT-L/14", device="cuda:0")

    # real_dir = '../../Datasets/Corvi_ICASSP23/REAL'
    # fake_dir = '../../Datasets/Corvi_ICASSP23/TAMING_DIFFUSION'
    # test_dataset = CorviEtAlDataset(fake_dir, real_dir, transform=transform)

    root_dir = '../../Datasets/Wang_CVPR20/stargan'
    test_dataset = ForenSynths(root_dir, transform=transform)

    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_real_embeddings, test_fake_embeddings = ImageNetFeatureExtractor().extract_features(test_dataloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model
    feature_size = test_fake_embeddings.shape[1] # Inspect the size of the embeddings
    model = BinaryClassifier(input_size=feature_size).to(device)
    model.load_state_dict(torch.load('checkpoints/rn50_imagenet_best.pt'))

    # Handle the possibility that there might be only one class in the dataset
    if test_real_embeddings.size != 0 and test_fake_embeddings.size != 0:
        # Both 'real' and 'fake' classes are present
        test_embeddings = np.concatenate((test_real_embeddings, test_fake_embeddings), axis=0)
        test_labels = np.array([0] * len(test_real_embeddings) + [1] * len(test_fake_embeddings))
    else:
        # Only 'fake' class is present
        test_embeddings = test_fake_embeddings
        test_labels = np.array([1] * len(test_fake_embeddings))
    
    model.cuda()    
    model.eval()  # set model to evaluation mode

    # Lists to store metrics for each run
    accuracies, average_precisions = [], []

    for _ in range(1):

        test_data = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=4)

        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label in tqdm(test_loader, "accessing test dataloader"):
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)

        accuracies.append(acc)
        average_precisions.append(ap)

avg_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

avg_ap = np.mean(average_precisions)
std_ap = np.std(average_precisions)

# Print results
print(f'Average Accuracy: {avg_acc}, Std Dev: {std_acc}')
print(f'Average Precision: {avg_ap}, Std Dev: {std_ap}')