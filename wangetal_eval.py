import torch
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, average_precision_score
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

from wangetal_classifier import resnet50
from wangetal_augment import ImageAugmentor
from dataset import CorviEtAlDataset, ForenSynths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################
############################################################################
pre_trained_net = '../Mahalanobis_Detector/pretrained/blur_jpg_prob0.1.pth'
model = resnet50(num_classes=1)
state_dict = torch.load(pre_trained_net, map_location=device)
model.load_state_dict(state_dict['model'])
model.cuda()
model.eval()

# Set options for augmentation
opt = {
    'blur_prob': 0.5,  # Set your value
    'blur_sig': [0.5],
    'jpg_prob': 0.5,  # Set your value
    'jpg_method': ['cv2'],
    'jpg_qual': [75]
}

augmentor = ImageAugmentor(opt)
transform = augmentor.create_transform()

# root_dir = '../../Datasets/Wang_CVPR20/imle'
# test_dataset = ForenSynths(root_dir, transform=transform)

real_dir = '../../Datasets/Corvi_ICASSP23/REAL'
fake_dir = '../../Datasets/Corvi_ICASSP23/STABLE_DIFFUSION'
test_dataset = CorviEtAlDataset(fake_dir, real_dir, transform=transform)

# Lists to store metrics for each run
real_accuracies, fake_accuracies, accuracies, average_precisions = [], [], [], []

for _ in range(1):
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in tqdm(test_loader, "accessing test dataloder"):
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    real_accuracies.append(r_acc)
    fake_accuracies.append(f_acc)
    accuracies.append(acc)
    average_precisions.append(ap)

# Calculate average and standard deviation for each metric
avg_real_acc = np.mean(real_accuracies)
std_real_acc = np.std(real_accuracies)

avg_fake_acc = np.mean(fake_accuracies)
std_fake_acc = np.std(fake_accuracies)

avg_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

avg_ap = np.mean(average_precisions)
std_ap = np.std(average_precisions)

# Print results
# print(f'Average Accuracy (Real): {avg_real_acc}, Std Dev: {std_real_acc}')
# print(f'Average Accuracy (Fake): {avg_fake_acc}, Std Dev: {std_fake_acc}')
print(f'Average Accuracy: {avg_acc}, Std Dev: {std_acc}')
print(f'Average Precision: {avg_ap}, Std Dev: {std_ap}')