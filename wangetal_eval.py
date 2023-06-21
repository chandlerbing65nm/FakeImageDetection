import torch
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

from wangetal_classifier import resnet50
from wangetal_augment import ImageAugmentor
from dataset import DiffusionDataset, ForenSynths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################
############################################################################
pre_trained_net = '/home/paperspace/Documents/chandler/Mahalanobis_Detector/pretrained/blur_jpg_prob0.5.pth'
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
    'jpg_method': ['cv2', 'pil'],
    'jpg_qual': [75]
}

augmentor = ImageAugmentor(opt)
transform = augmentor.create_transform()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# real_dir = '/home/paperspace/Documents/chandler/UniversalFakeDetection/DiffusionImages/REAL'
# fake_dir = '/home/paperspace/Documents/chandler/UniversalFakeDetection/DiffusionImages/TAMING-TRANSFORMERS'
# test_dataset = DiffusionDataset(fake_dir, real_dir, transform=transform)

root_dir = '/home/paperspace/Documents/chandler/ForenSynths/san'
test_dataset = ForenSynths(root_dir, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Running inference
predictions, labels = [], []

with torch.no_grad():
    for images, lbls in tqdm(test_loader, "looping dataset"):
        images = images.to(device)
        lbls = lbls.to(device)

        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu().numpy()
        pred = np.where(outputs > 0.5, 1, 0)

        predictions.extend(pred)
        labels.extend(lbls.cpu().numpy())

# Convert to numpy arrays
predictions_array = np.array(predictions)
labels_array = np.array(labels)

# Apply threshold to predictions to get binary label
predictions_binary = (predictions_array > 0.5).astype(int)

# Calculate metrics for each class
if np.any(labels_array == 0):
    precision_real = precision_score(labels_array, predictions_binary, pos_label=0) # assuming 0 label represents 'real'
    recall_real = recall_score(labels_array, predictions_binary, pos_label=0) # assuming 0 label represents 'real'
    print(f'Precision (Real): {precision_real}')
    print(f'Recall (Real): {recall_real}')

if np.any(labels_array == 1):
    precision_fake = precision_score(labels_array, predictions_binary, pos_label=1) # assuming 1 label represents 'fake'
    recall_fake = recall_score(labels_array, predictions_binary, pos_label=1) # assuming 1 label represents 'fake'
    print(f'Precision (Fake): {precision_fake}')
    print(f'Recall (Fake): {recall_fake}')

# Calculate accuracy
accuracy = accuracy_score(labels_array, predictions_binary)
print(f'Accuracy: {accuracy}')
