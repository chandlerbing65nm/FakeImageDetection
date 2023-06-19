import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class WangEtAlDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'fake']
        self.data = []

        # Iterate over the categories
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)

            # Iterate over class names (real/fake)
            for class_name in self.classes:
                class_path = os.path.join(category_path, class_name)

                # Iterate over files
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)

                    # Append a tuple (file_path, class_index)
                    self.data.append((file_path, self.classes.index(class_name)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label