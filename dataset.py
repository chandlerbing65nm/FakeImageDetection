import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

class ForenSynths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0_real', '1_fake']
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

class OjhaCVPR23(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.fake_index = 1  # Index of 'fake' in ['real', 'fake']
        self.real_index = 0  # Index of 'real' in ['real', 'fake']

        sub_folders = os.listdir(root_dir)
        
        if '1_fake' in sub_folders and '0_real' in sub_folders:
            # This is the 'biggan' case
            fake_dir = os.path.join(root_dir, '1_fake')
            real_dir = os.path.join(root_dir, '0_real')
            self._process_folder(fake_dir, self.fake_index)
            self._process_folder(real_dir, self.real_index)
        else:
            # This is the 'cyclegan' case
            for folder in sub_folders:
                fake_dir = os.path.join(root_dir, folder, '1_fake')
                real_dir = os.path.join(root_dir, folder, '0_real')
                self._process_folder(fake_dir, self.fake_index)
                self._process_folder(real_dir, self.real_index)
            
    def _process_folder(self, folder_path, index):
        # Iterate over files
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Append a tuple (file_path, index)
            self.data.append((file_path, index))
            
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

class Wang_CVPR20(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.fake_index = 1  # Index of 'fake' in ['real', 'fake']
        self.real_index = 0  # Index of 'real' in ['real', 'fake']

        sub_folders = os.listdir(root_dir)
        
        if '1_fake' in sub_folders and '0_real' in sub_folders:
            # This is the 'biggan' case
            fake_dir = os.path.join(root_dir, '1_fake')
            real_dir = os.path.join(root_dir, '0_real')
            self._process_folder(fake_dir, self.fake_index)
            self._process_folder(real_dir, self.real_index)
        else:
            # This is the 'cyclegan' case
            for folder in sub_folders:
                fake_dir = os.path.join(root_dir, folder, '1_fake')
                real_dir = os.path.join(root_dir, folder, '0_real')
                self._process_folder(fake_dir, self.fake_index)
                self._process_folder(real_dir, self.real_index)
            
    def _process_folder(self, folder_path, index):
        # Iterate over files
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Append a tuple (file_path, index)
            self.data.append((file_path, index))
            
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
