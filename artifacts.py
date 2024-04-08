#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os
import argparse
import random
import math
import torch.nn as nn
import torch
import scipy.stats
import torch
from scipy.ndimage import gaussian_filter


class FrequencyMaskGenerator:
    def __init__(self, ratio: float = 0.3, band: str = 'all') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all'

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features_np = features.cpu().numpy()
        features_np = self._apply_gaussian_filter(features_np, sigma=1.0)
        # features_np = features.cpu().numpy().astype(np.complex64)
        # freq_features = np.fft.fftn(features_np, axes=(-2, -1))
        batch, channel, height, width = features_np.shape
        freq_features = scipy.fftpack.dct(scipy.fftpack.dct(features_np, axis=-2, norm='ortho'), axis=-1, norm='ortho')
    
        mask = self._create_balanced_mask(batch, channel, height, width)
        masked_freq_features = freq_features * mask
        # masked_features_np = np.fft.ifftn(masked_freq_features, axes=(-2, -1)).real
        return masked_freq_features

    def _create_balanced_mask(self, batch, channel, height, width):
        # Initialize the mask with ones for all dimensions
        mask = np.ones((batch, channel, height, width), dtype=np.float32)

        # Determine the region of the frequency domain to mask
        if self.band == 'low':
            y_start, y_end = 0, 3*height // 4
            x_start, x_end = 0, 3*width // 4
        elif self.band == 'mid':
            y_start, y_end = height // 4, 3 * height // 4
            x_start, x_end = width // 4, 3 * width // 4
        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f"Invalid band: {self.band}")

        # Calculate the number of frequencies to mask
        num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
        mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]

        y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
        x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

        # Apply the mask to the specified frequencies for all batches and channels
        for b in range(batch):
            for c in range(channel):
                mask[b, c, y_indices, x_indices] = 0
        return mask

    def _apply_gaussian_filter(self, tensor, sigma=1.0):
        """
        Apply a Gaussian filter to a 4D numpy tensor of shape [batch, channel, H, W].

        Parameters:
        - tensor: A numpy tensor of shape [batch, channel, H, W].
        - sigma: The standard deviation for Gaussian kernel, controlling the amount of blurring.

        Returns:
        - A new numpy tensor of the same shape as input, with Gaussian filter applied.
        """
        # Apply the Gaussian filter to each image in the batch
        filtered_tensor = np.zeros_like(tensor)
        for i in range(tensor.shape[0]):  # Iterate through the batch
            for j in range(tensor.shape[1]):  # Iterate through the channels
                # Apply the Gaussian filter on the [H, W] dimensions
                filtered_tensor[i, j] = gaussian_filter(tensor[i, j], sigma=sigma)
        
        return filtered_tensor

def normalize(input_tensor):
    """
    Normalize the input tensor using min-max normalization.
    """
    tensor_min = torch.min(input_tensor)
    tensor_max = torch.max(input_tensor)
    normalized_tensor = (input_tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def spectral_energy(dct_coefficients):
    """
    Calculate the spectral energy of the DCT coefficients.
    """
    return np.sqrt(np.sum(dct_coefficients ** 2))

def extraction_tensor(input_tensor, output_dir, output_code, device='cuda:0', print_scalars=False):

    maskgen = FrequencyMaskGenerator(ratio=0.75, band='low')
    features = maskgen.transform(input_tensor)
    energy = spectral_energy(features)

    print(f"{output_code} - Spectral Energy: {energy:.3f}")

    # else:
    #     # Visualize and save results
    #     figures_output_dir = os.path.join(output_dir, output_code)
    #     os.makedirs(figures_output_dir, exist_ok=True)
    #     save_figures(dist_out, figures_output_dir)


if __name__ == "__main__":
    # Parameters for the test tensor
    batch_size = 64  # Number of images in the batch
    channels = 128    # Number of channels
    height = 28     # Height of the image
    width = 28      # Width of the image

    # Creating a random tensor to simulate the output of a conv2d layer
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Define the output directory and code (subdirectory)
    output_dir = "./artifacts"
    output_code = "test_run"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration - use CUDA if available, else CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Execute the extraction function with the test tensor
    extraction_tensor(input_tensor, output_dir, output_code, device=device)

    # print(f"Process completed. Check the {output_dir}/{output_code} directory for the output.")