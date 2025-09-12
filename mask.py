import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
import random

from dataset import Ojha_CVPR23, Wang_CVPR20

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imsave

import pywt
from scipy.fftpack import dct, idct

class FrequencyMaskGenerator:
    def __init__(self, ratio: float = 0.3, band: str = 'low+high', transform_type: str = 'fourier', channel: str = 'all') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all', 'low+high', 'low+mid', 'mid+high', 'prog'
        self.transform_type = transform_type  # 'fourier', 'cosine', 'wavelet'
        self.alpha = 1
        # Channel to apply masking: 'all', 'r', 'g', 'b', or '0'/'1'/'2'
        self.channel = channel

    def transform(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image).astype(np.complex64)
        if self.transform_type == 'fourier':
            freq_image = np.fft.fftn(image_array, axes=(0, 1))
        elif self.transform_type == 'cosine':
            freq_image = self._dct2(image_array)
        elif self.transform_type == 'wavelet':
            freq_image, self.coeff_slices = self._wavelet_transform(image_array)
        else:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

        height, width, _ = image_array.shape

        mask = self._create_balanced_mask(height, width)
        self.masked_freq_image = freq_image * mask

        if self.transform_type == 'fourier':
            masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real
        elif self.transform_type == 'cosine':
            masked_image_array = self._idct2(self.masked_freq_image).real
            # masked_image_array = np.clip(masked_image_array.real, 0, 255)
        elif self.transform_type == 'wavelet':
            masked_image_array = self._inverse_wavelet_transform(self.masked_freq_image, self.coeff_slices).real
            # masked_image_array = np.clip(masked_image_array.real, 0, 255)

        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)

        y_indices_all = []
        x_indices_all = []

        bands = self.band.split('+')
        use_progressive_masking = 'prog' in bands

        for band in bands:
            if 'prog' in band:
                continue  # skip the 'prog' part of the band

            if band == 'low':
                y_start, y_end = 0, height // 4
                x_start, x_end = 0, width // 4
            elif band == 'mid':
                y_start, y_end = height // 4, 3 * height // 4
                x_start, x_end = width // 4, 3 * width // 4
            elif band == 'high':
                y_start, y_end = 3 * height // 4, height
                x_start, x_end = 3 * width // 4, width
            elif band == 'all':
                y_start, y_end = 0, height
                x_start, x_end = 0, width
            else:
                raise ValueError(f"Invalid band: {band}")

            region_area = (y_end - y_start) * (x_end - x_start)
            num_frequencies = int(np.ceil(region_area * self.ratio))

            if use_progressive_masking:
                distances = np.sqrt((np.arange(y_start, y_end)[:, None] ** 2) + (np.arange(x_start, x_end)[None, :] ** 2))
                max_distance = distances.max()
                progressive_ratio = self.alpha * self.ratio * (1 - distances / max_distance)
                progressive_ratio = np.clip(progressive_ratio, 0, 1)

                flattened_distances = distances.flatten()
                flattened_ratio = progressive_ratio.flatten()

                mask_frequencies_indices = np.random.choice(
                    flattened_distances.size, 
                    size=num_frequencies, 
                    replace=False, 
                    p=flattened_ratio/flattened_ratio.sum()
                    )
                y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
                x_indices = mask_frequencies_indices % (x_end - x_start) + x_start
            else:
                mask_frequencies_indices = np.random.permutation(region_area)[:num_frequencies]
                y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
                x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

            y_indices_all.extend(y_indices)
            x_indices_all.extend(x_indices)

        # Apply masking on the selected channel(s)
        ch = self.channel
        if isinstance(ch, str):
            ch_l = ch.lower()
            mapping = {'all': None, 'r': 0, 'g': 1, 'b': 2, '0': 0, '1': 1, '2': 2}
            ch_idx = mapping.get(ch_l, None)
        else:
            ch_idx = int(ch)

        if ch_idx is None:
            # Mask all channels (default behavior)
            mask[y_indices_all, x_indices_all, :] = 0
        else:
            mask[y_indices_all, x_indices_all, ch_idx] = 0
        return mask

    def _dct2(self, a):
        return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

    def _idct2(self, a):
        return idct(idct(a, axis=1, norm='ortho'), axis=0, norm='ortho')

    def _wavelet_transform(self, a):
        # Perform wavelet decomposition on axes (0, 1)
        coeffs = pywt.wavedecn(a, wavelet='haar', level=2, axes=(0, 1))
        
        # Convert to array and slices with the same axes
        arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=(0, 1))
        return arr, coeff_slices

    def _inverse_wavelet_transform(self, arr, coeff_slices):
        # Convert array back to coefficients with the same axes
        coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedecn')
        
        # Perform wavelet reconstruction on axes (0, 1)
        rec = pywt.waverecn(coeffs, wavelet='haar', axes=(0, 1))
        return rec




class PatchMaskGenerator:
    def __init__(self, ratio: float = 0.3) -> None:
        self.ratio = ratio

    def transform(self, image: Image.Image) -> Image.Image:
        # Get the height and width of the image
        width, height = image.size

        # Compute the patch size
        patch_size = 16
        while height % patch_size != 0 or width % patch_size != 0:
            patch_size -= 1

        # Compute the number of patches
        num_patches = (height * width) // (patch_size * patch_size)

        # Compute the number of patches to mask
        mask_patches = int(np.ceil(num_patches * self.ratio))

        # Create a mask of ones
        mask = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(mask)

        # Randomly select patches to mask
        mask_patch_indices = random.sample(range(num_patches), mask_patches)
        
        for index in mask_patch_indices:
            start_y = (index // (width // patch_size)) * patch_size
            start_x = (index % (width // patch_size)) * patch_size
            draw.rectangle([start_x, start_y, start_x + patch_size, start_y + patch_size], fill=0)

        # Convert both image and mask to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask) / 255.0  # Normalize to [0, 1]

        # If the image is a 3-channel image, repeat the mask for all channels
        if len(image_np.shape) == 3:
            mask_np = np.expand_dims(mask_np, axis=-1)
            mask_np = np.repeat(mask_np, image_np.shape[-1], axis=-1)

        # Apply the mask
        masked_image_np = image_np * mask_np

        # Convert the numpy array back to a PIL Image
        masked_image = Image.fromarray(np.uint8(masked_image_np))

        return masked_image

class PixelMaskGenerator:
    def __init__(self, ratio: float = 0.6) -> None:
        self.ratio = ratio

    def transform(self, pil_image):
        # Convert PIL image to numpy array
        image = np.array(pil_image)

        # Infer the height and width from the image
        height, width, channels = image.shape
        
        pixel_count = height * width
        mask_count = int(np.ceil(pixel_count * self.ratio))

        # Generate random mask
        mask_idx = np.random.permutation(pixel_count)[:mask_count]
        mask = np.ones(pixel_count, dtype=np.float32)  # Initialize mask as ones
        mask[mask_idx] = 0  # Set selected indices to zero

        mask = mask.reshape((height, width))

        # Repeat the mask for all channels
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

        masked_image = image * mask

        # Convert numpy array back to PIL image
        masked_pil_image = Image.fromarray(np.uint8(masked_image))

        return masked_pil_image

# class FrequencyMaskGenerator:
#     def __init__(self, ratio: float = 0.3, band: str = 'all') -> None:
#         self.ratio = ratio
#         self.band = band  # 'low', 'mid', 'high', 'all'

#     def transform(self, image: Image.Image) -> Image.Image:
#         image_array = np.array(image).astype(np.complex64)
#         freq_image = np.fft.fftn(image_array, axes=(0, 1))

#         height, width, _ = image_array.shape

#         mask = self._create_balanced_mask(height, width)
#         self.masked_freq_image = freq_image * mask
#         masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real
#         masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
#         return masked_image

#     def _create_balanced_mask(self, height, width):
#         mask = np.ones((height, width, 3), dtype=np.complex64)

#         # Determine the region of the frequency domain to mask
#         if self.band == 'low':
#             y_start, y_end = 0, height // 4
#             x_start, x_end = 0, width // 4
#         elif self.band == 'mid':
#             y_start, y_end = height // 4, 3 * height // 4
#             x_start, x_end = width // 4, 3 * width // 4
#         elif self.band == 'high':
#             y_start, y_end = 3 * height // 4, height
#             x_start, x_end = 3 * width // 4, width
#         elif self.band == 'all':
#             y_start, y_end = 0, height
#             x_start, x_end = 0, width
#         else:
#             raise ValueError(f"Invalid band: {self.band}")

#         num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
#         mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
#         y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
#         x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

#         mask[y_indices, x_indices, :] = 0
#         return mask