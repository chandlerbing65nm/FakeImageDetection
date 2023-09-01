import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
import numpy as np
import torch.fft as fft
import torch.nn.functional as F

from dataset import *

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        

class FrequencyMaskGenerator:
    def __init__(self, ratio: float = 0.3) -> None:
        self.ratio = ratio

    def transform(self, image: Image.Image) -> Image.Image:
        # Convert the PIL Image to a complex-valued NumPy array
        image_array = np.array(image).astype(np.complex64)
        freq_image = np.fft.fftn(image_array, axes=(0, 1))

        # Get the height and width of the image
        height, width, _ = image_array.shape

        # Compute the balanced mask
        mask = self._create_balanced_mask(height, width)
        self.masked_freq_image = freq_image * mask
        masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)
        num_frequencies = int(np.ceil(height * width * self.ratio))
        mask_frequencies_indices = np.random.permutation(height * width)[:num_frequencies]
        y_indices = mask_frequencies_indices // width
        x_indices = mask_frequencies_indices % width
        mask[y_indices, x_indices, :] = 0
        return mask

    def get_magnitude_and_phase(self):
        magnitude = np.abs(self.masked_freq_image)
        phase = np.angle(self.masked_freq_image)
        return magnitude, phase

def test_mask_generator(
    image_path, 
    mask_type,
    ratio=0.13, 
    sample_size=20
    ):

    # Create a MaskGenerator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    else:
        mask_generator = None

    transform = transforms.Compose([
        transforms.Lambda(lambda img: mask_generator.transform(img)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data = ForenSynths(image_path, transform=transform)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    # Access the first image and label directly
    image, label = data[123]
    image_to_save = image

    # Convert the tensor image to NumPy and transpose if necessary
    image_to_save = image_to_save.numpy().transpose(1, 2, 0)

    # Clip the values to the range [0, 1] if the image is in float format
    if image_to_save.dtype == np.float32 or image_to_save.dtype == np.float64:
        image_to_save = np.clip(image_to_save, 0, 1)

    sample_path = f'./samples'
    os.makedirs(sample_path, exist_ok=True)

    # Display and save the image
    plt.imshow(image_to_save)
    plt.axis('off')  # Optional, to turn off axes
    plt.savefig(f"{sample_path}/masked_{mask_type}.jpg")

# Usage:
test_mask_generator(
    '../../Datasets/Wang_CVPR2020/validation', 
    mask_type='spectral',
    ratio=0.7
    )