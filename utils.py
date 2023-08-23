import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from scipy.signal import convolve2d

from dataset import *

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class RandomMaskGenerator:
    def __init__(self, ratio: float = 0.6, device: str = "cpu") -> None:
        self.ratio = ratio
        self.device = device

    def transform(self, image):
        # Infer the height and width from the image
        _, height, width = image.shape
        
        pixel_count = height * width
        mask_count = int(torch.ceil(torch.tensor(pixel_count * self.ratio).to(self.device)))

        # Move the image to the same device as the mask
        image = image.to(self.device)

        mask_idx = torch.randperm(pixel_count, device=self.device)[:mask_count]
        mask = torch.zeros(pixel_count, dtype=torch.float32, device=self.device)
        mask[mask_idx] = 1

        mask = mask.reshape((1, height, width))

        # Repeat the mask for all channels
        mask = mask.repeat((image.shape[0], 1, 1))

        masked_image = image * mask

        return masked_image

class InvBlockMaskGenerator:
    def __init__(self, ratio: float = 0.3, device: str = "cpu") -> None:
        self.ratio = 1-ratio
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

        # Get the height and width of the image
        _, height, width = image.shape

        # Compute the size of the unmasked block
        unmask_area = int(height * width * self.ratio)  # Total number of unmasked pixels
        side_length = int(np.sqrt(unmask_area))  # Side length of the square unmasked block
        unmask_height = side_length
        unmask_width = side_length

        # Create a mask of ones
        mask = torch.ones((1, height, width), dtype=torch.float32, device=self.device)

        # Randomly select the starting point for the unmasked block
        start_y = torch.randint(0, height - unmask_height + 1, (1,)).item()
        start_x = torch.randint(0, width - unmask_width + 1, (1,)).item()

        # Create the unmasked block
        mask[:, start_y:start_y + unmask_height, start_x:start_x + unmask_width] = 0

        # Invert the mask
        mask = 1 - mask

        # Repeat the mask for all channels
        mask = mask.repeat((image.shape[0], 1, 1))

        # # Count the number of 1s and 0s in the mask
        # num_ones = torch.sum(mask == 1).item()
        # num_zeros = torch.sum(mask == 0).item()
        # print(f"Number of 1s: {num_ones}, Number of 0s: {num_zeros}")

        masked_image = image * mask

        return masked_image

class BlockMaskGenerator:
    def __init__(self, ratio: float = 0.3, device: str = "cpu") -> None:
        self.ratio = ratio
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

        # Get the height and width of the image
        _, height, width = image.shape

        # Compute the size of the unmasked block
        unmask_area = int(height * width * self.ratio)  # Total number of unmasked pixels
        side_length = int(np.sqrt(unmask_area))  # Side length of the square unmasked block
        unmask_height = side_length
        unmask_width = side_length

        # Create a mask of ones
        mask = torch.ones((1, height, width), dtype=torch.float32, device=self.device)

        # Randomly select the starting point for the unmasked block
        start_y = torch.randint(0, height - unmask_height + 1, (1,)).item()
        start_x = torch.randint(0, width - unmask_width + 1, (1,)).item()

        # Create the unmasked block
        mask[:, start_y:start_y + unmask_height, start_x:start_x + unmask_width] = 0

        # Repeat the mask for all channels
        mask = mask.repeat((image.shape[0], 1, 1))

        # # Count the number of 1s and 0s in the mask
        # num_ones = torch.sum(mask == 1).item()
        # num_zeros = torch.sum(mask == 0).item()
        # print(f"Number of 1s: {num_ones}, Number of 0s: {num_zeros}")

        masked_image = image * mask

        return masked_image

class PatchMaskGenerator:
    def __init__(self, ratio: float = 0.3, device: str = "cpu") -> None:
        self.ratio = ratio
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

        # Get the height and width of the image
        _, height, width = image.shape

        # Compute the patch size
        patch_size = 16
        while height % patch_size != 0 or width % patch_size != 0:
            patch_size -= 1

        # Compute the number of patches
        num_patches = (height * width) // (patch_size * patch_size)

        # Compute the number of patches to mask
        mask_patches = int(np.ceil(num_patches * self.ratio))

        # Create a mask of ones
        mask = torch.ones((1, height, width), dtype=torch.float32, device=self.device)

        # Randomly select patches to mask
        mask_patch_indices = torch.randperm(num_patches, device=self.device)[:mask_patches]
        
        for index in mask_patch_indices:
            start_y = (index // (width // patch_size)) * patch_size
            start_x = (index % (width // patch_size)) * patch_size
            mask[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = 0

        # Repeat the mask for all channels
        mask = mask.repeat((image.shape[0], 1, 1))

        # know the patch size used
        # print(f"Patch Size: {patch_size}")

        masked_image = image * mask

        return masked_image

class ShiftedPatchMaskGenerator:
    def __init__(self, ratio: float = 0.3, grid_size: int = 16, device: str = "cpu") -> None:
        self.ratio = ratio
        self.grid_size = grid_size
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

        # Get the height and width of the image
        _, height, width = image.shape

        # Calculate the patch size
        patch_size_h = height // self.grid_size
        patch_size_w = width // self.grid_size

        # Compute the number of patches
        num_patches = self.grid_size * self.grid_size

        # Compute the number of patches to mask
        mask_patches = int(np.ceil(num_patches * self.ratio))

        # Create a mask of ones
        mask = torch.ones((1, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)

        # Randomly select patches to mask
        mask_patch_indices = torch.randperm(num_patches, device=self.device)[:mask_patches]
        mask[:, mask_patch_indices // self.grid_size, mask_patch_indices % self.grid_size] = 0

        # Upscale the mask to match the image size
        mask = mask.repeat_interleave(patch_size_h, dim=1).repeat_interleave(patch_size_w, dim=2)

        # Shift the mask
        shift_y = torch.randint(-patch_size_h // 2, patch_size_h // 2 + 1, (1,)).item()
        shift_x = torch.randint(-patch_size_w // 2, patch_size_w // 2 + 1, (1,)).item()
        mask_shifted = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))

        # Apply the original and shifted masks to the image
        masked_image = image * mask * mask_shifted

        return masked_image


# class FrequencyMaskGenerator:
#     def __init__(self, ratio: float = 0.3, device: str = "cpu") -> None:
#         self.ratio = ratio
#         self.device = device

#     def transform(self, image):

#         # Perform Fourier Transform
#         freq_image = torch.fft.fftn(image, dim=(1, 2))

#         # Get the height and width of the image
#         channels, height, width = image.shape

#         # Compute the balanced mask
#         mask = self._create_balanced_mask(height, width)

#         # Apply the mask to the frequency image
#         masked_freq_image = freq_image * mask.repeat((channels, 1, 1))

#         # Perform Inverse Fourier Transform
#         masked_image = torch.fft.ifftn(masked_freq_image, dim=(1, 2)).real

#         return masked_image

#     def _create_balanced_mask(self, height, width):
#         mask = torch.ones((1, height, width), dtype=torch.complex64)
#         num_frequencies = int(np.ceil(height * width * self.ratio))
#         mask_frequencies_indices = torch.randperm(height * width)[:num_frequencies]
#         y_indices = mask_frequencies_indices // width
#         x_indices = mask_frequencies_indices % width
#         mask[:, y_indices, x_indices] = 0
#         return mask

class ZoomBlockGenerator:
    def __init__(self, ratio: float = 0.1, device: str = "cpu") -> None:
        self.zoom_ratio = ratio
        self.device = device

    def transform(self, image):

        # Get the height and width of the image
        _, height, width = image.shape

        # Compute the size of the zoomed block
        zoom_height = int(height * self.zoom_ratio)
        zoom_width = int(width * self.zoom_ratio)

        # Randomly select the starting point for the zoomed block
        start_y = torch.randint(0, height - zoom_height + 1, (1,)).item()
        start_x = torch.randint(0, width - zoom_width + 1, (1,)).item()

        # Extract the zoomed block
        zoomed_block = image[:, start_y:start_y + zoom_height, start_x:start_x + zoom_width]

        # Resize the zoomed block to the original image size
        zoomed_image = F.interpolate(zoomed_block.unsqueeze(0), size=(height, width), mode='bilinear').squeeze(0)

        return zoomed_image

# class EdgeAwareMaskGenerator:
#     def __init__(self, ratio: float = 0.3, threshold: float = 0.5, device: str = "cpu") -> None:
#         self.ratio = ratio
#         self.threshold = threshold
#         self.device = device
#         self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     def transform(self, image):
#         channels, height, width = image.shape
#         image = image

#         # Apply Sobel filters
#         grad_x = torch.nn.functional.conv2d(image.unsqueeze(0), self.sobel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
#         grad_y = torch.nn.functional.conv2d(image.unsqueeze(0), self.sobel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
#         grad_magnitude = torch.sqrt(torch.sum(grad_x**2 + grad_y**2, dim=1)).squeeze(0)

#         edge_map = (grad_magnitude > self.threshold).float()

#         # Compute patch size
#         patch_size = 8
#         while height % patch_size != 0 or width % patch_size != 0:
#             patch_size -= 1

#         # Compute edge content using convolution with a patch-sized kernel
#         patch_kernel = torch.ones((1, 1, patch_size, patch_size))
#         edge_content = torch.nn.functional.conv2d(edge_map.unsqueeze(0).unsqueeze(0), patch_kernel, stride=patch_size)

#         # Select patches to mask
#         num_patches_to_mask = int(np.ceil(edge_content.numel() * self.ratio))
#         mask_patch_indices = torch.topk(edge_content.view(-1), num_patches_to_mask).indices

#         # Create the mask
#         mask = torch.ones((1, height, width))
#         for index in mask_patch_indices:
#             start_y = (index // (width // patch_size)) * patch_size
#             start_x = (index % (width // patch_size)) * patch_size
#             mask[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = 0

#         # Repeat the mask for all channels
#         mask = mask.repeat((channels, 1, 1))

#         masked_image = image * mask
#         return masked_image

class HighFrequencyMaskGenerator:
    def __init__(self, emphasis_factor: float = 2.0, device: str = "cpu") -> None:
        self.emphasis_factor = emphasis_factor
        self.device = device

    def transform(self, image):
        image = image.to(self.device)

        # Compute the Fourier Transform
        spectral_image = torch.fft.fftn(image, dim=(1, 2))
        
        # Compute the magnitude
        magnitude = torch.abs(spectral_image)

        # Create a high-frequency emphasis mask using a radial gradient
        _, height, width = image.shape
        cy, cx = height // 2, width // 2
        y = torch.linspace(-cy, cy, height, device=self.device)
        x = torch.linspace(-cx, cx, width, device=self.device)
        y, x = torch.meshgrid(y, x, indexing='xy')  # Include the indexing argument
        radial_distance = torch.sqrt(x**2 + y**2)
        hf_mask = 1 + self.emphasis_factor * (radial_distance / radial_distance.max())

        # Apply the mask
        masked_magnitude = magnitude * hf_mask

        # Compute the phase
        phase = torch.angle(spectral_image)

        # Convert back to the complex form
        masked_spectral_image = masked_magnitude * torch.exp(1j * phase)

        # Inverse Fourier Transform
        masked_image = torch.fft.ifftn(masked_spectral_image, dim=(1, 2)).real

        return masked_image

# # Let's create a simple test script that generates a masked image and saves it to a jpg file.
# def test_mask_generator(image_path, mask_type, ratio):
#     # Create a MaskGenerator
#     if mask_type == 'spectral':
#         mask_generator = FrequencyMaskGenerator(ratio=ratio)
#     elif mask_type == 'zoom':
#         mask_generator = ZoomBlockGenerator(ratio=ratio)
#     elif mask_type == 'patch':
#         mask_generator = PatchMaskGenerator(ratio=ratio)
#     elif mask_type == 'shiftedpatch':
#         mask_generator = ShiftedPatchMaskGenerator(ratio=ratio)
#     elif mask_type == 'invblock':
#         mask_generator = InvBlockMaskGenerator(ratio=ratio)
#     elif mask_type == 'edge':
#         mask_generator = EdgeAwareMaskGenerator(ratio=ratio)
#     elif mask_type == 'highfreq':
#         mask_generator = HighFrequencyMaskGenerator(device="cpu")
#     else:
#         raise ValueError('Invalid mask_type')

#     # Load an image
#     image = Image.open(image_path)  # replace with your image file path
#     image = image.resize((224, 224))  # resize the image to match the mask size
#     original_image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0  # convert the image to a PyTorch tensor

#     # Save the original image as a PIL image
#     original_pil_image = ToPILImage()(original_image_tensor.cpu())
#     original_pil_image.save(f"samples/original_image.jpg")

#     # Generate a masked image
#     masked_image = mask_generator.transform(original_image_tensor)

#     # Convert the masked image back to a PIL image
#     pil_image = ToPILImage()(masked_image.cpu())
#     pil_image.save(f"samples/masked_{mask_type}.jpg")


# test_mask_generator(
#     image_path="/home/paperspace/Documents/chandler/Datasets/Wang_CVPR20/crn/0_real/00100001.png",
#     mask_type='spectral', 
#     ratio=0.13
#     )

class EdgeAwareMaskGenerator:
    def __init__(self, ratio: float = 0.3, threshold: float = 0.5) -> None:
        self.ratio = ratio
        self.threshold = threshold
        self.sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.sobel_y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

    def transform(self, pil_image: Image.Image) -> Image.Image:
        image = np.array(pil_image) / 255.0
        channels, height, width = image.shape
        grad_magnitude = np.zeros((height, width))

        # Apply Sobel filters to each channel
        for ch in range(channels):
            grad_x = convolve2d(image[ch], self.sobel_x, mode="same")
            grad_y = convolve2d(image[ch], self.sobel_y, mode="same")
            grad_magnitude += np.sqrt(grad_x**2 + grad_y**2)

        edge_map = (grad_magnitude > self.threshold).astype(float)

        # Compute patch size
        patch_size = 16
        while height % patch_size != 0 or width % patch_size != 0:
            patch_size -= 1

        # Compute edge content using convolution with a patch-sized kernel
        patch_kernel = np.ones((patch_size, patch_size))
        edge_content = convolve2d(edge_map, patch_kernel, mode="valid", stride=patch_size)

        # Select patches to mask
        num_patches_to_mask = int(np.ceil(edge_content.size * self.ratio))
        mask_patch_indices = np.argpartition(edge_content.flatten(), -num_patches_to_mask)[-num_patches_to_mask:]

        # Create the mask
        mask = np.ones((height, width))
        for index in mask_patch_indices:
            start_y = (index // (width // patch_size)) * patch_size
            start_x = (index % (width // patch_size)) * patch_size
            mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = 0

        # Repeat the mask for all channels
        mask = mask[np.newaxis, :, :].repeat(channels, axis=0)

        masked_image = (image * mask).clip(0, 1)
        return Image.fromarray((masked_image * 255).astype(np.uint8))

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


class MaskingTransform(torch.nn.Module):
    def __init__(self, mask_generator):
        super().__init__()
        self.mask_generator = mask_generator

    def forward(self, image):
        return self.mask_generator.transform(image)

def test_mask_generator(
    image_path, 
    mask_type,
    ratio=0.13, 
    sample_size=20
    ):

    # Create a MaskGenerator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    elif mask_type == 'zoom':
        mask_generator = ZoomBlockGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    elif mask_type == 'shiftedpatch':
        mask_generator = ShiftedPatchMaskGenerator(ratio=ratio)
    elif mask_type == 'invblock':
        mask_generator = InvBlockMaskGenerator(ratio=ratio)
    elif mask_type == 'edge':
        mask_generator = EdgeAwareMaskGenerator(ratio=ratio)
    elif mask_type == 'highfreq':
        mask_generator = HighFrequencyMaskGenerator()
    else:
        mask_generator = None

    # # Define the custom transform
    # masking_transform = MaskingTransform(mask_generator)

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

    # Display and save the image
    plt.imshow(image_to_save)
    plt.axis('off')  # Optional, to turn off axes
    plt.savefig(f"samples/masked_{mask_type}.jpg")

# Usage:
test_mask_generator(
    '../../Datasets/Wang_CVPR20/wang_et_al/validation', 
    mask_type='spectral',
    ratio=0.7
    )