import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import torch.fft as fft
import torch.nn.functional as F

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


class BalancedSpectralMaskGenerator:
    def __init__(self, ratio: float = 0.1, device: str = "cpu") -> None:
        self.ratio = ratio
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

        # Convert the image to grayscale using standard weights
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=self.device)
        grayscale_image = torch.einsum("chw,c->hw", image, grayscale_weights).unsqueeze(0)

        # Apply Fourier transformation to decompose the grayscale image into spectral bands
        x_mul = torch.fft.fftn(grayscale_image, dim=(1, 2))

        # Compute the image contents in each band and normalize
        I_x = torch.sum(torch.sum(torch.abs(x_mul), dim=1), dim=1)
        I_prime_x = I_x / torch.sum(I_x)

        # Generate the balanced spectral mask
        m_spectral = torch.bernoulli(1 - I_prime_x * self.ratio).to(self.device)

        # Initialize an empty tensor to hold the masked image
        num_channels, height, width = image.shape
        masked_image = torch.zeros_like(image)

        # Loop over each channel and apply the computed spectral mask
        for c in range(num_channels):
            channel_image = image[c].unsqueeze(0)
            spectral_image = torch.fft.fftn(channel_image, dim=(1, 2))
            masked_spectral_image = spectral_image * m_spectral.view(-1, 1, 1)
            masked_channel_image = torch.fft.ifftn(masked_spectral_image, dim=(1, 2)).real
            masked_image[c] = masked_channel_image.squeeze()

        return masked_image

class ZoomBlockGenerator:
    def __init__(self, ratio: float = 0.1, device: str = "cpu") -> None:
        self.zoom_ratio = ratio
        self.device = device

    def transform(self, image):
        # Move the image to the same device as the mask
        image = image.to(self.device)

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

# Let's create a simple test script that generates a masked image and saves it to a jpg file.
def test_mask_generator():
    # Create a MaskGenerator
    mask_generator = ZoomBlockGenerator(ratio=0.5, device="cpu")

    # Load an image
    image = Image.open("/home/paperspace/Documents/chandler/Datasets/Wang_CVPR20/crn/0_real/00100001.png")  # replace "test.jpg" with your image file path
    image = image.resize((224, 224))  # resize the image to match the mask size
    image = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0  # convert the image to a PyTorch tensor

    # Generate a masked image
    masked_image = mask_generator.transform(image)

    # Convert the masked image back to a PIL image
    pil_image = ToPILImage()(masked_image.cpu())

    # Save the PIL image to a jpg file
    pil_image.save("samples/masked_zoom.jpg")

test_mask_generator()