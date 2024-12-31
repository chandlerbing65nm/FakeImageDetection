import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset import ForenSynths
from augment import ImageAugmentor
from utils import train_augment
from mask import FrequencyMaskGenerator, PatchMaskGenerator, PixelMaskGenerator

# ------------------------
# 1. JS Divergence and Wasserstein Helpers
# ------------------------
def kl_divergence(p, q, eps=1e-12):
    """
    Computes the Kullback–Leibler divergence KL(p || q) for each row in p, q.
    Both p and q must be > 0 and sum to 1 across features.
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=1)

def js_divergence(p, q):
    """
    Computes the Jensen–Shannon divergence for each row in p, q.
    p and q must be probability distributions (sum to 1).
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def wasserstein_distance(p, q):
    """
    Computes the Wasserstein distance between two distributions p and q.
    p and q should be probability distributions (sum to 1).
    """
    return torch.sum(torch.abs(torch.cumsum(p, dim=1) - torch.cumsum(q, dim=1)), dim=1)

def apply_mixup_cutmix_cutout_batch(images, mask_type, ratio=0.5):
    """
    Applies Mixup, CutMix, or Cutout on a batch of images using a grid-based approach
    for CutMix and Cutout.

    Args:
        images (Tensor): A batch of images of shape (B, C, H, W).
        mask_type (str): One of ['mixup', 'cutmix', 'cutout'].
        ratio (float): A value in [0, 1] * max_value_possible.
                       - Mixup: how much to blend between original and permuted images.
                       - CutMix/Cutout: ratio * (# of 16x16 grid boxes) to replace.
    
    Returns:
        Tensor: Transformed batch of shape (B, C, H, W).
    """
    # Safety clamp
    ratio = max(0.0, min(1.0, ratio))

    B, C, H, W = images.shape

    # -----------------------------------------------------------
    # MIXUP (Batch version)
    # -----------------------------------------------------------
    if mask_type == 'mixup':
        # Shuffle the batch to create a "partner" for each image
        perm = torch.randperm(B)
        # Weighted sum between each image and its shuffled partner
        images = ratio * images + (1 - ratio) * images[perm]

    # -----------------------------------------------------------
    # CUTMIX (Batch version, grid-based)
    # -----------------------------------------------------------
    elif mask_type == 'cutmix':
        # Shuffle the batch to know which other image we'll copy from
        perm = torch.randperm(B)
        images2 = images[perm]  # partner batch to copy patches from

        patch_size = 16
        num_x_boxes = W // patch_size
        num_y_boxes = H // patch_size
        total_boxes = num_x_boxes * num_y_boxes
        n_to_select = int(ratio * total_boxes)

        # For each image in the batch, randomly pick boxes from images2
        for i in range(B):
            all_boxes = list(range(total_boxes))
            random.shuffle(all_boxes)
            selected_boxes = all_boxes[:n_to_select]

            for box_idx in selected_boxes:
                row = box_idx // num_x_boxes
                col = box_idx % num_x_boxes

                x_start = col * patch_size
                y_start = row * patch_size
                x_end = x_start + patch_size
                y_end = y_start + patch_size

                # Replace region i with region from images2
                images[i, :, y_start:y_end, x_start:x_end] = \
                    images2[i, :, y_start:y_end, x_start:x_end]

    # -----------------------------------------------------------
    # CUTOUT (Batch version, grid-based)
    # -----------------------------------------------------------
    elif mask_type == 'cutout':
        patch_size = 16
        num_x_boxes = W // patch_size
        num_y_boxes = H // patch_size
        total_boxes = num_x_boxes * num_y_boxes
        n_to_select = int(ratio * total_boxes)

        # For each image in the batch, zero out random boxes
        for i in range(B):
            all_boxes = list(range(total_boxes))
            random.shuffle(all_boxes)
            selected_boxes = all_boxes[:n_to_select]

            for box_idx in selected_boxes:
                row = box_idx // num_x_boxes
                col = box_idx % num_x_boxes

                x_start = col * patch_size
                y_start = row * patch_size
                x_end = x_start + patch_size
                y_end = y_start + patch_size

                images[i, :, y_start:y_end, x_start:x_end] = 0.0

    return images

# -------------------------------------------------------
# 2. Main function to compute JS Divergence or Wasserstein Distance
# -------------------------------------------------------
def compute_distance_resnet50(unmasked_loader, masked_loader, metric='js', device='cuda', args=None):
    """
    Given two DataLoaders:
      - unmasked_loader: DataLoader returning unmasked images
      - masked_loader:   DataLoader returning masked images
    Computes either:
      - JS Divergence (if metric == 'js')
      - Wasserstein Distance (if metric == 'wasserstein')
    """

    # ------------------------------
    # Load a pretrained ResNet-50
    # ------------------------------
    resnet = models.resnet50(pretrained=True)
    # Remove the final FC layer, keep up to (and including) global average pool
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    distance_values = []

    # ---------------------------------------------
    # Evaluate both DataLoaders in parallel (zip)
    # ---------------------------------------------
    num_batches = len(unmasked_loader)  # or len(masked_loader), if the same

    for (unmasked_imgs, _), (masked_imgs, _) in tqdm(zip(unmasked_loader, masked_loader), total=num_batches):
        unmasked_imgs = unmasked_imgs.to(device, non_blocking=True)
        masked_imgs   = masked_imgs.to(device, non_blocking=True)

        # import ipdb; ipdb.set_trace() 
        # print(masked_imgs.shape)

        if args.mask_type in ['mixup', 'cutmix', 'cutout']:
            masked_imgs = apply_mixup_cutmix_cutout_batch(masked_imgs, args.mask_type, args.ratio)

        # -----------------------------------
        # Extract features (no grad needed)
        # -----------------------------------
        with torch.no_grad():
            unmasked_feats = feature_extractor(unmasked_imgs)  # shape [B, 2048, 1, 1]
            masked_feats   = feature_extractor(masked_imgs)

        # Flatten to [B, 2048]
        unmasked_feats = unmasked_feats.view(unmasked_feats.size(0), -1)
        masked_feats   = masked_feats.view(masked_feats.size(0), -1)

        # ---------------------------------------
        # L1-normalize along feature dimension
        # so that each row sums to 1
        # ---------------------------------------
        unmasked_feats = F.normalize(unmasked_feats, p=1, dim=1)
        masked_feats   = F.normalize(masked_feats,   p=1, dim=1)

        # ------------------------------
        # Compute per-sample distance (JS or Wasserstein)
        # ------------------------------
        if metric == 'js':
            batch_distance = js_divergence(unmasked_feats, masked_feats)  # shape [B]
        elif metric == 'wasserstein':
            batch_distance = wasserstein_distance(unmasked_feats, masked_feats)  # shape [B]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        distance_values.append(batch_distance.mean().item())

    # Final average across all batches
    return float(np.mean(distance_values))


# -----------------------------
# 3. Example usage in main code
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Distance Metrics Example")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--band', default='low+mid', type=str, 
                        choices=['all', 'low', 'mid', 'high'])
    parser.add_argument('--mask_type', default='fourier', 
                        choices=[
                            'fourier', 'cosine', 'wavelet', 'pixel', 'patch', 'rotate', 'translate', 'shear', 'scale',
                            'mixup', 'cutmix', 'cutout'])
    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN50', 'RN50_mod', 'CLIP_vitl14'
        ],
        help='Type of model to use; includes ResNet'
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=15)
    parser.add_argument('--metric', type=str, default='js', 
                        choices=['js', 'wasserstein'],
                        help="Metric to compute: 'js' for JS Divergence, 'wasserstein' for Wasserstein Distance")
    args = parser.parse_args()

    # --------------
    # Random seeds
    # --------------
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    np.random.seed(44)
    random.seed(44)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------
    # Define your mask generator
    # ------------------------------------
    if args.mask_type in ['rotate', 'translate', 'shear', 'scale']:
        mask_generator = None
        if args.mask_type == 'rotate':
            args.ratio = (args.ratio / 100) * 180
        elif args.mask_type == 'translate':
            args.ratio = args.ratio / 100
        elif args.mask_type == 'shear':
            args.ratio =  (args.ratio / 100) * 45
        elif args.mask_type == 'scale':
            args.ratio = args.ratio / 100
    elif args.mask_type in ['mixup', 'cutmix', 'cutout']:
        mask_generator = None
        args.ratio = args.ratio / 100
    else:
        if args.mask_type in ['fourier', 'cosine', 'wavelet']:
            mask_generator = FrequencyMaskGenerator(
                ratio=args.ratio / 100.0,
                band=args.band,
                transform_type=args.mask_type
            )
        elif args.mask_type == 'pixel':
            mask_generator = PixelMaskGenerator(
                ratio=args.ratio / 100.0,
            )
        elif args.mask_type == 'patch':
            mask_generator = PatchMaskGenerator(
                ratio=args.ratio / 100.0,
            )
        else:
            raise ValueError(f"Unsupported mask type: {args.mask_type}")

    # ------------------------------------
    # Example train options & transform
    # ------------------------------------
    train_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,
        'blur_sig': [0.0, 3.0],
        'jpg_prob': 0.1,
        'jpg_method': ['cv2', 'pil'],
        'jpg_qual': [30, 100]
    }

    # Transform for unmasked images (no mask generator)
    train_transform_unmasked = train_augment(
        ImageAugmentor(train_opt), 
        mask_generator=None,  
        args=args
    )

    # Transform for masked images (use mask generator)
    train_transform_masked = train_augment(
        ImageAugmentor(train_opt), 
        mask_generator=mask_generator,  
        args=args
    )

    # ------------------------------------------------
    # Create two datasets for unmasked vs. masked
    # ------------------------------------------------
    dataset_unmasked = ForenSynths(
        '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/validation',
        transform=train_transform_unmasked
    )
    dataset_masked = ForenSynths(
        '/mnt/SCRATCH/chadolor/Datasets/Wang_CVPR2020/validation',
        transform=train_transform_masked
    )

    # ---------------------------
    # Create DataLoaders (no sampler)
    # ---------------------------
    train_loader_unmasked = DataLoader(
        dataset_unmasked,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    train_loader_masked = DataLoader(
        dataset_masked,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # -----------------------------------------
    # Compute selected metric (JS or Wasserstein)
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_value = compute_distance_resnet50(
        unmasked_loader=train_loader_unmasked,
        masked_loader=train_loader_masked,
        metric=args.metric,
        device=device,
        args=args
    )

    print(f"Average {args.metric.capitalize()} Distance (masked vs. unmasked) = {distance_value:.4f}")
