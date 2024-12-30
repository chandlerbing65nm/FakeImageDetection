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
# 1. JS Divergence Helpers
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

# -------------------------------------------------------
# 2. Main function to compute JS divergence across a set
# -------------------------------------------------------
def compute_js_divergence_resnet50(unmasked_loader, masked_loader, device='cuda'):
    """
    Given two DataLoaders:
      - unmasked_loader: DataLoader returning unmasked images
      - masked_loader:   DataLoader returning masked images
    Both should yield the same images in the same order, except one is masked.
    
    1) Extract features from each (via pretrained ResNet-50).
    2) Normalize the features along dim=1 to treat them as distributions.
    3) Compute the average JS divergence across the entire dataset.
    """

    # ------------------------------
    # Load a pretrained ResNet-50
    # ------------------------------
    resnet = models.resnet50(pretrained=True)
    # Remove the final FC layer, keep up to (and including) global average pool
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    js_values = []

    # ---------------------------------------------
    # Evaluate both DataLoaders in parallel (zip)
    # ---------------------------------------------
    num_batches = len(unmasked_loader)  # or len(masked_loader), if the same

    for (unmasked_imgs, _), (masked_imgs, _) in tqdm(zip(unmasked_loader, masked_loader), total=num_batches):
        unmasked_imgs = unmasked_imgs.to(device, non_blocking=True)
        masked_imgs   = masked_imgs.to(device, non_blocking=True)

        # import ipdb; ipdb.set_trace() 
        # print(masked_imgs.shape)

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
        # Compute per-sample JS Divergence
        # ------------------------------
        batch_js = js_divergence(unmasked_feats, masked_feats)  # shape [B]
        js_values.append(batch_js.mean().item())

    # Final average across all batches
    return float(np.mean(js_values))


# -----------------------------
# 3. Example usage in main code
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute JS Divergence example")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--band', default='low+mid', type=str, 
                        choices=['all', 'low', 'mid', 'high'])
    parser.add_argument('--mask_type', default='patch', 
                        choices=['fourier', 'cosine', 'wavelet', 'pixel', 'patch'])
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
    # Finally compute JS divergence
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    js_value = compute_js_divergence_resnet50(
        unmasked_loader=train_loader_unmasked,
        masked_loader=train_loader_masked,
        device=device
    )

    print(f"Average JS Divergence (masked vs. unmasked) = {js_value:.4f}")
