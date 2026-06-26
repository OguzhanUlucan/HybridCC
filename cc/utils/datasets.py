"""
Data augmentation.
Augmentation operations (applied to non-original crops only):
  1. Random crop (random size and position)
  2. Random rotation 
  3. Resize to crop_size
  4. Random horizontal/vertical flip
  5. Intensity gain with overexposure masking
  6. Gaussian blur / Gaussian noise
  7. Color augmentation (per-channel scale) + ground truth transform
"""

import os
import csv
import random
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import imageio.v2 as imageio
from torch.utils.data import Dataset

def load_image(path):
    """
    Load an image from path.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
    else:
        arr = imageio.imread(path)
 
    if arr.shape[-1] == 4: 
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.integer):
        maxv = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / float(maxv)
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

def collate_fn_gpu(batch):
    """
    Stack individual samples into batch tensors.
    Returns (images, illuminants, is_original_flags).
    """
    images = torch.stack([b['image'] for b in batch], dim=0)
    lgt = torch.stack([b['illuminant'] for b in batch], dim=0)
    is_orig = torch.tensor([b['is_original'] for b in batch], dtype=torch.bool)
    return images, lgt, is_orig

def _find_image_with_extensions(base_path: str, root=None):
    """Find an image file trying multiple extensions (png, jpg, tif, etc.)."""
    extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.tif', '.TIF', '.tiff', '.TIFF', '.bmp', '.BMP'] # you can add or remove according to your needs
    base_path = os.path.expanduser(str(base_path))
    if root and not os.path.isabs(base_path):
        base_path = os.path.normpath(os.path.join(root, base_path))
    if os.path.exists(base_path):
        return base_path

    base_without_ext = os.path.splitext(base_path)[0]
    for ext in extensions:
        candidate = base_without_ext + ext
        if os.path.exists(candidate):
            return candidate

    parent = os.path.dirname(base_without_ext)
    filename = os.path.basename(base_without_ext)
    if os.path.isdir(parent):
        for item in os.listdir(parent):
            item_without_ext = os.path.splitext(item)[0]
            if item_without_ext.lower() == filename.lower():
                found = os.path.join(parent, item)
                if os.path.isfile(found):
                    return found

    raise FileNotFoundError(f"Image not found: {base_path}")


def _get_num(row, keys):
    """Extract a numeric value from a CSV row, trying multiple column names."""
    for k in keys:
        if k in row and row[k] is not None:
            s = str(row[k]).strip()
            if s != "":
                return float(s)
    raise KeyError(f"None of {keys} found in CSV")


class CCMinimalDataset(Dataset):
    """
    Dataset that preloads all images into RAM at initialization.
    
    Each image can produce multiple variants:
    - 1 original (if include_original=True)
    - N random crops (crops_per_image)
    
    The actual cropping/augmentation happens later on GPU (GPUAugmentation).
    Here we just return the full resized image and a flag indicating
    whether this sample is an original or a crop variant.
    
    Returns dict:
        'image':       [C, load_size, load_size]  float32, [0,1]
        'illuminant':  [3]  float32, unit-norm
        'is_original': bool   (True = skip augmentation)
        'var_idx':     int    (which variant of this image)
        'img_idx':     int    (which image in the dataset)
    """

    def __init__(self, csv_file, root=None, *,
                 load_size=384,
                 crops_per_image=1,
                 include_original=True):
        super().__init__()
        self.root = root
        self.load_size = int(load_size)
        self.crops_per_image = int(crops_per_image)
        self.include_original = bool(include_original)

        # extract image paths and ground truth illuminants
        paths = []
        illuminants = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                # Find path column
                path = (row.get('path') or row.get('image')
                        or row.get('img') or row.get('filename'))
                if path is None:
                    raise KeyError(
                        f"Row {row_idx}: CSV needs 'path','image','img', or 'filename'")
                try:
                    path = _find_image_with_extensions(path, self.root)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Row {row_idx+1} in CSV: {e}") from e

                # Find illuminant columns. While creating the csv we used Lr, Lg, Lb, and R, G, B
                # but you can edit according to your needs.
                # Below you can find an example.
                Lr = _get_num(row, ['Lr','lr','R','r','wr','Wr','white_r','mean_r','Mean_r'])
                Lg = _get_num(row, ['Lg','lg','G','g','wg','Wg','white_g','mean_g','Mean_g'])
                Lb = _get_num(row, ['Lb','lb','B','b','wb','Wb','white_b','mean_b','Mean_b'])

                # Normalize to unit vector
                L = torch.tensor([Lr, Lg, Lb], dtype=torch.float32)
                L = L / (L.norm() + 1e-8)

                paths.append(path)
                illuminants.append(L)

        # Preload all images into RAM
        s = self.load_size
        n = len(paths)
        print(f"Loading {n} images to {s}x{s} ...")

        def _load_and_resize(p):
            img = load_image(p).to(torch.float32)
            _, H, W = img.shape
            if H != s or W != s:
                mode = 'area' if (H >= s and W >= s) else 'bilinear'
                img = F.interpolate(
                    img.unsqueeze(0), size=(s, s), mode=mode,
                    align_corners=False if mode == 'bilinear' else None
                ).squeeze(0)
            return img.clamp(0.0, 1.0).contiguous()

        t0 = time.time()
        self.image_cache = [None] * n
        self.illuminant_cache = list(illuminants)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_load_and_resize, paths[i]): i for i in range(n)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                self.image_cache[i] = fut.result()
                done += 1
                if done % 500 == 0 or done == n:
                    ram_mb = done * s * s * 3 * 4 / (1024 ** 2)
                    print(f"  [{done}/{n}] loaded  (~{ram_mb:.0f} MB)")

        print(f"Preload is done. {n} images in {time.time() - t0:.1f}s")

        self.N = n
        # Each image produces: 1 original + N crops
        self.per_image = (1 if self.include_original else 0) + self.crops_per_image
        assert self.per_image > 0

    def __len__(self):
        return self.N * self.per_image

    def __getitem__(self, idx):
        img_idx = idx // self.per_image     # which image
        var_idx = idx % self.per_image      # which variant (0 = original if included)

        img = self.image_cache[img_idx]
        L = self.illuminant_cache[img_idx]
        is_original = (self.include_original and var_idx == 0)

        return {
            'image': img,
            'illuminant': L,
            'is_original': is_original,
            'var_idx': var_idx,
            'img_idx': img_idx,
        }


class GPUAugmentation(nn.Module):
    """
    Batch-wise GPU augmentation.
    
    Only augments non-original samples (is_original=False).
    Original samples pass through untouched.
    """

    def __init__(self, *,
                 crop_size=384,
                 scale_range=(0.1, 1.0),
                 augmentation_angle=360.0,
                 hflip_p=0.5,
                 vflip_p=0.5,
                 gain_range=(0.8, 1.2),
                 gain_prob=0.5,
                 color_aug_strength=0.4):
        super().__init__()
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.augmentation_angle = augmentation_angle
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.gain_range = gain_range
        self.gain_prob = gain_prob
        self.color_aug_strength = color_aug_strength

    @torch.no_grad()
    def forward(self, images, illuminants, is_original):
        B, C, H, W = images.shape
        device = images.device

        # Only augment non-original samples
        aug_mask = ~is_original
        n_aug = aug_mask.sum().item()
        if n_aug == 0:
            return images, illuminants

        x = images[aug_mask]
        L = illuminants[aug_mask]
        N = x.shape[0]

        # 1-3. Random crop, rotate, and resize (per-sample, different params each)
        x = self._per_sample_crop_rotate_resize(x)

        # 4. Random horizontal and vertical flips
        hflip = torch.rand(N, device=device) < self.hflip_p
        vflip = torch.rand(N, device=device) < self.vflip_p
        if hflip.any():
            x[hflip] = x[hflip].flip(dims=(3,))    # flip width
        if vflip.any():
            x[vflip] = x[vflip].flip(dims=(2,))    # flip height

        # 5. Random intensity gain with overexposure masking
        x = self._batch_gain_masking(x, N, device)

        # 6. Random blur and noise
        x = self._batch_blur_noise(x, N, device)

        # 7. Color augmentation (also transforms ground truth illuminant)
        x, L = self._batch_color_aug(x, L, N, device)

        x = x.clamp(0.0, 1.0)

        out_images = images.clone()
        out_illum = illuminants.clone()
        out_images[aug_mask] = x
        out_illum[aug_mask] = L

        return out_images, out_illum


    def _per_sample_crop_rotate_resize(self, x):
        """
        Per-sample: 
        1. Random crop: random scale and position
        2. Random rotation
        3. Resize
        """
        N, C, H, W = x.shape
        device = x.device
        min_scale, max_scale = self.scale_range
        min_dim = min(H, W)
        target = self.crop_size

        results = torch.empty(N, C, target, target, device=device, dtype=x.dtype)

        for i in range(N):
            img = x[i]

            # Step 1: Random crop
            log_scale = torch.rand(1, device=device).item() * math.log(max_scale / min_scale)
            scale = min_scale * math.exp(log_scale)
            s = max(10, min(int(round(min_dim * scale)), min_dim))

            # Random position within image
            max_y = max(0, H - s)
            max_x = max(0, W - s)
            y0 = torch.randint(0, max_y + 1, (1,), device=device).item() if max_y > 0 else 0
            x0 = torch.randint(0, max_x + 1, (1,), device=device).item() if max_x > 0 else 0
            img = img[:, y0:y0+s, x0:x0+s]

            # Step 2: Random rotation + center crop
            _, rH, rW = img.shape
            angle = (torch.rand(1, device=device).item() - 0.5) * self.augmentation_angle
            angle_rad = angle * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Build matrix for rotation
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0]
            ], dtype=img.dtype, device=device).unsqueeze(0)

            # Apply rotation
            img_b = img.unsqueeze(0)
            grid = F.affine_grid(theta, img_b.size(), align_corners=False)
            rotated = F.grid_sample(img_b, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False).squeeze(0)

            # Center crop to 70% to remove zero-padded rotation borders
            crop_s = int(min(rH, rW) * 0.7)
            cy, cx = rH // 2, rW // 2
            half = crop_s // 2
            rotated = rotated[:, max(0, cy-half):min(rH, cy+half),
                                max(0, cx-half):min(rW, cx+half)]

            # Step 3: Resize to crop_size 
            _, fH, fW = rotated.shape
            if fH != target or fW != target:
                mode = 'area' if (fH >= target and fW >= target) else 'bilinear'
                rotated = F.interpolate(
                    rotated.unsqueeze(0), size=(target, target), mode=mode,
                    align_corners=False if mode == 'bilinear' else None
                ).squeeze(0)

            results[i] = rotated

        return results

    def _mask_overexposure_batch(self, img, threshold=0.99):
        """Zero out pixels where any channel exceeds threshold (saturated)."""
        mask = (img >= threshold).any(dim=1, keepdim=True)
        return torch.where(mask.expand_as(img), torch.zeros_like(img), img)

    def _batch_gain_masking(self, x, N, device):
        """
        Random intensity gain: multiply by random factor.
        """
        apply = torch.rand(N, device=device) < self.gain_prob
        if not apply.any():
            return x

        sub = x[apply]
        sub = self._mask_overexposure_batch(sub)
        n_sub = sub.shape[0]
        gain = torch.empty(n_sub, 1, 1, 1, device=device).uniform_(
            self.gain_range[0], self.gain_range[1])
        sub = (sub * gain).clamp(0.0, 1.0)
        sub = self._mask_overexposure_batch(sub)
        x[apply] = sub
        return x

    def _batch_blur_noise(self, x, N, device):
        """
        Per-sample: 50% chance of Gaussian blur, 50% chance of Gaussian noise.
        Each sample gets independent random parameters.
        """
        for i in range(N):
            # Random blur
            if torch.rand(1, device=device).item() < 0.5:
                k = random.choice((3, 5, 7))
                sigma = random.uniform(0.05, 0.8)
                x[i:i+1] = self._gaussian_blur_batch(x[i:i+1], k, sigma)

            # Random noise
            if torch.rand(1, device=device).item() < 0.5:
                std = random.uniform(0.001, 0.015)
                if std > 0.0:
                    x[i] = (x[i] + torch.randn_like(x[i]) * std).clamp_(0.0, 1.0)

        return x

    @staticmethod
    def _gaussian_blur_batch(img, kernel_size, sigma):
        C = img.shape[1]
        k = kernel_size
        coords = torch.arange(k, dtype=img.dtype, device=img.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        pad = k // 2

        kh = g.view(1, 1, k, 1).expand(C, 1, k, 1)
        kw = g.view(1, 1, 1, k).expand(C, 1, 1, k)
        img = F.conv2d(img, kh, padding=(pad, 0), groups=C)
        img = F.conv2d(img, kw, padding=(0, pad), groups=C)
        return img

    def _batch_color_aug(self, x, L, N, device):
        """
        Per-channel color jitter: multiply each RGB channel by a random factor.
        """
        strength = self.color_aug_strength
        color_aug = 1.0 + torch.rand(N, 3, 1, 1, device=device) * strength \
                        - 0.5 * strength

        x = (x * color_aug).clamp(0.0, 1.0)

        ca = color_aug.squeeze(-1).squeeze(-1)
        L = L * ca
        L = L / (L.norm(dim=1, keepdim=True) + 1e-8)

        return x, L
