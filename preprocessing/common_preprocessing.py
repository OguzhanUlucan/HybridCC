"""
Shared utilities for the preprocessing.
This step is used to save images on disk. You can edit the
resolution, bit depth, etc. accordingly
"""

import numpy as np
import cv2
import tifffile
from typing import Tuple

def resize_to_1080(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Resize image to 1080 height while preserving aspect ratio.
    The model process 384x384 images, so you can edit the resolution accordingly.
    """
    target_height = 1080
    h, w = image.shape[:2]

    scale = target_height / h
    new_width = int(round(w * scale))
 
    if scale < 1:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4

    resized = cv2.resize(image, (new_width, target_height), interpolation=interp)

    return resized, scale

def save_as_tiff(image: np.ndarray, output_path: str):
    """
    Save normalized [0,1] image as 8-bit TIFF.
    In case you want to quantize differently please modify this step.
    """
    image_8bit = (image * 255).astype(np.uint8) ###
    tifffile.imwrite(output_path, image_8bit, photometric='rgb')

def normalize_illuminant(illuminant: np.ndarray) -> np.ndarray:
    """
    Normalize illuminant to unit vector [R, G, B].
    """
    illuminant = np.array(illuminant).astype(np.float64)
    norm = np.linalg.norm(illuminant)
    if norm > 0:
        illuminant = illuminant / norm
    return illuminant

def save_wp_file(wp_path: str, illuminant: np.ndarray):
    """
    Save illuminant as .wp file.
    Format: "R   G   B   "
    """
    r, g, b = illuminant
    with open(wp_path, 'w') as f:
        f.write(f"{r}   {g}   {b}   ")
