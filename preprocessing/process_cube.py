"""
Preprocess Cube+ dataset

Expected folder structure:
    cube+/
        PNG/
            1.PNG
            2.PNG
            ...
        cube+_gt.txt (or similar)
"""

import os
import numpy as np
import cv2
import argparse
from typing import List
from tqdm import tqdm
import yaml

from common_preprocessing import resize_to_1080, save_as_tiff, normalize_illuminant, save_wp_file

# Cube+ dataset constants
CUBE_BLACK_LEVEL = 2048

# Color checker mask region 
CC_MASK_ROW_START = 1050
CC_MASK_COL_START = 2050

def load_ground_truth(gt_path: str) -> List[np.ndarray]:
    """
    Load Cube+ ground truth file.
    Format: R G B
    Line N corresponds to image N.png

    Returns:
        List of [R, G, B] arrays
    """
    ground_truths = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                ground_truths.append(np.array([r, g, b]))
    return ground_truths

def process_single_image(
    input_path: str,
    output_path: str,
    black_level: float = CUBE_BLACK_LEVEL,
    debug: bool = False,
) -> bool:

    try:
        img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"Error: Could not load {input_path}")
            return False

        if len(img_bgr.shape) == 3 and img_bgr.shape[2] >= 3:
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = img_bgr

        original_dtype = image.dtype

        if debug:
            print(f"\n  DEBUG: Input: {input_path}")
            print(f"    Shape: {image.shape}, dtype: {original_dtype}")
            print(f"    Range: {image.min()} - {image.max()}")

        saturation_level = float(image.max()) - 2

        if debug:
            print(f"    Saturation level (max-2): {saturation_level}")
            print(f"    Black level: {black_level}")

        image = image.astype(np.float64)
        image = image - black_level
        image = np.clip(image, 0, None)

        if debug:
            print(f"    After black level subtraction: {image.min():.2f} - {image.max():.2f}")

        sat_threshold = saturation_level - black_level
        saturation_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        for ch in range(3):
            saturation_mask = saturation_mask | (image[:, :, ch] >= sat_threshold)

        if debug:
            num_saturated = saturation_mask.sum()
            print(f"    Saturation threshold: {sat_threshold}")
            print(f"    Saturated pixels: {num_saturated} ({100*num_saturated/saturation_mask.size:.2f}%)")

        h, w = image.shape[:2]
        if CC_MASK_ROW_START < h and CC_MASK_COL_START < w:
            saturation_mask[CC_MASK_ROW_START:, CC_MASK_COL_START:] = True

        for ch in range(3):
            channel = image[:, :, ch]
            channel[saturation_mask] = 0
            image[:, :, ch] = channel

        if debug:
            print(f"    After masking: {image.min():.2f} - {image.max():.2f}")

        max_possible = saturation_level - black_level
        if max_possible > 0:
            image = image / max_possible
        image = np.clip(image, 0, 1)

        if debug:
            print(f"    After normalization: {image.min():.4f} - {image.max():.4f}")

        image, scale = resize_to_1080(image)

        if debug:
            print(f"    After resize: {image.shape}, range: {image.min():.4f} - {image.max():.4f}")

        save_as_tiff(image, output_path)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_cube_plus_dataset(
    input_dir: str,
    output_dir: str,
    gt_path: str,
):
    """
    Process entire Cube+ dataset.

    Args:
        input_dir: Directory containing PNG images (1.PNG, 2.PNG, ...)
        output_dir: Output directory for all_images/
        gt_path: Path to ground truth file (cube+_gt.txt)
    """
    print("=" * 60)
    print("Process Cube+")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    ground_truths = load_ground_truth(gt_path)
    print(f"Ground truths loaded: {len(ground_truths)}")

    all_images_dir = os.path.join(output_dir, 'all_images')
    os.makedirs(all_images_dir, exist_ok=True)

    successful = 0
    for idx in tqdm(range(len(ground_truths)), desc="Processing"):
        img_num = idx + 1  # Images are 1-indexed
        png_candidates = [
            f"{img_num}.PNG",
            f"{img_num}.png",
            f"{img_num:04d}.PNG",
            f"{img_num:04d}.png",
        ]

        input_path = None
        for candidate in png_candidates:
            test_path = os.path.join(input_dir, candidate)
            if os.path.exists(test_path):
                input_path = test_path
                break

        if input_path is None:
            print(f"\n  Image not found for index {img_num}!")
            continue

        output_filename = f"cube_plus_{img_num:04d}.tiff"
        output_path = os.path.join(all_images_dir, output_filename)

        debug_this = (idx == 0)  # Debug first image
        success = process_single_image(
            input_path=input_path,
            output_path=output_path,
            debug=debug_this,
        )

        if success:
            successful += 1

            # Normalize and save ground truth
            gt = ground_truths[idx]
            gt_normalized = normalize_illuminant(gt)

            # Save .wp file
            wp_filename = f"cube_plus_{img_num:04d}.wp"
            wp_path = os.path.join(all_images_dir, wp_filename)
            save_wp_file(wp_path, gt_normalized)

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Process Cube+ dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using config file:
    python process_cube.py --config cube.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--gt', '-g',
        type=str,
        default=None,
        help='Path to ground truth file'
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        input_dir = config.get('input', args.input)
        output_dir = config.get('output', args.output)
        gt_path = config.get('gt', args.gt)
    else:
        input_dir = args.input
        output_dir = args.output
        gt_path = args.gt

    if not input_dir or not output_dir or not gt_path:
        parser.error("--input, --output, and --gt are required")

    process_cube_plus_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        gt_path=gt_path,
    )


if __name__ == '__main__':
    main()
