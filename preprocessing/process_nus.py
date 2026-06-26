"""
Preprocess NUS-8 dataset 

Expected folder structure:
    nus/
        <CameraName>/
            <CameraName>_gt.mat
            Images/ (or images/)
                *.PNG
"""

import os
import numpy as np
import scipy.io as sio
import cv2
import argparse
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm
import yaml

from common_preprocessing import resize_to_1080, save_as_tiff, normalize_illuminant, save_wp_file


def load_ground_truth_mat(mat_path: str) -> Dict:
    """
    Load NUS8 ground truth .mat file.
    
    Returns:
        Dictionary with:
        - black_level: black level to subtract
        - saturation_level: saturation point
        - all_image_names: list of image names
        - groundtruth_illuminants: Nx3 array of illuminants
        - CC_coords: Nx4 array of color checker coordinates
    """
    mat_data = sio.loadmat(mat_path)
    
    result = {
        'black_level': int(mat_data['darkness_level'][0, 0]),
        'saturation_level': int(mat_data['saturation_level'][0, 0]),
        'CC_coords': mat_data['CC_coords'],
        'groundtruth_illuminants': mat_data['groundtruth_illuminants'],
    }
    
    names_raw = mat_data['all_image_names']
    result['all_image_names'] = [str(names_raw[i, 0][0]) for i in range(names_raw.shape[0])]
    
    return result

def mask_color_checker(image: np.ndarray, cc_coords: np.ndarray,
                       fill_value: float = 0.0) -> np.ndarray:
    """
    Mask the color checker region.
    """
    x1, x2, y1, y2 = cc_coords
    
    row1, row2 = int(x1), int(x2)
    col1, col2 = int(y1), int(y2)
    
    if row1 > row2:
        row1, row2 = row2, row1
    if col1 > col2:
        col1, col2 = col2, col1
    
    h, w = image.shape[:2]
    row1 = max(0, min(row1, h - 1))
    row2 = max(0, min(row2, h))
    col1 = max(0, min(col1, w - 1))
    col2 = max(0, min(col2, w))
    
    if row2 > row1 and col2 > col1:
        image[row1:row2, col1:col2] = fill_value
    
    return image

def process_single_image(
    input_path: str,
    output_path: str,
    black_level: float,
    saturation: float,
    cc_coords: Optional[np.ndarray] = None,
    mask_cc: bool = True,
    debug: bool = False,
) -> bool:

    try:
        img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"Could not load {input_path}!")
            return False
        
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] >= 3:
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = img_bgr
        
        original_dtype = image.dtype
        
        if debug:
            print(f"\n  DEBUG: Input image: {input_path}")
            print(f"    Shape: {image.shape}, dtype: {original_dtype}")
            print(f"    Value range: {image.min()} - {image.max()}")
            print(f"    Black level: {black_level}, Saturation: {saturation}")
        
        image = image.astype(np.float64)
        image = image - black_level
        image = np.clip(image, 0, None)
        
        if debug:
            print(f"    After black level subtraction: {image.min():.4f} - {image.max():.4f}")
        
        max_possible = saturation - black_level
        if max_possible > 0:
            image = image / max_possible
        image = np.clip(image, 0, 1)

        if mask_cc and cc_coords is not None:
            image = mask_color_checker(image, cc_coords)
            if debug:
                print(f"    Color checker masked at coords: {cc_coords}")
        
        image, scale = resize_to_1080(image)
        
        if debug:
            print(f"    After resize: shape={image.shape}, range={image.min():.4f} - {image.max():.4f}")
        
        save_as_tiff(image, output_path)
        
        if debug:
            print(f"    Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_camera_folder(
    camera_dir: str,
    output_dir: str,
    camera_name: str,
    mask_cc: bool = True,
) -> List[Dict]:
    """
    Process all images from a single camera folder.
    
    Args:
        camera_dir: Camera directory (e.g., nus/Canon1DsMkIII/)
        output_dir: Output directory (all_images folder)
        camera_name: Camera name
        mask_cc: Whether to mask the color checker region
        
    Returns:
        List of processed image info dicts
    """
    mat_files = [f for f in os.listdir(camera_dir) if f.endswith('_gt.mat')]
    if not mat_files:
        print(f"No *_gt.mat file found in {camera_dir}")
        return []
    
    mat_path = os.path.join(camera_dir, mat_files[0])
    
    gt_data = load_ground_truth_mat(mat_path)
    black_level = gt_data['black_level']
    saturation = gt_data['saturation_level']
    image_names = gt_data['all_image_names']
    illuminants = gt_data['groundtruth_illuminants']
    cc_coords_all = gt_data['CC_coords']
    
    print(f"  Camera: {camera_name}")
    print(f"  Black level: {black_level}")
    print(f"  Saturation level: {saturation}")
    print(f"  Number of images: {len(image_names)}")
    
    images_dir = os.path.join(camera_dir, 'Images')
    if not os.path.exists(images_dir):
        images_dir = os.path.join(camera_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Images folder not found in {camera_dir}")
        return []
    
    gt_output = []
    successful = 0
    
    for idx, img_name in enumerate(tqdm(image_names, desc=f"  Processing")):
        png_candidates = [
            f"{img_name}.PNG",
            f"{img_name}.png",
            f"{img_name}.tif",
            f"{img_name}.tiff",
        ]
        
        input_path = None
        for candidate in png_candidates:
            test_path = os.path.join(images_dir, candidate)
            if os.path.exists(test_path):
                input_path = test_path
                break
        
        if input_path is None:
            print(f"\n Image not found for {img_name}!")
            continue
        
        output_filename = f"{img_name}.tiff"
        output_path = os.path.join(output_dir, output_filename)
        
        cc_coords = cc_coords_all[idx] if mask_cc else None
        
        debug_this = (idx == 0)  # Debug first image only
        success = process_single_image(
            input_path=input_path,
            output_path=output_path,
            black_level=black_level,
            saturation=saturation,
            cc_coords=cc_coords,
            mask_cc=mask_cc,
            debug=debug_this,
        )
        
        if success:
            successful += 1
            gt_normalized = normalize_illuminant(illuminants[idx])
            
            wp_filename = f"{img_name}.wp"
            wp_path = os.path.join(output_dir, wp_filename)
            save_wp_file(wp_path, gt_normalized)
            
            gt_output.append({
                'filename': output_filename,
                'wp_filename': wp_filename,
                'illuminant': gt_normalized.tolist(),
                'camera': camera_name,
            })
    
    print(f"  Successfully processed: {successful}/{len(image_names)} images")
    return gt_output


def process_nus8_dataset(
    nus8_root: str,
    output_root: str,
    cameras: Optional[List[str]] = None,
    mask_cc: bool = True,
):
    """
    Process entire NUS8 dataset.
    
    Args:
        nus8_root: Root directory of NUS8 dataset (e.g., 'nus/')
        output_root: Output directory for processed dataset
        cameras: List of camera folder names to process (None = all found)
        mask_cc: Whether to mask color checker regions
    
    Output structure:
        output_root/
            all_images/
                Canon1DsMkIII_0001.tiff
                Canon1DsMkIII_0001.wp
                SonyA57_0001.tiff
                SonyA57_0001.wp
                ...
    """
    print("=" * 60)
    print("Process NUS8")
    print("=" * 60)
    print(f"Input:  {nus8_root}")
    print(f"Output: {output_root}")
    
    # Find all camera folders (folders containing *_gt.mat)
    if cameras is None:
        cameras = []
        for item in os.listdir(nus8_root):
            item_path = os.path.join(nus8_root, item)
            if os.path.isdir(item_path):
                mat_files = [f for f in os.listdir(item_path) if f.endswith('_gt.mat')]
                if mat_files:
                    cameras.append(item)
        cameras = sorted(cameras)
        
    all_images_dir = os.path.join(output_root, 'all_images')
    os.makedirs(all_images_dir, exist_ok=True)
    
    # Process each camera (all images go to all_images folder)
    all_gt_output = []
    for camera in cameras:
        camera_dir = os.path.join(nus8_root, camera)
        
        gt_output = process_camera_folder(
            camera_dir=camera_dir,
            output_dir=all_images_dir,
            camera_name=camera,
            mask_cc=mask_cc,
        )
        all_gt_output.extend(gt_output)
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Process NUS8 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using config file:
    python process_nus.py --config nus8.yaml
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
        help='Root directory of NUS8 dataset'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--cameras', '-c',
        type=str,
        nargs='+',
        default=None,
        help='Specific camera folders to process (default: all found)'
    )
    parser.add_argument(
        '--no-mask-cc',
        action='store_true',
        help='Do not mask color checker region (default: mask it)'
    )
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_dir = config.get('input', args.input)
        output_dir = config.get('output', args.output)
        cameras = config.get('cameras', args.cameras)
        mask_cc = config.get('mask_cc', not args.no_mask_cc)
    else:
        input_dir = args.input
        output_dir = args.output
        cameras = args.cameras
        mask_cc = not args.no_mask_cc
    
    if not input_dir or not output_dir:
        parser.error("--input and --output are required (via arguments or config file)")
    
    process_nus8_dataset(
        nus8_root=input_dir,
        output_root=output_dir,
        cameras=cameras,
        mask_cc=mask_cc,
    )


if __name__ == '__main__':
    main()
