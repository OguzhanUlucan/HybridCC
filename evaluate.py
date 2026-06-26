"""
Evaluation script for HYBRIDCC
"""

import os
import yaml
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from cc.utils.datasets import CCMinimalDataset
from cc.model.hybridcc import HYBRID
from cc.utils.metrics import angular_error_deg, summarize_angles


def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    ills = torch.stack([b['illuminant'] for b in batch], dim=0)
    return images, ills


def print_model_info(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  FP32: {n_params * 4 / 1024**2:.2f} MB | FP16: {n_params * 2 / 1024**2:.2f} MB")
    print(f"  Learned τ={model.wp_tau:.4f}")


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt.get('cfg')
    if cfg is None or 'model' not in cfg:
        raise RuntimeError("No 'cfg.model' in checkpoint.")

    m = cfg['model']
    model = HYBRID(
        block=tuple(m['block_size']),
        pretrained=False,
        sat_thresh=m['sat_thresh'],
        wp_temperature=m['wp_temperature'],
        wp_tau_init=m['wp_tau_init'],
        backbone_exit_layer=m['backbone_exit_layer'],
        backbone_encoder_dim=m['backbone_encoder_dim'],
        stat_encoder_dim=m['stat_encoder_dim'],
        cp_hidden=m['cp_hidden'],
        trunk_dropout=m['trunk_dropout'],
        cross_attn_dropout=m['cross_attn_dropout'],
        cross_attn_heads=m['cross_attn_heads'],
        mix_blocks=m['mix_blocks'],
        mix_pw_expand=m['mix_pw_expand'],
        mix_dropout=m['mix_dropout'],
    )

    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)
    if device.startswith('cuda'):
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    print_model_info(model)

    return model



def evaluate(model, loader, device, use_amp):
    """
    Returns statistics, predictions, and per-image angles.
    """
    all_angles, all_preds = [], []
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    with torch.inference_mode(), torch.amp.autocast(device_type, enabled=use_amp):
        for imgs, lgt in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            if device.startswith('cuda'):
                imgs = imgs.to(memory_format=torch.channels_last)
            lgt = lgt.to(device, non_blocking=True)

            out = model(imgs)
            ang = angular_error_deg(out['Ln'], lgt)

            all_preds.append(out['Ln'].detach().cpu())
            all_angles.append(ang.detach().cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    angles = torch.cat(all_angles, dim=0).numpy()
    stats = summarize_angles(torch.from_numpy(angles))

    return stats, preds, angles


def parse_config(args):
    cfg = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    settings = {
        'ckpt': args.ckpt or cfg.get('model', {}).get('ckpt'),
        'csv': args.csv or cfg.get('data', {}).get('csv'),
        'root': args.root if args.root is not None else cfg.get('data', {}).get('root', ""),
        'size': args.size if args.size is not None else cfg.get('eval', {}).get('size', 384),
        'batch': args.batch if args.batch is not None else cfg.get('eval', {}).get('batch_size', 32),
        'workers': args.workers if args.workers is not None else cfg.get('eval', {}).get('num_workers', 4),
        'use_amp': cfg.get('eval', {}).get('amp', True) and (not args.no_amp),
        'save_csv': args.save_csv or cfg.get('eval', {}).get('save_csv'),
    }

    if not settings['ckpt']:
        raise RuntimeError("Provide checkpoint via --ckpt or model.ckpt in config")
    if not settings['csv']:
        raise RuntimeError("Provide CSV via --csv or data.csv in config")

    return settings


def find_gt_columns(df):
    options = [
        ['Lr', 'Lg', 'Lb'],
        ['R', 'G', 'B'],
        ['mean_r', 'mean_g', 'mean_b'], # do not forget to add more if you changed the structure of the csv.
    ]
    for cols in options:
        if all(c in df.columns for c in cols):
            return cols
    raise RuntimeError(f"No GT columns found. Available: {list(df.columns)}")


def find_path_column(df):
    for col in ['path', 'image', 'img', 'filename']:
        if col in df.columns:
            return col
    raise RuntimeError(f"No path column found. Available: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate HYBRID")
    ap.add_argument('--config', type=str, default=None, help='Path to YAML config')
    ap.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    ap.add_argument('--csv', type=str, default=None, help='Path to evaluation CSV')
    ap.add_argument('--root', type=str, default=None, help='Data root directory')
    ap.add_argument('--size', type=int, default=None, help='Image size')
    ap.add_argument('--batch', type=int, default=None, help='Batch size')
    ap.add_argument('--workers', type=int, default=None, help='DataLoader workers')
    ap.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    ap.add_argument('--save_csv', type=str, default=None, help='Save per-image results')
    args = ap.parse_args()

    s = parse_config(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    use_amp = device.startswith('cuda') and s['use_amp']

    if not os.path.isfile(s['csv']):
        raise FileNotFoundError(f"CSV not found: {s['csv']}")

    df = pd.read_csv(s['csv'])
    if len(df) == 0:
        raise RuntimeError("CSV is empty!")

    path_col = find_path_column(df)
    gt_cols = find_gt_columns(df)
    path_list = df[path_col].tolist()
    gt_np = df[gt_cols].to_numpy(dtype=np.float32)

    ds = CCMinimalDataset(
        s['csv'], root=s['root'],
        load_size=s['size'], include_original=True,
        crops_per_image=0
    )

    loader = DataLoader(
        ds, batch_size=s['batch'], shuffle=False,
        num_workers=s['workers'],
        pin_memory=device.startswith('cuda'),
        persistent_workers=(s['workers'] > 0),
        collate_fn=collate_fn, drop_last=False
    )

    print(f" Checkpoint: {s['ckpt']}")
    if not os.path.isfile(s['ckpt']):
        raise FileNotFoundError(f"Checkpoint not found: {s['ckpt']}")

    ckpt = torch.load(s['ckpt'], map_location='cpu', weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  Dataset: {s['csv']} ({len(ds)} images)")
    print(f"  Size: {s['size']}×{s['size']}, Batch: {s['batch']}")
    print(f"{'='*60}")

    stats, preds, angles = evaluate(model, loader, device, use_amp)

    print(f"\n{'='*60}")
    print(f"  Mean:     {stats['mean']:.3f}°")
    print(f"  Median:   {stats['median']:.3f}°")
    print(f"  Best 25%: {stats['best25']:.3f}°")
    print(f"  Worst 25%: {stats['worst25']:.3f}°")
    print(f"{'='*60}\n")

    if s['save_csv']:
        out_df = pd.DataFrame({
            'path': path_list,
            'gt_r': gt_np[:, 0], 'gt_g': gt_np[:, 1], 'gt_b': gt_np[:, 2],
            'pred_r': preds[:, 0], 'pred_g': preds[:, 1], 'pred_b': preds[:, 2],
            'ang_deg': angles
        })

        out_dir = os.path.dirname(s['save_csv'])
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(s['save_csv'], index=False)
        print(f"Per-image results: {s['save_csv']}")

        dataset_name = Path(s['root']).parent.name if s['root'] else Path(s['csv']).parent.parent.name
        results_path = os.path.join(out_dir or ".", f"results_{dataset_name}.txt")
        with open(results_path, "w") as f:
            f.write(f"Dataset: {s['csv']}\n")
            f.write(f"Mean:     {stats['mean']:.3f}°\n")
            f.write(f"Median:   {stats['median']:.3f}°\n")
            f.write(f"Best 25%: {stats['best25']:.3f}°\n")
            f.write(f"Worst 25%: {stats['worst25']:.3f}°\n")
        print(f"Summary: {results_path}")


if __name__ == "__main__":
    main()
