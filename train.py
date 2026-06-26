"""
Training script for HYBRIDCC
"""

import os
import yaml
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm

from cc.utils.datasets import CCMinimalDataset, GPUAugmentation, collate_fn_gpu
from cc.model.hybridcc import HYBRID
from cc.utils.metrics import angular_error_deg, summarize_angles
from cc.utils.losses import total_loss
from cc.utils.balanced_sampler import create_balanced_dataloader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

def collate_fn_val(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    lgt = torch.stack([b['illuminant'] for b in batch], dim=0)
    return images, lgt


class WarmupCosineScheduler:
    """
    Learning rate schedule with two phases:
    
    Phase 1 (warmup): Linearly increase LR from 0 to base_lr
    
    Phase 2 (cosine decay): Smoothly decrease LR following a cosine curve
        - Starts at base_lr, ends at min_lr
    """
    def __init__(self, optimizer, warmup_epochs, T_max, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max                # total cosine decay epochs
        self.min_lr = min_lr              # minimum LR at the end
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.max_base_lr = max(self.base_lrs)
        self.current_epoch = 0

    def step(self, epoch=None):
        """Update learning rate based on current epoch."""
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        epoch_0idx = epoch - 1

        if epoch_0idx < self.warmup_epochs:
            # Phase 1: linear warmup
            scale = (epoch_0idx + 1) / self.warmup_epochs
        else:
            # Phase 2: cosine decay
            cosine_epoch = epoch_0idx - self.warmup_epochs
            if cosine_epoch >= self.T_max:
                scale = self.min_lr / self.max_base_lr
            else:
                progress = cosine_epoch / self.T_max
                min_scale = self.min_lr / self.max_base_lr
                scale = min_scale + (1.0 - min_scale) * 0.5 * (1.0 + np.cos(np.pi * progress))

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]



class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.0005, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None

    def __call__(self, value):
        """Returns True if training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def build_datasets(cfg: Dict):
    """
    Load training and validation datasets from CSV files.
    Supports single-source or multi-source (multiple cameras/datasets).
    -> Please check balanced_sampler.py for further information. 
    
    Returns: (train_datasets_list, val_datasets_list)
    """
    data_cfg = cfg['data']
    aug_cfg = cfg['augmentation']
    crop_size = int(aug_cfg['crop_size'])
    include_original = bool(aug_cfg['include_original'])
    crops_per_image = int(aug_cfg['crops_per_image'])

    train_datasets = []
    val_datasets = []

    if 'sources' in data_cfg:
        print("Multi-source training enabled")
        for i, source in enumerate(data_cfg['sources']):
            train_csv = source.get('train_csv')
            val_csv = source.get('val_csv')
            root = source.get('root')

            if train_csv and os.path.exists(train_csv):
                ds = CCMinimalDataset(train_csv, root=root, load_size=crop_size,
                                      include_original=include_original,
                                      crops_per_image=crops_per_image)
                train_datasets.append(ds)
                print(f"  Source {i+1} train: {len(ds)} samples ({train_csv})")

            if val_csv and os.path.exists(val_csv):
                ds = CCMinimalDataset(val_csv, root=root, load_size=crop_size,
                                      include_original=True, crops_per_image=0)
                val_datasets.append(ds)
                print(f"  Source {i+1} val: {len(ds)} samples ({val_csv})")
    else:
        print("Single-source training")
        train_csv = data_cfg.get('train_csv')
        val_csv = data_cfg.get('val_csv')
        root = data_cfg.get('root')

        if train_csv and os.path.exists(train_csv):
            ds = CCMinimalDataset(train_csv, root=root, load_size=crop_size,
                                  include_original=include_original,
                                  crops_per_image=crops_per_image)
            train_datasets.append(ds)
            print(f"  Train: {len(ds)} samples")

        if val_csv and os.path.exists(val_csv):
            ds = CCMinimalDataset(val_csv, root=root, load_size=crop_size,
                                  include_original=True, crops_per_image=0)
            val_datasets.append(ds)
            print(f"  Val: {len(ds)} samples")

    if not train_datasets:
        raise ValueError("No training dataset found!")

    print(f"Total train: {sum(len(ds) for ds in train_datasets)}")
    if val_datasets:
        print(f"Total val: {sum(len(ds) for ds in val_datasets)}")

    return train_datasets, val_datasets



def create_train_loader(train_datasets, cfg, device):
    """
    Create training DataLoader.
    
    If multiple datasets + balance_sources: uses balanced sampling
    so each dataset contributes equally regardless of size.
    """
    data_cfg = cfg['data']
    batch_size = data_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = isinstance(device, str) and device.startswith('cuda')
    seed = cfg.get('train', {}).get('seed', 1337)
    balance_sources = data_cfg.get('balance_sources', False)

    if balance_sources and len(train_datasets) > 1:
        # Balanced: each dataset gets equal representation per batch
        print(f"Using balanced multi-dataset sampling")
        loader, sampler = create_balanced_dataloader(
            datasets=train_datasets, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
            seed=seed, epoch_size=None, collate_fn=collate_fn_gpu)
        return loader, sampler
    else:
        # Standard: just concatenate all datasets
        if len(train_datasets) > 1:
            dataset = ConcatDataset(train_datasets)
        else:
            dataset = train_datasets[0]

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn_gpu, drop_last=True)
        return loader, None


def create_val_loader(val_datasets, cfg, device):
    """Create validation DataLoader. No shuffling, no augmentation."""
    data_cfg = cfg['data']
    batch_size = data_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = isinstance(device, str) and device.startswith('cuda')

    if len(val_datasets) > 1:
        dataset = ConcatDataset(val_datasets)
    else:
        dataset = val_datasets[0]

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn_val, drop_last=False)


def build_model(cfg: Dict, device: str):
    """Create HYBRID model from config and move to GPU."""
    m = cfg['model']
    model = HYBRID(
        block=tuple(m['block_size']),
        pretrained=m.get('pretrained', True),
        sat_thresh=m.get('sat_thresh', 0.98),
        wp_temperature=m.get('wp_temperature', 1.0),
        wp_tau_init=m.get('wp_tau_init', 0.02),
        backbone_exit_layer=m.get('backbone_exit_layer', 8),
        backbone_encoder_dim=m.get('backbone_encoder_dim', 64),
        stat_encoder_dim=m.get('stat_encoder_dim', 64),
        cp_hidden=m.get('cp_hidden', 64),
        trunk_dropout=m.get('trunk_dropout', 0.2),
        cross_attn_dropout=m.get('cross_attn_dropout', 0.15),
        cross_attn_heads=m.get('cross_attn_heads', 2),
        mix_blocks=m.get('mix_blocks', 1),
        mix_pw_expand=m.get('mix_pw_expand', 2.0),
        mix_dropout=m.get('mix_dropout', 0.05)
    )

    model = model.to(device)

    if isinstance(device, str) and device.startswith('cuda'):
        model = model.to(memory_format=torch.channels_last)

    return model


def create_optimizer(model, cfg):
    lr = float(cfg['train']['lr'])
    weight_decay = float(cfg['train']['weight_decay'])
    backbone_lr_scale = float(cfg['train'].get('backbone_lr_scale', 1.0))

    if backbone_lr_scale < 1.0:
        print(f"Differential LR: head={lr:.2e}, backbone={lr * backbone_lr_scale:.2e}")
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.'):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = AdamW([
            {'params': head_params, 'lr': lr},
            {'params': backbone_params, 'lr': lr * backbone_lr_scale}
        ], weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def train_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch,
                sampler=None, gpu_aug=None):

    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    lambda_sparsity = float(cfg['loss']['lambda_sparsity'])
    use_amp = bool(cfg['train']['amp']) and device.startswith('cuda')
    grad_clip = float(cfg['train']['grad_clip_norm'])
    device_type = 'cuda' if isinstance(device, str) and device.startswith('cuda') else 'cpu'

    epoch_losses = {'total': 0.0, 'angular': 0.0, 'sparse': 0.0}
    n_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for images, lgt, is_orig in pbar:
        batch_size = images.size(0)

        images = images.to(device, non_blocking=True)
        lgt = lgt.to(device, non_blocking=True)
        is_orig = is_orig.to(device, non_blocking=True)

        if gpu_aug is not None:
            images, lgt = gpu_aug(images, lgt, is_orig)

        if device_type == 'cuda':
            images = images.to(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type, enabled=use_amp):
            out = model(images)
            losses = total_loss(
                out['Ln'], lgt,
                out.get('weights'),
                lambda_sparsity=lambda_sparsity
            )

        if use_amp:
            scaler.scale(losses['total']).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        for key in epoch_losses:
            epoch_losses[key] += losses[key].item() * batch_size
        n_samples += batch_size

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'ang': f"{losses['angular'].item():.4f}",
            'lr': f"{get_lr(optimizer):.2e}"
        })

    for key in epoch_losses:
        epoch_losses[key] /= n_samples

    return epoch_losses



def validate(model, val_loader, device, cfg, use_amp=True):
    model.eval()
    lambda_sparsity = float(cfg['loss']['lambda_sparsity'])
    device_type = 'cuda' if isinstance(device, str) and device.startswith('cuda') else 'cpu'

    all_angles = []
    total_loss_val = 0.0
    total_angular = 0.0
    total_sparse = 0.0
    n_samples = 0

    with torch.inference_mode(), torch.amp.autocast(device_type, enabled=use_amp):
        for images, lgt in tqdm(val_loader, desc='Validation', leave=False):
            batch_size = images.size(0)
            images = images.to(device, non_blocking=True)
            if device_type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            lgt = lgt.to(device, non_blocking=True)

            out = model(images)
            losses = total_loss(out['Ln'], lgt, out.get('weights'),
                                lambda_sparsity=lambda_sparsity)

            total_loss_val += losses['total'].item() * batch_size
            total_angular += losses['angular'].item() * batch_size
            total_sparse += losses['sparse'].item() * batch_size

            angles = angular_error_deg(out['Ln'], lgt)
            all_angles.append(angles.detach().cpu())
            n_samples += batch_size

    all_angles = torch.cat(all_angles, dim=0)
    stats = summarize_angles(all_angles)
    stats['composite_loss'] = total_loss_val / n_samples
    stats['angular_loss'] = total_angular / n_samples
    stats['sparse_loss'] = total_sparse / n_samples
    return stats


def save_checkpoint(model, optimizer, scheduler, epoch, val_stats, cfg, save_path):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler_epoch': scheduler.current_epoch if scheduler else None,
        'scheduler_base_lrs': scheduler.base_lrs if scheduler else None,
        'val_stats': val_stats,
        'cfg': cfg,
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")



def main():
    parser = argparse.ArgumentParser(description="Train HYBRID")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if cfg['train']['deterministic']:
        set_seed(int(cfg['train']['seed']))
    elif isinstance(device, str) and device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True 

    if isinstance(device, str) and device.startswith('cuda'):
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    # Create output directory
    exp_name = cfg['experiment']['name']
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(cfg['experiment']['out_dir']) / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    train_datasets, val_datasets = build_datasets(cfg)
    train_loader, sampler = create_train_loader(train_datasets, cfg, device)
    val_loader = create_val_loader(val_datasets, cfg, device) if val_datasets else None

    aug_cfg = cfg['augmentation']
    gpu_aug = GPUAugmentation(
        crop_size=int(aug_cfg['crop_size']),
        scale_range=tuple(aug_cfg.get('scale_range', [0.1, 1.0])),
        augmentation_angle=float(aug_cfg.get('augmentation_angle', 360.0)),
        hflip_p=float(aug_cfg.get('hflip_p', 0.5)),
        vflip_p=float(aug_cfg.get('vflip_p', 0.5)),
        gain_range=tuple(aug_cfg.get('gain_range', [0.8, 1.2])),
        gain_prob=float(aug_cfg.get('gain_prob', 0.5)),
        color_aug_strength=float(aug_cfg.get('color_aug_strength', 0.4)),
    ).to(device)

    print(f"\n{'='*60}")
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    if val_loader:
        print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Balanced sampling: {sampler is not None}")
    print(f"{'='*60}\n")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M ({n_params * 4 / 1024**2:.1f} MB FP32)")

    optimizer = create_optimizer(model, cfg)

    epochs = int(cfg['train']['epochs'])
    lr_cfg = cfg['train']['lr_schedule']
    scheduler = None
    if lr_cfg['type'] == 'cosine':
        warmup = int(lr_cfg['warmup_epochs'])
        T_max = int(lr_cfg.get('T_max', epochs - warmup))
        min_lr = float(lr_cfg['min_lr'])
        scheduler = WarmupCosineScheduler(optimizer, warmup, T_max, min_lr)
        print(f"LR schedule: warmup={warmup}, cosine T_max={T_max}, min_lr={min_lr}")

    use_amp = bool(cfg['train']['amp']) and device.startswith('cuda')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    early_stopper = EarlyStopper(
        patience=int(cfg['train']['early_stop_patience']),
        min_delta=float(cfg['train']['early_stop_min_delta']),
        mode='min'
    )

    best_loss = float('inf')
    best_error = float('inf')
    start_epoch = 0
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        if scheduler and ckpt.get('scheduler_epoch') is not None:
            scheduler.current_epoch = ckpt['scheduler_epoch']
            if ckpt.get('scheduler_base_lrs') is not None:
                scheduler.base_lrs = ckpt['scheduler_base_lrs']
                scheduler.max_base_lr = max(scheduler.base_lrs)
            scheduler.step(ckpt['scheduler_epoch'])

        if isinstance(device, str) and device.startswith('cuda'):
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
        if start_epoch >= epochs:
            raise ValueError(
                f"start_epoch ({start_epoch}) >= epochs ({epochs}): "
                f"the resume checkpoint appears to be already at the last epoch. "
                f"Nothing to train. Increase 'epochs' in the config or check the checkpoint."
            )

    val_interval = int(cfg['train']['val_interval'])
    log_file = run_dir / 'training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Angular,Val_Mean,Val_Median,Val_Trimean,Val_Composite_Loss,LR\n")


    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"[EPOCH {epoch+1}/{epochs}]")
        print(f"{'='*60}")

        if scheduler is not None:
            scheduler.step(epoch + 1)
            lrs = scheduler.get_last_lr()
            print(f"LR: {', '.join(f'{lr:.2e}' for lr in lrs)}")

        train_losses = train_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            epoch+1, sampler, gpu_aug=gpu_aug
        )
        print(f"Train -> total: {train_losses['angular']:.4f}")

        val_stats = None
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            val_stats = validate(model, val_loader, device, cfg, use_amp)
            model.train()

            print(f"Val -> mean: {val_stats['mean']:.3f}°, "
                  f"median: {val_stats['median']:.3f}°")
            print(f"best25: {val_stats['best25']:.3f}°, "
                  f"worst25: {val_stats['worst25']:.3f}°")

            if hasattr(model, 'wp_tau'):
                print(f" τ={model.wp_tau:.4f}, T={model.wp_temperature:.4f}")

            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{train_losses['total']:.4f},{train_losses['angular']:.4f},"
                        f"{val_stats['mean']:.3f},{val_stats['median']:.3f},"
                        f"{val_stats['trimean']:.3f},{val_stats['composite_loss']:.4f},"
                        f"{get_lr(optimizer):.2e}\n")

            if val_stats['composite_loss'] < best_loss:
                best_loss = val_stats['composite_loss']
                best_error = val_stats['mean']
                save_checkpoint(model, optimizer, scheduler, epoch, val_stats, cfg,
                                run_dir / 'best.ckpt')
                print(f"loss={best_loss:.4f}, error={best_error:.3f}°")
            else:
                print(f"Best so far: loss={best_loss:.4f}, error={best_error:.3f}°")

            if early_stopper(val_stats['composite_loss']):
                print(f"\n No improvement for {early_stopper.patience} epochs")
                break
            elif early_stopper.counter > 0:
                print(f"Early stop: {early_stopper.counter}/{early_stopper.patience}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_stats, cfg,
                            run_dir / 'last.ckpt')

    if val_loader:
        val_stats = validate(model, val_loader, device, cfg, use_amp)

    save_checkpoint(model, optimizer, scheduler, epoch,
                    val_stats if val_loader else None, cfg,
                    run_dir / 'last.ckpt')

    print(f"\n{'='*60}")
    print(f"Best error: {best_error:.3f}°")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
