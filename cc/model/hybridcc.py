# ========================================================================
# HybridCC — Multi-Block-Attention-based Color Constancy (ECCV 2026)
#
# Copyright (c) 2026 Oguzhan Ulucan, Diclehan Ulucan, and Marc Ebner
# University of Greifswald, Germany
#
# Corresponding author: Oguzhan Ulucan <oguzhan.ulucan@uni-greifswald.de>
#
# This software is released for academic, non-commercial use under the
# Creative Commons Attribution-NonCommercial 4.0 International License
# (CC BY-NC 4.0). See the LICENSE file in the project root for details.
#
# Provided "as is", without warranty of any kind, express or implied.
# If you use this code, please cite the accompanying paper.
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EPS = 1e-8

MOBILENET_V3_SMALL_CHANNELS = {
    0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40, 6: 40,
    7: 48, 8: 48, 9: 96, 10: 96, 11: 96, 12: 576
}


class MobileNetSaliency(nn.Module):
    """Backbone with learned saliency module."""
    def __init__(self, pretrained: bool = True, exit_layer: int = 8):
        super().__init__()
        self.exit_layer = min(max(int(exit_layer), 0), 12)
        
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        
        self.features = nn.Sequential(*list(m.features.children())[:self.exit_layer + 1])
        self.out_channels = MOBILENET_V3_SMALL_CHANNELS[self.exit_layer]
        
        C = self.out_channels
        self.sal = nn.Sequential(
            nn.Conv2d(C, C // 2, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(C // 2, C // 2, 3, padding=2, dilation=2, groups=C // 2, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C // 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(C, 1, 1)
        )

    def forward(self, img: torch.Tensor):
        f = self.features(img)
        S = torch.sigmoid(self.sal(f))
        return f, S

class PatchFeaturePool(nn.Module):
    """
    Pools backbone feature maps to match the block grid dimensions.
    Converts spatial feature map into per-block feature vectors.
    """
    def __init__(self, block=(32, 32)):
        super().__init__()
        self.block = tuple(block)

    def forward(self, fmap: torch.Tensor, Nh: int, Nw: int):
        up = fmap.contiguous(memory_format=torch.contiguous_format)
        Fp = F.adaptive_avg_pool2d(up, (Nh, Nw))
        return Fp.permute(0, 2, 3, 1).reshape(up.size(0), Nh * Nw, up.size(1))


class DWPointwiseMix(nn.Module):
    """Local spatial mixing layer for neighboring block communication."""
    def __init__(self, channels, kernel_size=3, pw_expand=2.0, dropout_p=0.05):
        super().__init__()
        padding = kernel_size // 2
        mid = int(round(channels * pw_expand))

        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.dw_act = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(mid)
        self.pw_act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.dw_act(self.dw_bn(self.dw(x)))
        y = self.pw_act(self.pw_bn(self.pw(y)))
        y = self.proj_bn(self.proj(y))
        y = self.dropout(y)
        return x + self.alpha * y


class PatchStatsEnhanced(nn.Module):
    """
    Per-block statistics using saliency-weighted soft maximum.
    
    Computes three complementary descriptors per block:
    1. m_b: Saliency-weighted soft-max color descriptor
    2. mu_b: Saliency-weighted mean intensity
    3. c_b: Block-level illumination prior
    """
    
    def __init__(self, block=(32, 32), sat_thresh=0.98, tau_init=0.02):
        super().__init__()
        self.block = tuple(block)
        self.sat_thresh = float(sat_thresh)

        # Learnable temperature for soft-max
        # Lower tau -> closer to hard max, higher tau -> closer to mean
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        
    @property
    def tau(self) -> float:
        """Return current temperature value to check."""
        with torch.no_grad():
            return torch.clamp(self.log_tau.exp(), 0.01, 0.5).item()

    def forward(self, img: torch.Tensor, S_up: torch.Tensor):
        B, C, H, W = img.shape
        h, w = self.block
        
        # Split image and saliency map into non-overlapping blocks and flatten each block's pixels
        x = img.unfold(2, h, h).unfold(3, w, w).flatten(-2, -1)
        s = S_up.unfold(2, h, h).unfold(3, w, w).flatten(-2, -1)

        # Mask saturated pixels
        sat = (x > self.sat_thresh).any(dim=1, keepdim=True).float()
        s = s * (1.0 - sat)

        # Normalize saliency weights per block so they sum to 1
        sumw = s.sum(dim=-1, keepdim=True)
        s_norm = s / (sumw + EPS)

        # If all pixels in a block are saturated,
        # we use uniform weights instead, fallback.
        fallback = torch.full_like(s, 1.0 / s.size(-1))
        mask = (sumw <= EPS)
        s = torch.where(mask, fallback, s_norm)

        # m_b: Saliency-weighted soft-max per channel
        tau = torch.clamp(self.log_tau.exp(), 0.01, 0.5)
        log_s = torch.log(s.clamp_min(EPS))
        m_b = tau * torch.logsumexp(x / tau + log_s, dim=-1)
        
        # mu_b: Saliency-weighted mean intensity
        x_intensity = x.mean(dim=1, keepdim=True)
        mu_b = (x_intensity * s).sum(dim=-1)
        
        # c_b: Block-level illumination prior
        m_b_norm_sq = (m_b ** 2).sum(dim=1, keepdim=True) + EPS
        c_b = (mu_b / m_b_norm_sq) * m_b
        
        # Reshape for output
        _, _, Nh, Nw = m_b.shape
        P = Nh * Nw  # total number of blocks
        
        return {
            'c_b': c_b.permute(0, 2, 3, 1).reshape(B, P, 3),    # (B, P, 3) illumination prior
            'm_b': m_b.permute(0, 2, 3, 1).reshape(B, P, 3),    # (B, P, 3) soft-max descriptor
            'mu_b': mu_b.squeeze(1).reshape(B, P),              # (B, P) mean intensity
            'grid': (Nh, Nw)                                    # grid dimensions
        }

class CrossPatchAttention(nn.Module):
    """
    Multi-head self-attention for global cross-block communication.
    Every block can attend to every other block regardless of spatial distance.
    """

    def __init__(self, dim=64, num_heads=2, dropout_p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout_p, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(dim * 2, dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2)
        x = x + x2

        x2 = self.norm2(x)
        x2 = self.ffn(x2)
        x = x + x2

        return x


class HYBRID(nn.Module):
    """
    HybridCC

    Two-branch architecture:
    1. Statistics Branch: Saliency-weighted block-level illumination priors
    2. Backbone Branch: Scene features

    Features from both branches are fused and refined through multi-head
    self-attention, then used to predict per-block illuminant estimates
    and combination weights for the final global illuminant.
    """

    def __init__(self,
                 block=(32, 32),               # block size for dividing the image
                 pretrained=True,              # use ImageNet pretrained backbone
                 sat_thresh=0.99,              # saturation threshold for masking bright pixels
                 wp_temperature=1.2,           # initial temperature for weight softmax
                 wp_tau_init=0.02,             # initial temperature for soft-max statistics
                 backbone_exit_layer=8,        # which MobileNet layer to exit at
                 backbone_encoder_dim=64,      # backbone feature projection dimension
                 stat_encoder_dim=64,          # statistics feature projection dimension
                 cp_hidden=64,                 # trunk hidden dimension
                 trunk_dropout=0.2,            # dropout in trunk
                 cross_attn_dropout=0.15,      # dropout in attention
                 cross_attn_heads=2,           # number of attention heads
                 mix_blocks=2,                 # number of spatial mixing layers
                 mix_pw_expand=2.0,            # expansion ratio in mixing layers
                 mix_dropout=0.05,             # dropout in mixing layers
                 ):

        super().__init__()

        self.block = tuple(block)

        # Backbone
        self.backbone = MobileNetSaliency(pretrained=pretrained, exit_layer=backbone_exit_layer)
        self.backbone_channels = self.backbone.out_channels

        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_imnet_mean", mean, persistent=False)
        self.register_buffer("_imnet_std", std, persistent=False)

        # Statistics branch: computes illumination priors from image pixels
        self.stats = PatchStatsEnhanced(
            block=block,
            sat_thresh=sat_thresh,
            tau_init=wp_tau_init
        )

        # Statistics output: c_b(3) + m_b(3) + mu_b(1) = 7 values per block
        stats_input_dim = 7

        # Encode 7-dim statistics into feature vector per block
        self.stat_encoder = nn.Sequential(
            nn.Linear(stats_input_dim, stat_encoder_dim // 2),
            nn.LayerNorm(stat_encoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(stat_encoder_dim // 2, stat_encoder_dim),
            nn.LayerNorm(stat_encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15)
        )
        self.stat_dim = stat_encoder_dim

        # Backbone feature branch: pools and encodes backbone features per block
        self.pfeat = PatchFeaturePool(block)

        # Project backbone channels to same dim as stats branch
        self.backbone_encoder = nn.Sequential(
            nn.Linear(self.backbone_channels, backbone_encoder_dim),
            nn.LayerNorm(backbone_encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )
        self.backbone_dim = backbone_encoder_dim

        if mix_blocks > 0:
            self.ctx_fp = nn.Sequential(*[
                DWPointwiseMix(backbone_encoder_dim,
                               kernel_size=3,
                               pw_expand=mix_pw_expand,
                               dropout_p=mix_dropout)
                for _ in range(mix_blocks)            # stack multiple mixing layers
            ])
        else:
            self.ctx_fp = nn.Identity()               # no mixing, pass through (we do not use this)

        # Fusion: concatenate stats + backbone
        self.in_dim = self.stat_dim + self.backbone_dim

        # Learnable temperature for weight softmax
        self.wp_logT = nn.Parameter(torch.log(torch.tensor(float(wp_temperature))))

        # Trunk that process fused features through attention
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, cp_hidden),
            nn.ReLU(inplace=True),
            CrossPatchAttention(cp_hidden, num_heads=cross_attn_heads, dropout_p=cross_attn_dropout),
            nn.Linear(cp_hidden, cp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=trunk_dropout)
        )

        # Prediction heads: two separate linear layers
        # cp_head: predicts per-block illuminant (one estimation per block)
        # wp_head: predicts per-block confidence weight (one scalar per block)
        self.cp_head = nn.Linear(cp_hidden, 3)
        self.wp_head = nn.Linear(cp_hidden, 1)

    @property
    def wp_temperature(self) -> float:
        """Current weight temperature."""
        with torch.no_grad():
            T = torch.clamp(self.wp_logT.exp(), 0.2, 2.0)
        return float(T.item())

    @property
    def wp_tau(self) -> float:
        """Current soft-max temperature from statistics module."""
        return self.stats.tau

    def forward(self, img: torch.Tensor):
        B, C, H, W = img.shape
        h, w = self.block
        Nh, Nw = H // h, W // w   # number of blocks, height and width

        # Backbone: extract features and saliency map
        # Normalize image to ImageNet stats
        # According to PyTorch, it is required for the pretrained models on ImageNet.
        x_bb = (img - self._imnet_mean) / self._imnet_std
        
        # Get feature map and saliency
        fmap, S_low = self.backbone(x_bb)

        # Upsample saliency to full image resolution for per-pixel weighting
        S_up = F.interpolate(S_low, size=(H, W), mode='bilinear', align_corners=False).clamp(EPS, 1.0)

        # Statistics branch: compute per-block illumination descriptors
        stats_out = self.stats(img, S_up)
        c_b = stats_out['c_b']                 # (B, P, 3) block illumination priors
        m_b = stats_out['m_b']                 # (B, P, 3) soft-max color descriptors
        mu_b = stats_out['mu_b']               # (B, P)    mean intensities

        # Concatenate all statistics
        stats_cat = torch.cat([c_b, m_b, mu_b.unsqueeze(-1)], dim=-1)

        # Encode statistics
        stat_feat = self.stat_encoder(stats_cat)

        # Backbone feature branch
        Fp = self.pfeat(fmap, Nh, Nw)
        Fp_proj = self.backbone_encoder(Fp)

        # Reshape to 2D grid for spatial mixing convolution
        Bf, P_f, Cf = Fp_proj.shape
        Fp2d = Fp_proj.view(Bf, Nh, Nw, Cf).permute(0, 3, 1, 2).contiguous()
        
        # Local spatial mixing
        Fp2d = self.ctx_fp(Fp2d)

        # Reshape back to flat block
        Fp_mixed = Fp2d.permute(0, 2, 3, 1).reshape(Bf, P_f, Cf)

        # Feature fusion: concatenate both branches
        feat = torch.cat([stat_feat, Fp_mixed], dim=-1)
        trunk_out = self.trunk(feat)

        # Prediction heads
        z = self.cp_head(trunk_out)
        Cp = F.softplus(z)

        w_logits = self.wp_head(trunk_out).squeeze(-1)

        T = torch.clamp(self.wp_logT.exp(), 0.2, 2.0)
        w = torch.softmax(w_logits / T, dim=-1)

        # Global illuminant: weighted average of per-block estimates
        # L = Σ (weight_i × illuminant_i) for all blocks
        L = (w.unsqueeze(-1) * Cp).sum(dim=1)

        # Normalize to unit vector
        Ln = L / (L.norm(dim=-1, keepdim=True) + EPS)

        return {
            'L': L,            # (B, 3) raw illuminant estimate
            'Ln': Ln,          # (B, 3) unit-norm illuminant
            'c_b': c_b,        # (B, P, 3) block illumination priors
            'm_b': m_b,        # (B, P, 3) soft-max descriptors
            'mu_b': mu_b,      # (B, P) mean intensities
            'weights': w,      # (B, P) per-block confidence weights
            'S': S_up,         # (B, 1, H, W) learned saliency map
            'Cp': Cp,          # (B, P, 3) per-block illuminant estimates
        }
