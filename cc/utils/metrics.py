import torch
import numpy as np

def angular_error_deg(pred, target, eps=1e-8):
    pred = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    target = target / (target.norm(dim=-1, keepdim=True) + eps)
    cos = (pred * target).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    ang = torch.acos(cos) * (180.0 / torch.pi)
    return ang

def summarize_angles(angles):
    e = angles.detach().cpu().numpy() if torch.is_tensor(angles) else np.asarray(angles)
    
    if len(e) == 0:
        raise ValueError("Empty angle array")
    
    p25, p50, p75 = np.percentile(e, [25, 50, 75])
    
    return {
        'mean':    float(np.mean(e)),
        'median':  float(p50),
        'trimean': float((p25 + 2*p50 + p75) / 4),
        'best25':  float(np.mean(e[e <= p25])),
        'worst25': float(np.mean(e[e >= p75]))
    }
