import torch

def angular_loss(Ln_pred, Ln_gt, eps=1e-8):
    Ln_pred = Ln_pred / (Ln_pred.norm(dim=-1, keepdim=True) + eps)
    Ln_gt = Ln_gt / (Ln_gt.norm(dim=-1, keepdim=True) + eps)
    
    cos_sim = (Ln_pred * Ln_gt).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    
    ang_rad = torch.acos(cos_sim)
    
    return (ang_rad * 180.0 / torch.pi).mean()

def sparsity_loss(weights, eps=1e-8):
    weights = weights.clamp(min=eps)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
    return entropy.mean()


def total_loss(
    Ln_pred, Ln_gt,
    weights,
    lambda_sparsity=5e-4
):
    """
    Total training loss combining all components.
    
    Args:
        Ln_pred: Predicted normalized illuminant [B, 3]
        Ln_gt: Ground truth normalized illuminant [B, 3]
        weights: Patch weights [B, P]
        
    Returns:
        Dictionary with total loss and individual components
    """
    loss_angular = angular_loss(Ln_pred, Ln_gt)
    loss_sparse = sparsity_loss(weights)
    
    loss_total = loss_angular + lambda_sparsity * loss_sparse
    
    return {
        'total': loss_total,
        'angular': loss_angular,
        'sparse': loss_sparse,
    }
