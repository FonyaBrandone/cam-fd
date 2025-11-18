"""
Utilities for adversarial attacks.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


def clip_by_tensor(x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    """
    Clip tensor by min and max tensors (element-wise).
    
    Args:
        x: Input tensor
        x_min: Minimum values
        x_max: Maximum values
    
    Returns:
        Clipped tensor
    """
    return torch.max(torch.min(x, x_max), x_min)


def normalize_grad(grad: torch.Tensor, norm_type: str = 'inf') -> torch.Tensor:
    """
    Normalize gradient.
    
    Args:
        grad: Gradient tensor
        norm_type: Type of norm ('inf', 'l1', 'l2')
    
    Returns:
        Normalized gradient
    """
    if norm_type == 'inf':
        return grad.sign()
    elif norm_type == 'l1':
        grad_abs = grad.abs()
        return grad / (grad_abs.sum(dim=[1, 2, 3], keepdim=True) + 1e-12)
    elif norm_type == 'l2':
        grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1)
        return grad / (grad_norm + 1e-12)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def project_perturbation(
    perturbation: torch.Tensor,
    epsilon: float,
    norm_type: str = 'inf'
) -> torch.Tensor:
    """
    Project perturbation to epsilon ball.
    
    Args:
        perturbation: Perturbation tensor
        epsilon: Maximum perturbation magnitude
        norm_type: Type of norm ('inf', 'l2')
    
    Returns:
        Projected perturbation
    """
    if norm_type == 'inf':
        return torch.clamp(perturbation, -epsilon, epsilon)
    elif norm_type == 'l2':
        batch_size = perturbation.size(0)
        perturbation_flat = perturbation.view(batch_size, -1)
        l2_norm = perturbation_flat.norm(p=2, dim=1, keepdim=True)
        scale = torch.min(
            torch.ones_like(l2_norm),
            epsilon / (l2_norm + 1e-12)
        )
        scale = scale.view(batch_size, 1, 1, 1)
        return perturbation * scale
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def compute_attack_success_rate(
    clean_pred: torch.Tensor,
    adv_pred: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> float:
    """
    Compute attack success rate.
    
    Args:
        clean_pred: Clean predictions
        adv_pred: Adversarial predictions
        labels: True labels
        targeted: Whether attack is targeted
        target_labels: Target labels (for targeted attacks)
    
    Returns:
        Attack success rate
    """
    if targeted:
        if target_labels is None:
            raise ValueError("target_labels required for targeted attacks")
        success = (adv_pred == target_labels).float()
    else:
        # Untargeted: successful if prediction changes from correct
        clean_correct = (clean_pred == labels)
        adv_wrong = (adv_pred != labels)
        success = (clean_correct & adv_wrong).float()
    
    return success.mean().item()


def compute_perturbation_metrics(
    clean: torch.Tensor,
    adv: torch.Tensor
) -> dict:
    """
    Compute perturbation metrics.
    
    Args:
        clean: Clean images
        adv: Adversarial images
    
    Returns:
        Dictionary of metrics
    """
    perturbation = (adv - clean).abs()
    
    # Flatten spatial dimensions
    batch_size = clean.size(0)
    pert_flat = perturbation.view(batch_size, -1)
    
    metrics = {
        'linf': pert_flat.max(dim=1)[0].mean().item(),
        'l2': pert_flat.norm(p=2, dim=1).mean().item(),
        'l1': pert_flat.norm(p=1, dim=1).mean().item(),
        'l0': (pert_flat > 0).sum(dim=1).float().mean().item(),
        'mean': perturbation.mean().item(),
        'std': perturbation.std().item()
    }
    
    return metrics


if __name__ == "__main__":
    print("Attack utilities module")