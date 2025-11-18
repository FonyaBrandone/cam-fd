"""
Mixup loss for smooth decision boundaries.
Implements cross-entropy with mixup augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class MixupLoss(nn.Module):
    """
    Cross-entropy loss with Mixup data augmentation.
    
    Mixup interpolates between pairs of examples and their labels,
    encouraging smoother decision boundaries and better generalization.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 2):
        """
        Args:
            alpha: Beta distribution parameter for mixup. 
                   Higher alpha = more aggressive mixing
            num_classes: Number of classes for label smoothing
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def sample_lambda(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample mixing coefficient from Beta distribution.
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
        
        Returns:
            Lambda values for mixing (batch_size,)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, batch_size)
        else:
            lam = np.ones(batch_size)
        
        return torch.from_numpy(lam).float().to(device)
    
    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lam: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mixup to input and labels.
        
        Args:
            x: Input tensor (B, C, H, W)
            y: Target labels (B,)
            lam: Mixing coefficients (B,)
        
        Returns:
            mixed_x: Mixed input
            y_a: Original labels
            y_b: Shuffled labels
            lam: Mixing coefficients (adjusted shape)
        """
        batch_size = x.size(0)
        
        # Random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Reshape lambda for broadcasting
        lam_expanded = lam.view(-1, 1, 1, 1)
        
        # Mix inputs
        mixed_x = lam_expanded * x + (1 - lam_expanded) * x[index]
        
        y_a = y
        y_b = y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(
        self,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mixed cross-entropy loss.
        
        Args:
            pred: Model predictions (B, num_classes)
            y_a: Original labels (B,)
            y_b: Shuffled labels (B,)
            lam: Mixing coefficients (B,)
        
        Returns:
            Mixed cross-entropy loss
        """
        loss_a = self.ce_loss(pred, y_a)
        loss_b = self.ce_loss(pred, y_b)
        
        # Weight losses by lambda
        loss = lam * loss_a + (1 - lam) * loss_b
        
        return loss.mean()
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: nn.Module = None,
        pred: torch.Tensor = None,
        apply_mixup: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute mixup loss.
        
        Args:
            x: Input tensor (B, C, H, W)
            y: Target labels (B,)
            model: Model to use for prediction (if pred not provided)
            pred: Pre-computed predictions (if available)
            apply_mixup: Whether to apply mixup (False for validation)
        
        Returns:
            loss: Mixup cross-entropy loss
            info: Dict with additional information
        """
        if not apply_mixup:
            # Standard cross-entropy without mixup
            if pred is None:
                pred, _, _ = model(x)
            loss = F.cross_entropy(pred, y)
            return loss, {'mixup_applied': False}
        
        # Sample lambda
        batch_size = x.size(0)
        lam = self.sample_lambda(batch_size, x.device)
        
        # Apply mixup
        mixed_x, y_a, y_b, lam = self.mixup_data(x, y, lam)
        
        # Forward pass with mixed inputs
        if pred is None:
            pred, _, _ = model(mixed_x)
        
        # Compute mixed loss
        loss = self.mixup_criterion(pred, y_a, y_b, lam)
        
        info = {
            'mixup_applied': True,
            'mean_lambda': lam.mean().item(),
            'mixed_input': mixed_x
        }
        
        return loss, info


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix: alternative to Mixup that cuts and pastes patches.
    Can be used as a drop-in replacement for mixup.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # Random permutation
    index = torch.randperm(batch_size, device=x.device)
    
    # Generate random box
    _, _, h, w = x.size()
    cut_ratio = np.sqrt(1. - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    
    # Uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    
    return mixed_x, y, y[index], lam


if __name__ == "__main__":
    # Test mixup loss
    print("Testing Mixup Loss...")
    
    mixup_loss = MixupLoss(alpha=1.0, num_classes=2)
    
    # Create dummy data
    x = torch.randn(8, 3, 96, 96)
    y = torch.randint(0, 2, (8,))
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2)
    )
    
    # Compute loss with mixup
    loss, info = mixup_loss(x, y, model=model, apply_mixup=True)
    print(f"Mixup loss: {loss.item():.4f}")
    print(f"Mean lambda: {info['mean_lambda']:.4f}")
    
    # Compute loss without mixup
    loss, info = mixup_loss(x, y, model=model, apply_mixup=False)
    print(f"Standard CE loss: {loss.item():.4f}")
    
    # Test CutMix
    print("\nTesting CutMix...")
    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
    print(f"CutMix lambda: {lam:.4f}")
    print(f"Mixed input shape: {mixed_x.shape}")