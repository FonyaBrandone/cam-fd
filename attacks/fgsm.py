# FGSM Attack
"""
Fast Gradient Sign Method (FGSM) attack implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Generates adversarial examples using a single gradient step:
    x_adv = x + epsilon * sign(grad_x Loss(x, y))
    
    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,  # 8/255
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False
    ):
        """
        Args:
            model: Model to attack
            epsilon: Maximum perturbation magnitude
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
            targeted: Whether to perform targeted attack
        """
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            x: Clean input images (B, C, H, W)
            y: True labels (B,)
            return_logits: Whether to return adversarial logits
        
        Returns:
            x_adv: Adversarial examples
            logits_adv: Adversarial logits (if return_logits=True)
        """
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        # Forward pass
        if hasattr(self.model, 'forward'):
            # Handle CAM-FD model
            if hasattr(self.model, 'backbone'):
                logits, _, _ = self.model(x_adv)
            else:
                logits = self.model(x_adv)
        else:
            logits = self.model(x_adv)
        
        # Compute loss
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        grad = x_adv.grad.data
        
        # Generate adversarial example
        if self.targeted:
            # Targeted: minimize loss (move toward target)
            x_adv = x_adv - self.epsilon * grad.sign()
        else:
            # Untargeted: maximize loss (move away from true label)
            x_adv = x_adv + self.epsilon * grad.sign()
        
        # Clip to valid range
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        # Ensure perturbation is within epsilon ball
        perturbation = x_adv - x
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + perturbation, self.clip_min, self.clip_max)
        
        x_adv = x_adv.detach()
        
        if return_logits:
            with torch.no_grad():
                if hasattr(self.model, 'backbone'):
                    logits_adv, _, _ = self.model(x_adv)
                else:
                    logits_adv = self.model(x_adv)
            return x_adv, logits_adv
        
        return x_adv
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convenience method for generating adversarial examples."""
        return self.generate(x, y)


class FGSM_Targeted(FGSM):
    """Targeted version of FGSM attack."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.031, **kwargs):
        super().__init__(model, epsilon, targeted=True, **kwargs)


class FGSM_Random(FGSM):
    """
    FGSM with random initialization.
    Adds small random noise before applying FGSM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,
        alpha: float = 0.01,  # Random initialization magnitude
        **kwargs
    ):
        super().__init__(model, epsilon, **kwargs)
        self.alpha = alpha
    
    def generate(self, x: torch.Tensor, y: torch.Tensor, return_logits: bool = False):
        """Generate adversarial examples with random start."""
        # Add random noise
        x_random = x + torch.empty_like(x).uniform_(-self.alpha, self.alpha)
        x_random = torch.clamp(x_random, self.clip_min, self.clip_max)
        
        # Apply FGSM from random start
        x_adv = x_random.clone().detach()
        x_adv.requires_grad = True
        
        # Forward pass
        if hasattr(self.model, 'backbone'):
            logits, _, _ = self.model(x_adv)
        else:
            logits = self.model(x_adv)
        
        # Compute loss and gradient
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = x_adv.grad.data
        
        # Generate adversarial example
        x_adv = x_adv + self.epsilon * grad.sign()
        
        # Clip to valid range and epsilon ball
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        perturbation = x_adv - x
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + perturbation, self.clip_min, self.clip_max)
        
        x_adv = x_adv.detach()
        
        if return_logits:
            with torch.no_grad():
                if hasattr(self.model, 'backbone'):
                    logits_adv, _, _ = self.model(x_adv)
                else:
                    logits_adv = self.model(x_adv)
            return x_adv, logits_adv
        
        return x_adv


if __name__ == "__main__":
    # Test FGSM attack
    print("Testing FGSM Attack...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2)
    )
    model.eval()
    
    # Create FGSM attacker
    fgsm = FGSM(model, epsilon=0.031)
    
    # Generate adversarial examples
    x = torch.randn(4, 3, 96, 96)
    y = torch.randint(0, 2, (4,))
    
    # Get clean predictions
    with torch.no_grad():
        logits_clean = model(x)
        pred_clean = logits_clean.argmax(dim=1)
    
    # Generate adversarial examples
    x_adv = fgsm.generate(x, y)
    
    # Get adversarial predictions
    with torch.no_grad():
        logits_adv = model(x_adv)
        pred_adv = logits_adv.argmax(dim=1)
    
    # Compute perturbation
    perturbation = (x_adv - x).abs()
    max_pert = perturbation.max().item()
    mean_pert = perturbation.mean().item()
    
    print(f"Clean predictions: {pred_clean.tolist()}")
    print(f"Adversarial predictions: {pred_adv.tolist()}")
    print(f"Max perturbation: {max_pert:.4f}")
    print(f"Mean perturbation: {mean_pert:.4f}")
    print(f"Attack success rate: {(pred_clean != pred_adv).float().mean().item():.2%}")
    
    # Test FGSM with random initialization
    print("\nTesting FGSM with Random Initialization...")
    fgsm_random = FGSM_Random(model, epsilon=0.031, alpha=0.01)
    x_adv_random = fgsm_random.generate(x, y)
    
    with torch.no_grad():
        logits_adv_random = model(x_adv_random)
        pred_adv_random = logits_adv_random.argmax(dim=1)
    
    print(f"Clean predictions: {pred_clean.tolist()}")
    print(f"Adversarial predictions (random): {pred_adv_random.tolist()}")
    print(f"Attack success rate: {(pred_clean != pred_adv_random).float().mean().item():.2%}")