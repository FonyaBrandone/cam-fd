"""
Adversarial Weight Perturbation (AWP) for robust generalization.
Perturbs model weights to find flatter minima in the loss landscape.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from contextlib import contextmanager


class AWPLoss(nn.Module):
    """
    Adversarial Weight Perturbation (AWP) for improving robust generalization.
    
    AWP perturbs model weights in the direction that increases loss,
    encouraging the optimizer to find flatter minima that generalize better.
    
    Reference: Wu et al., "Adversarial Weight Perturbation Helps Robust 
    Generalization", NeurIPS 2020
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_awp: float = 0.01,
        gamma: float = 0.01,
        start_epoch: int = 0
    ):
        """
        Args:
            model: Model to apply AWP to
            lambda_awp: Weight for AWP loss
            gamma: Perturbation radius (step size for weight perturbation)
            start_epoch: Epoch to start applying AWP (for stability)
        """
        super().__init__()
        self.model = model
        self.lambda_awp = lambda_awp
        self.gamma = gamma
        self.start_epoch = start_epoch
        self.current_epoch = 0
        
        # Store backup of original weights
        self.backup = {}
        self.backup_eps = {}
    
    def enable(self):
        """Enable AWP (called after warmup epochs)."""
        self.current_epoch += 1
    
    def is_enabled(self) -> bool:
        """Check if AWP is enabled based on current epoch."""
        return self.current_epoch >= self.start_epoch
    
    def calc_awp(self, loss: torch.Tensor, eps: float = None):
        """
        Calculate adversarial weight perturbation.
        
        Args:
            loss: Loss to compute gradients from
            eps: Perturbation radius (uses self.gamma if None)
        """
        if eps is None:
            eps = self.gamma
        
        # Compute gradient w.r.t. model parameters
        grad_params = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False,
            retain_graph=True
        )
        
        # Perturb weights in gradient direction
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grad_params):
                if param.requires_grad and grad is not None:
                    # Normalize gradient
                    grad_norm = torch.norm(grad)
                    if grad_norm > 0:
                        # Perturb weight: w' = w + eps * grad / ||grad||
                        param.add_(grad, alpha=eps / (grad_norm + 1e-8))
    
    def save(self):
        """Save current model weights before perturbation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def restore(self):
        """Restore original model weights after computing AWP loss."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    @contextmanager
    def perturb_weights(self, loss: torch.Tensor):
        """
        Context manager for temporarily perturbing weights.
        
        Usage:
            with awp.perturb_weights(loss):
                perturbed_loss = compute_loss(model, data)
        """
        if not self.is_enabled():
            yield
            return
        
        # Save original weights
        self.save()
        
        # Perturb weights
        self.calc_awp(loss)
        
        try:
            yield
        finally:
            # Restore original weights
            self.restore()
    
    def forward(
        self,
        base_loss: torch.Tensor,
        compute_loss_fn: callable,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute AWP loss.
        
        Args:
            base_loss: Base loss before weight perturbation
            compute_loss_fn: Function to recompute loss with perturbed weights
            *args, **kwargs: Arguments to pass to compute_loss_fn
        
        Returns:
            AWP loss component
        """
        if not self.is_enabled():
            return torch.tensor(0.0, device=base_loss.device)
        
        # Perturb weights based on base loss
        with self.perturb_weights(base_loss):
            # Recompute loss with perturbed weights
            perturbed_loss = compute_loss_fn(*args, **kwargs)
        
        # AWP loss is the additional loss from perturbation
        awp_loss = self.lambda_awp * perturbed_loss
        
        return awp_loss


class SAM(nn.Module):
    """
    Sharpness-Aware Minimization (SAM).
    
    Alternative to AWP that directly minimizes sharpness.
    Can be used as a drop-in replacement for AWP.
    
    Reference: Foret et al., "Sharpness-Aware Minimization for 
    Efficiently Improving Generalization", ICLR 2021
    """
    
    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.05,
        adaptive: bool = False
    ):
        """
        Args:
            model: Model to apply SAM to
            rho: Neighborhood size for perturbation
            adaptive: Whether to use adaptive perturbation (ASAM)
        """
        super().__init__()
        self.model = model
        self.rho = rho
        self.adaptive = adaptive
        self.backup = {}
    
    def first_step(self, loss: torch.Tensor):
        """
        First step: compute gradient and perturb weights.
        
        Args:
            loss: Loss to compute gradient from
        """
        # Compute gradients
        grad_params = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False,
            retain_graph=False
        )
        
        # Save current weights and compute perturbation
        with torch.no_grad():
            # Compute perturbation norm
            if self.adaptive:
                # Adaptive SAM: normalize by parameter norm
                grad_norm = torch.norm(
                    torch.stack([
                        (torch.abs(p) if self.adaptive else 1.0) * g.norm(p=2)
                        for p, g in zip(self.model.parameters(), grad_params)
                        if p.requires_grad and g is not None
                    ])
                )
            else:
                # Standard SAM
                grad_norm = torch.norm(
                    torch.stack([
                        g.norm(p=2)
                        for g in grad_params
                        if g is not None
                    ])
                )
            
            # Perturb weights
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Save original weight
                    self.backup[name] = param.data.clone()
                    
                    # Compute perturbation
                    if self.adaptive:
                        eps = param.grad * (self.rho / (grad_norm + 1e-12)) * torch.abs(param)
                    else:
                        eps = param.grad * (self.rho / (grad_norm + 1e-12))
                    
                    # Apply perturbation
                    param.add_(eps)
    
    def second_step(self):
        """
        Second step: restore original weights.
        Call after computing loss with perturbed weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    # Test AWP
    print("Testing AWP Loss...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2)
    )
    
    # Create AWP
    awp = AWPLoss(model, lambda_awp=0.01, gamma=0.01, start_epoch=0)
    awp.current_epoch = 5  # Simulate being past start epoch
    
    # Dummy data
    x = torch.randn(4, 3, 96, 96)
    y = torch.randint(0, 2, (4,))
    
    # Compute base loss
    logits = model(x)
    base_loss = nn.functional.cross_entropy(logits, y)
    print(f"Base loss: {base_loss.item():.4f}")
    
    # Define loss computation function
    def compute_loss():
        logits = model(x)
        return nn.functional.cross_entropy(logits, y)
    
    # Compute AWP loss
    awp_loss = awp(base_loss, compute_loss)
    print(f"AWP loss: {awp_loss.item():.4f}")
    
    # Test context manager
    print("\nTesting AWP context manager...")
    with awp.perturb_weights(base_loss):
        perturbed_logits = model(x)
        perturbed_loss = nn.functional.cross_entropy(perturbed_logits, y)
        print(f"Perturbed loss: {perturbed_loss.item():.4f}")
    
    # After context, weights should be restored
    restored_logits = model(x)
    restored_loss = nn.functional.cross_entropy(restored_logits, y)
    print(f"Restored loss: {restored_loss.item():.4f}")
    print(f"Weights restored correctly: {torch.allclose(base_loss, restored_loss)}")
    
    # Test SAM
    print("\nTesting SAM...")
    sam = SAM(model, rho=0.05)
    
    # First step
    loss = compute_loss()
    loss.backward()
    sam.first_step(loss)
    
    # Compute loss with perturbed weights
    perturbed_loss = compute_loss()
    print(f"SAM perturbed loss: {perturbed_loss.item():.4f}")
    
    # Second step
    sam.second_step()
    restored_loss = compute_loss()
    print(f"SAM restored loss: {restored_loss.item():.4f}")