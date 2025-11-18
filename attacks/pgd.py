"""
Projected Gradient Descent (PGD) attack implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD:
    """
    Projected Gradient Descent (PGD) attack.
    
    Generates adversarial examples using multiple gradient steps with projection:
    x_adv^(t+1) = Proj_epsilon(x_adv^(t) + alpha * sign(grad_x Loss(x_adv^(t), y)))
    
    PGD is a stronger iterative version of FGSM and is commonly used for
    adversarial training.
    
    Reference: Madry et al., "Towards Deep Learning Models Resistant to 
    Adversarial Attacks", ICLR 2018
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,  # 8/255
        step_size: float = 0.007,  # 2/255
        num_steps: int = 10,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        loss_fn: str = 'ce'  # 'ce' or 'cw'
    ):
        """
        Args:
            model: Model to attack
            epsilon: Maximum perturbation magnitude (L-inf norm)
            step_size: Step size for each iteration
            num_steps: Number of iterations
            random_start: Whether to start from random point in epsilon ball
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
            targeted: Whether to perform targeted attack
            loss_fn: Loss function ('ce' for cross-entropy, 'cw' for C&W)
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.loss_fn = loss_fn
    
    def _project(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """
        Project adversarial example back onto epsilon ball around x.
        
        Args:
            x: Original clean input
            x_adv: Current adversarial example
        
        Returns:
            Projected adversarial example
        """
        # Clip perturbation to epsilon ball
        perturbation = x_adv - x
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        # Project back to valid pixel range
        x_adv = torch.clamp(x + perturbation, self.clip_min, self.clip_max)
        
        return x_adv
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for adversarial example generation."""
        if self.loss_fn == 'ce':
            loss = F.cross_entropy(logits, y)
        elif self.loss_fn == 'cw':
            # Carlini-Wagner loss
            y_onehot = F.one_hot(y, num_classes=logits.size(1)).float()
            correct_logit = (y_onehot * logits).sum(dim=1)
            wrong_logit = ((1 - y_onehot) * logits - y_onehot * 10000).max(dim=1)[0]
            loss = -F.relu(correct_logit - wrong_logit + 50).mean()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")
        
        return loss
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        Args:
            x: Clean input images (B, C, H, W)
            y: True labels (B,)
            return_logits: Whether to return adversarial logits
        
        Returns:
            x_adv: Adversarial examples
            logits_adv: Adversarial logits (if return_logits=True)
        """
        # Initialize adversarial example
        if self.random_start:
            # Start from random point in epsilon ball
            x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        
        # Iterative attack
        for step in range(self.num_steps):
            x_adv.requires_grad = True
            
            # Forward pass
            if hasattr(self.model, 'backbone'):
                logits, _, _ = self.model(x_adv)
            else:
                logits = self.model(x_adv)
            
            # Compute loss
            loss = self._compute_loss(logits, y)
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            grad = x_adv.grad.data
            
            # Update adversarial example
            with torch.no_grad():
                if self.targeted:
                    # Targeted: minimize loss
                    x_adv = x_adv - self.step_size * grad.sign()
                else:
                    # Untargeted: maximize loss
                    x_adv = x_adv + self.step_size * grad.sign()
                
                # Project back to epsilon ball
                x_adv = self._project(x, x_adv)
            
            # Clear gradients
            if x_adv.grad is not None:
                x_adv.grad.zero_()
        
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


class PGD_L2(PGD):
    """
    PGD attack with L2 norm constraint instead of L-infinity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        step_size: float = 0.2,
        num_steps: int = 10,
        **kwargs
    ):
        super().__init__(model, epsilon, step_size, num_steps, **kwargs)
    
    def _project(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Project onto L2 ball."""
        perturbation = x_adv - x
        
        # Compute L2 norm per sample
        batch_size = x.size(0)
        perturbation_flat = perturbation.view(batch_size, -1)
        l2_norm = perturbation_flat.norm(p=2, dim=1, keepdim=True)
        
        # Scale if exceeds epsilon
        scale = torch.min(
            torch.ones_like(l2_norm),
            self.epsilon / (l2_norm + 1e-8)
        )
        scale = scale.view(batch_size, 1, 1, 1)
        perturbation = perturbation * scale
        
        # Project to valid range
        x_adv = torch.clamp(x + perturbation, self.clip_min, self.clip_max)
        
        return x_adv


class PGD_Adaptive:
    """
    Adaptive PGD that adjusts step size based on attack progress.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,
        initial_step_size: float = 0.007,
        num_steps: int = 20,
        decay_factor: float = 0.9,
        **kwargs
    ):
        self.base_pgd = PGD(model, epsilon, initial_step_size, num_steps, **kwargs)
        self.initial_step_size = initial_step_size
        self.decay_factor = decay_factor
    
    def generate(self, x: torch.Tensor, y: torch.Tensor, return_logits: bool = False):
        """Generate adversarial examples with adaptive step size."""
        # Initialize
        if self.base_pgd.random_start:
            x_adv = x + torch.empty_like(x).uniform_(-self.base_pgd.epsilon, self.base_pgd.epsilon)
            x_adv = torch.clamp(x_adv, self.base_pgd.clip_min, self.base_pgd.clip_max)
        else:
            x_adv = x.clone()
        
        step_size = self.initial_step_size
        
        for step in range(self.base_pgd.num_steps):
            x_adv.requires_grad = True
            
            # Forward pass
            if hasattr(self.base_pgd.model, 'backbone'):
                logits, _, _ = self.base_pgd.model(x_adv)
            else:
                logits = self.base_pgd.model(x_adv)
            
            # Compute loss
            loss = self.base_pgd._compute_loss(logits, y)
            loss.backward()
            
            # Get gradient
            grad = x_adv.grad.data
            
            # Update with current step size
            with torch.no_grad():
                if self.base_pgd.targeted:
                    x_adv = x_adv - step_size * grad.sign()
                else:
                    x_adv = x_adv + step_size * grad.sign()
                
                x_adv = self.base_pgd._project(x, x_adv)
            
            # Decay step size
            step_size *= self.decay_factor
            
            if x_adv.grad is not None:
                x_adv.grad.zero_()
        
        x_adv = x_adv.detach()
        
        if return_logits:
            with torch.no_grad():
                if hasattr(self.base_pgd.model, 'backbone'):
                    logits_adv, _, _ = self.base_pgd.model(x_adv)
                else:
                    logits_adv = self.base_pgd.model(x_adv)
            return x_adv, logits_adv
        
        return x_adv


if __name__ == "__main__":
    # Test PGD attack
    print("Testing PGD Attack...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2)
    )
    model.eval()
    
    # Create PGD attacker
    pgd = PGD(model, epsilon=0.031, step_size=0.007, num_steps=10, random_start=True)
    
    # Generate adversarial examples
    x = torch.randn(4, 3, 96, 96)
    y = torch.randint(0, 2, (4,))
    
    # Get clean predictions
    with torch.no_grad():
        logits_clean = model(x)
        pred_clean = logits_clean.argmax(dim=1)
    
    # Generate adversarial examples
    print("Generating adversarial examples with PGD...")
    x_adv = pgd.generate(x, y)
    
    # Get adversarial predictions
    with torch.no_grad():
        logits_adv = model(x_adv)
        pred_adv = logits_adv.argmax(dim=1)
    
    # Compute perturbation
    perturbation = (x_adv - x).abs()
    max_pert = perturbation.max().item()
    mean_pert = perturbation.mean().item()
    linf_pert = perturbation.view(x.size(0), -1).max(dim=1)[0].mean().item()
    
    print(f"Clean predictions: {pred_clean.tolist()}")
    print(f"Adversarial predictions: {pred_adv.tolist()}")
    print(f"Max perturbation: {max_pert:.4f}")
    print(f"Mean perturbation: {mean_pert:.4f}")
    print(f"L-inf perturbation: {linf_pert:.4f}")
    print(f"Attack success rate: {(pred_clean != pred_adv).float().mean().item():.2%}")
    
    # Test PGD-L2
    print("\nTesting PGD-L2 Attack...")
    pgd_l2 = PGD_L2(model, epsilon=1.0, step_size=0.2, num_steps=10)
    x_adv_l2 = pgd_l2.generate(x, y)
    
    with torch.no_grad():
        logits_adv_l2 = model(x_adv_l2)
        pred_adv_l2 = logits_adv_l2.argmax(dim=1)
    
    perturbation_l2 = (x_adv_l2 - x).view(x.size(0), -1)
    l2_norm = perturbation_l2.norm(p=2, dim=1).mean().item()
    
    print(f"Clean predictions: {pred_clean.tolist()}")
    print(f"Adversarial predictions (L2): {pred_adv_l2.tolist()}")
    print(f"L2 perturbation norm: {l2_norm:.4f}")
    print(f"Attack success rate: {(pred_clean != pred_adv_l2).float().mean().item():.2%}")
    
    # Test Adaptive PGD
    print("\nTesting Adaptive PGD...")
    pgd_adaptive = PGD_Adaptive(model, epsilon=0.031, initial_step_size=0.01, num_steps=20)
    x_adv_adaptive = pgd_adaptive.generate(x, y)
    
    with torch.no_grad():
        logits_adv_adaptive = model(x_adv_adaptive)
        pred_adv_adaptive = logits_adv_adaptive.argmax(dim=1)
    
    print(f"Clean predictions: {pred_clean.tolist()}")
    print(f"Adversarial predictions (adaptive): {pred_adv_adaptive.tolist()}")
    print(f"Attack success rate: {(pred_clean != pred_adv_adaptive).float().mean().item():.2%}")