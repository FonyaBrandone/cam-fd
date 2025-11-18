"""
AutoAttack wrapper for robust evaluation.
"""

import torch
import torch.nn as nn
from typing import Optional


class AutoAttackWrapper:
    """
    Wrapper for AutoAttack evaluation.
    
    AutoAttack is an ensemble of diverse parameter-free attacks
    that is considered the gold standard for adversarial robustness evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,
        norm: str = 'Linf',
        version: str = 'standard',
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model to attack
            epsilon: Perturbation budget
            norm: Perturbation norm ('Linf' or 'L2')
            version: AutoAttack version ('standard', 'plus', 'rand')
            device: Device to run on
        """
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError(
                "autoattack not installed. Install with: pip install autoattack"
            )
        
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.version = version
        self.device = device
        
        # Initialize AutoAttack
        self.adversary = AutoAttack(
            model,
            norm=norm,
            eps=epsilon,
            version=version,
            device=device
        )
    
    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Run AutoAttack on input.
        
        Args:
            x: Clean images (N, C, H, W)
            y: True labels (N,)
            batch_size: Batch size for evaluation
        
        Returns:
            Adversarial examples
        """
        self.model.eval()
        
        # Run AutoAttack
        x_adv = self.adversary.run_standard_evaluation(
            x, y, bs=batch_size
        )
        
        return x_adv
    
    def evaluate_accuracy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 128
    ) -> float:
        """
        Evaluate robust accuracy under AutoAttack.
        
        Args:
            x: Clean images
            y: True labels
            batch_size: Batch size
        
        Returns:
            Robust accuracy
        """
        # Generate adversarial examples
        x_adv = self.run(x, y, batch_size=batch_size)
        
        # Evaluate accuracy
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'backbone'):
                logits, _, _ = self.model(x_adv)
            else:
                logits = self.model(x_adv)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
        
        return acc


if __name__ == "__main__":
    print("AutoAttack wrapper - requires 'pip install autoattack'")