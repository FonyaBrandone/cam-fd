"""
Combined CAM-FD Loss integrating all components.
L = L_CE(mix) + λ_TR·KL(f(x)||f(x_adv)) + λ_rec·||F_adv - F_clean||² + λ_AWP·L_AWP
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .mixup_loss import MixupLoss
from .trades_loss import TRADESLoss
from .denoising_loss import DenoisingLoss
from .awp_loss import AWPLoss


class CAMFDLoss(nn.Module):
    """
    Complete CAM-FD loss function combining all four components:
    1. Cross-Entropy with Mixup (L_CE(mix))
    2. TRADES KL Divergence (λ_TR·KL)
    3. Feature Denoising (λ_rec·||F_adv - F_clean||²)
    4. Adversarial Weight Perturbation (λ_AWP·L_AWP)
    """
    
    def __init__(
        self,
        model: nn.Module,
        # Mixup config
        mixup_enabled: bool = True,
        mixup_alpha: float = 1.0,
        # TRADES config
        trades_enabled: bool = True,
        lambda_trades: float = 6.0,
        # Denoising config
        denoising_enabled: bool = True,
        lambda_rec: float = 0.5,
        denoising_loss_type: str = 'mse',
        layer_weights: Optional[Dict[str, float]] = None,
        # AWP config
        awp_enabled: bool = True,
        lambda_awp: float = 0.01,
        awp_gamma: float = 0.01,
        awp_start_epoch: int = 5,
        # General
        num_classes: int = 2
    ):
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        
        # Component flags
        self.mixup_enabled = mixup_enabled
        self.trades_enabled = trades_enabled
        self.denoising_enabled = denoising_enabled
        self.awp_enabled = awp_enabled
        
        # Initialize loss components
        if mixup_enabled:
            self.mixup_loss = MixupLoss(alpha=mixup_alpha, num_classes=num_classes)
        
        if trades_enabled:
            self.trades_loss = TRADESLoss(lambda_trades=lambda_trades)
        
        if denoising_enabled:
            self.denoising_loss = DenoisingLoss(
                lambda_rec=lambda_rec,
                loss_type=denoising_loss_type,
                layer_weights=layer_weights
            )
        
        if awp_enabled:
            self.awp_loss = AWPLoss(
                model=model,
                lambda_awp=lambda_awp,
                gamma=awp_gamma,
                start_epoch=awp_start_epoch
            )
    
    def forward(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        targets: torch.Tensor,
        apply_mixup: bool = True,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete CAM-FD loss.
        
        Args:
            x_clean: Clean input images (B, C, H, W)
            x_adv: Adversarial input images (B, C, H, W)
            targets: Ground truth labels (B,)
            apply_mixup: Whether to apply mixup (False during validation)
            return_components: Whether to return individual loss components
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # ===== Component 1: Cross-Entropy with Mixup =====
        if self.mixup_enabled:
            mixup_loss, mixup_info = self.mixup_loss(
                x_clean,
                targets,
                model=self.model,
                apply_mixup=apply_mixup
            )
            total_loss += mixup_loss
            loss_dict['loss_mixup'] = mixup_loss.item()
            
            # Get clean predictions for TRADES
            if 'mixed_input' in mixup_info and apply_mixup:
                x_for_forward = mixup_info['mixed_input']
            else:
                x_for_forward = x_clean
        else:
            # Standard cross-entropy
            logits_clean, _, _ = self.model(x_clean, return_features=True)
            ce_loss = nn.functional.cross_entropy(logits_clean, targets)
            total_loss += ce_loss
            loss_dict['loss_ce'] = ce_loss.item()
            x_for_forward = x_clean
        
        # ===== Get Clean and Adversarial Predictions with Features =====
        logits_clean, features_clean, _ = self.model(
            x_clean,
            return_features=True,
            denoise_features=False
        )
        
        logits_adv, features_adv, features_denoised = self.model(
            x_adv,
            return_features=True,
            denoise_features=self.denoising_enabled
        )
        
        # ===== Component 2: TRADES KL Divergence =====
        if self.trades_enabled:
            trades_loss = self.trades_loss(
                logits_clean,
                logits_adv,
                targets=None  # Natural loss already computed in mixup
            )
            total_loss += trades_loss
            loss_dict['loss_trades'] = trades_loss.item()
        
        # ===== Component 3: Feature Denoising =====
        if self.denoising_enabled and features_denoised is not None:
            denoising_loss = self.denoising_loss(
                features_clean,
                features_denoised
            )
            total_loss += denoising_loss
            loss_dict['loss_denoising'] = denoising_loss.item()
        
        # ===== Component 4: Adversarial Weight Perturbation =====
        if self.awp_enabled and self.awp_loss.is_enabled():
            # Define function to recompute loss with perturbed weights
            def compute_awp_loss():
                logits_awp, _, _ = self.model(x_adv)
                return nn.functional.cross_entropy(logits_awp, targets)
            
            # Compute base loss for AWP
            base_loss_for_awp = nn.functional.cross_entropy(logits_adv, targets)
            
            # Compute AWP loss
            awp_loss_value = self.awp_loss(
                base_loss_for_awp,
                compute_awp_loss
            )
            total_loss += awp_loss_value
            loss_dict['loss_awp'] = awp_loss_value.item()
        
        # Add total loss
        loss_dict['loss_total'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_dict
        
        return total_loss, loss_dict
    
    def update_epoch(self, epoch: int):
        """Update epoch for components that use curriculum learning."""
        if self.awp_enabled:
            self.awp_loss.current_epoch = epoch


class CAMFDLossFactory:
    """Factory for creating CAM-FD loss from configuration."""
    
    @staticmethod
    def from_config(model: nn.Module, config: dict) -> CAMFDLoss:
        """
        Create CAM-FD loss from configuration dictionary.
        
        Args:
            model: Model to apply loss to
            config: Configuration dictionary
        
        Returns:
            CAMFDLoss instance
        """
        loss_config = config.get('loss', {})
        
        # Extract component configs
        mixup_config = loss_config.get('mixup', {})
        trades_config = loss_config.get('trades', {})
        denoising_config = loss_config.get('denoising', {})
        awp_config = loss_config.get('awp', {})
        
        return CAMFDLoss(
            model=model,
            # Mixup
            mixup_enabled=mixup_config.get('enabled', True),
            mixup_alpha=mixup_config.get('alpha', 1.0),
            # TRADES
            trades_enabled=trades_config.get('enabled', True),
            lambda_trades=trades_config.get('lambda_tr', 6.0),
            # Denoising
            denoising_enabled=denoising_config.get('enabled', True),
            lambda_rec=denoising_config.get('lambda_rec', 0.5),
            denoising_loss_type=denoising_config.get('loss_type', 'mse'),
            # AWP
            awp_enabled=awp_config.get('enabled', True),
            lambda_awp=awp_config.get('lambda_awp', 0.01),
            awp_gamma=awp_config.get('gamma', 0.01),
            awp_start_epoch=awp_config.get('start_epoch', 5),
            # General
            num_classes=config.get('model', {}).get('num_classes', 2)
        )


if __name__ == "__main__":
    # Test combined CAM-FD loss
    print("Testing CAM-FD Combined Loss...")
    
    # Import model
    import sys
    sys.path.append('..')
    from models.cam_fd_model import CAMFDModel
    
    # Create model
    model = CAMFDModel(
        backbone_name="resnet50",
        num_classes=2,
        pretrained=False,
        enable_denoiser=True
    )
    
    # Create loss function
    cam_fd_loss = CAMFDLoss(
        model=model,
        mixup_enabled=True,
        mixup_alpha=1.0,
        trades_enabled=True,
        lambda_trades=6.0,
        denoising_enabled=True,
        lambda_rec=0.5,
        awp_enabled=True,
        lambda_awp=0.01,
        awp_start_epoch=0,
        num_classes=2
    )
    
    # Enable AWP
    cam_fd_loss.awp_loss.current_epoch = 5
    
    # Dummy data
    batch_size = 4
    x_clean = torch.randn(batch_size, 3, 96, 96)
    x_adv = x_clean + 0.03 * torch.randn_like(x_clean)  # Simulated adversarial
    targets = torch.randint(0, 2, (batch_size,))
    
    # Compute loss
    print("\nComputing CAM-FD loss with all components...")
    total_loss, loss_dict = cam_fd_loss(
        x_clean,
        x_adv,
        targets,
        apply_mixup=True,
        return_components=True
    )
    
    print(f"\nLoss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test with individual components disabled
    print("\n" + "="*50)
    print("Testing with individual components disabled...")
    
    # Test without mixup
    cam_fd_loss.mixup_enabled = False
    total_loss, loss_dict = cam_fd_loss(x_clean, x_adv, targets)
    print(f"\nWithout Mixup - Total: {loss_dict['loss_total']:.4f}")
    
    # Test without TRADES
    cam_fd_loss.mixup_enabled = True
    cam_fd_loss.trades_enabled = False
    total_loss, loss_dict = cam_fd_loss(x_clean, x_adv, targets)
    print(f"Without TRADES - Total: {loss_dict['loss_total']:.4f}")
    
    # Test without denoising
    cam_fd_loss.trades_enabled = True
    cam_fd_loss.denoising_enabled = False
    total_loss, loss_dict = cam_fd_loss(x_clean, x_adv, targets)
    print(f"Without Denoising - Total: {loss_dict['loss_total']:.4f}")
    
    # Test without AWP
    cam_fd_loss.denoising_enabled = True
    cam_fd_loss.awp_enabled = False
    total_loss, loss_dict = cam_fd_loss(x_clean, x_adv, targets)
    print(f"Without AWP - Total: {loss_dict['loss_total']:.4f}")
    
    # Test backward pass
    print("\n" + "="*50)
    print("Testing backward pass...")
    cam_fd_loss.mixup_enabled = True
    cam_fd_loss.trades_enabled = True
    cam_fd_loss.denoising_enabled = True
    cam_fd_loss.awp_enabled = False  # Disable AWP for simpler test
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    total_loss, _ = cam_fd_loss(x_clean, x_adv, targets)
    total_loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"Gradient norm: {grad_norm:.4f}")
    print("✓ Backward pass successful!")
    
    # Test factory method
    print("\n" + "="*50)
    print("Testing CAMFDLossFactory...")
    
    config = {
        'loss': {
            'mixup': {'enabled': True, 'alpha': 1.0},
            'trades': {'enabled': True, 'lambda_tr': 6.0},
            'denoising': {'enabled': True, 'lambda_rec': 0.5},
            'awp': {'enabled': True, 'lambda_awp': 0.01, 'start_epoch': 5}
        },
        'model': {'num_classes': 2}
    }
    
    factory_loss = CAMFDLossFactory.from_config(model, config)
    total_loss, loss_dict = factory_loss(x_clean, x_adv, targets)
    print(f"Factory-created loss - Total: {loss_dict['loss_total']:.4f}")
    print("✓ Factory method successful!")