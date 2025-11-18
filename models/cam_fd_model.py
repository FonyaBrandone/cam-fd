"""
Complete CAM-FD Model integrating backbone and feature denoiser.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .backbones import get_backbone, get_feature_dims, BackboneWrapper
from .feature_denoiser import MultiScaleDenoiser


class CAMFDModel(nn.Module):
    """
    Complete CAM-FD model with backbone and feature denoiser.
    
    This model:
    1. Extracts features from multiple layers using the backbone
    2. Applies feature denoising to adversarial features
    3. Returns both predictions and features for loss computation
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        feature_layers: Optional[list] = None,
        img_size: int = 96,
        # Denoiser config
        enable_denoiser: bool = True,
        denoiser_type: str = "autoencoder",
        denoiser_hidden_dim: int = 512,
        denoiser_num_blocks: int = 3,
    ):
        super().__init__()
        
        # Create backbone with feature extraction
        self.backbone = get_backbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            feature_layers=feature_layers,
            img_size=img_size
        )
        
        self.num_classes = num_classes
        self.enable_denoiser = enable_denoiser
        
        # Create feature denoiser if enabled
        if enable_denoiser:
            # Get feature dimensions by running a dummy forward pass
            feature_dims = get_feature_dims(self.backbone, img_size=img_size)
            
            self.denoiser = MultiScaleDenoiser(
                feature_dims=feature_dims,
                hidden_dim=denoiser_hidden_dim,
                num_blocks=denoiser_num_blocks,
                denoiser_type=denoiser_type
            )
        else:
            self.denoiser = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        denoise_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features
            denoise_features: Whether to apply denoising to features
        
        Returns:
            logits: Model predictions (B, num_classes)
            features: Dict of intermediate features (if return_features=True)
            denoised_features: Dict of denoised features (if denoise_features=True)
        """
        # Forward through backbone
        logits, features = self.backbone(x)
        
        # Apply denoising if requested
        denoised_features = None
        if denoise_features and self.enable_denoiser and self.denoiser is not None:
            denoised_features = self.denoiser(features)
        
        if not return_features:
            features = None
        
        return logits, features, denoised_features
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract only intermediate features without final prediction."""
        _, features = self.backbone(x)
        return features
    
    def denoise_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply denoising to given features."""
        if not self.enable_denoiser or self.denoiser is None:
            return features
        return self.denoiser(features)


def create_cam_fd_model(config: dict) -> CAMFDModel:
    """
    Factory function to create CAM-FD model from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        CAMFDModel instance
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    denoiser_config = model_config.get('denoiser', {})
    
    model = CAMFDModel(
        backbone_name=model_config.get('backbone', 'resnet50'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        feature_layers=denoiser_config.get('feature_layers', None),
        img_size=data_config.get('image_size', 96),
        enable_denoiser=denoiser_config.get('enabled', True),
        denoiser_type=denoiser_config.get('architecture', 'autoencoder'),
        denoiser_hidden_dim=denoiser_config.get('hidden_dim', 512),
        denoiser_num_blocks=denoiser_config.get('num_blocks', 3),
    )
    
    return model


if __name__ == "__main__":
    # Test CAM-FD model
    print("Testing CAM-FD Model...")
    
    model = CAMFDModel(
        backbone_name="resnet50",
        num_classes=2,
        pretrained=False,
        enable_denoiser=True,
        denoiser_type="autoencoder"
    )
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)
    
    # Standard forward
    logits, _, _ = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # Forward with features
    logits, features, _ = model(x, return_features=True)
    print(f"\nFeatures extracted:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Forward with denoising
    logits, features, denoised = model(x, return_features=True, denoise_features=True)
    print(f"\nDenoised features:")
    for name, feat in denoised.items():
        print(f"  {name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")