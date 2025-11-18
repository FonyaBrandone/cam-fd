"""
Feature Denoising Loss for representation invariance.
Reconstructs clean features from adversarial perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DenoisingLoss(nn.Module):
    """
    Feature denoising loss that enforces representation invariance.
    
    Measures the distance between clean features and reconstructed 
    (denoised) adversarial features, encouraging the model to learn
    stable representations under perturbations.
    """
    
    def __init__(
        self,
        lambda_rec: float = 0.5,
        loss_type: str = 'mse',
        normalize: bool = False,
        layer_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            lambda_rec: Weight for reconstruction loss
            loss_type: Type of reconstruction loss ('mse', 'l1', 'smooth_l1', 'cosine')
            normalize: Whether to normalize features before computing loss
            layer_weights: Optional weights for different layers
        """
        super().__init__()
        self.lambda_rec = lambda_rec
        self.loss_type = loss_type
        self.normalize = normalize
        self.layer_weights = layer_weights or {}
    
    def compute_distance(
        self,
        features_clean: torch.Tensor,
        features_denoised: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance between clean and denoised features.
        
        Args:
            features_clean: Clean features (B, C, H, W) or (B, C)
            features_denoised: Denoised adversarial features (same shape)
        
        Returns:
            Scalar loss value
        """
        if self.normalize:
            # L2 normalize features
            features_clean = F.normalize(features_clean.flatten(1), p=2, dim=1)
            features_denoised = F.normalize(features_denoised.flatten(1), p=2, dim=1)
        else:
            # Flatten spatial dimensions if present
            if len(features_clean.shape) == 4:
                features_clean = features_clean.flatten(1)
                features_denoised = features_denoised.flatten(1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(features_denoised, features_clean)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(features_denoised, features_clean)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(features_denoised, features_clean)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_similarity)
            loss = 1 - F.cosine_similarity(features_denoised, features_clean, dim=1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(
        self,
        features_clean: Dict[str, torch.Tensor],
        features_denoised: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-layer denoising loss.
        
        Args:
            features_clean: Dict of clean features from multiple layers
            features_denoised: Dict of denoised adversarial features
        
        Returns:
            Weighted reconstruction loss
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_name in features_clean.keys():
            if layer_name not in features_denoised:
                continue
            
            # Compute loss for this layer
            layer_loss = self.compute_distance(
                features_clean[layer_name],
                features_denoised[layer_name]
            )
            
            # Apply layer-specific weight if provided
            weight = self.layer_weights.get(layer_name, 1.0)
            total_loss += weight * layer_loss
            num_layers += 1
        
        # Average over layers and apply lambda
        if num_layers > 0:
            total_loss = self.lambda_rec * (total_loss / num_layers)
        
        return total_loss


class PerceptualDenoisingLoss(nn.Module):
    """
    Perceptual denoising loss using pre-trained features.
    More suitable for medical imaging where semantic features matter.
    """
    
    def __init__(
        self,
        lambda_rec: float = 0.5,
        feature_extractor: Optional[nn.Module] = None
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        
        # Use a pre-trained network as feature extractor
        if feature_extractor is None:
            # Default: use ImageNet pre-trained VGG features
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features[:23]))
            
            # Freeze feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = feature_extractor
    
    def forward(
        self,
        images_clean: torch.Tensor,
        images_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss between clean and reconstructed images.
        
        Args:
            images_clean: Clean images (B, C, H, W)
            images_reconstructed: Reconstructed images (B, C, H, W)
        
        Returns:
            Perceptual loss
        """
        # Extract features
        feat_clean = self.feature_extractor(images_clean)
        feat_reconstructed = self.feature_extractor(images_reconstructed)
        
        # MSE on features
        loss = F.mse_loss(feat_reconstructed, feat_clean)
        
        return self.lambda_rec * loss


class ContrastiveDenoisingLoss(nn.Module):
    """
    Contrastive denoising loss that encourages:
    1. Clean and denoised adversarial features to be close
    2. Adversarial features from different classes to be far apart
    """
    
    def __init__(
        self,
        lambda_rec: float = 0.5,
        temperature: float = 0.1,
        margin: float = 1.0
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        features_clean: torch.Tensor,
        features_denoised: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive denoising loss.
        
        Args:
            features_clean: Clean features (B, D)
            features_denoised: Denoised features (B, D)
            labels: Class labels (B,)
        
        Returns:
            Contrastive loss
        """
        batch_size = features_clean.size(0)
        
        # Normalize features
        features_clean = F.normalize(features_clean, p=2, dim=1)
        features_denoised = F.normalize(features_denoised, p=2, dim=1)
        
        # Positive pairs: clean and denoised from same sample
        pos_loss = F.mse_loss(features_denoised, features_clean)
        
        # Negative pairs: features from different classes
        # Create label mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features_denoised, features_clean.T) / self.temperature
        
        # Mask out diagonal and same-class pairs
        mask = mask * (1 - torch.eye(batch_size, device=mask.device))
        
        # Negative loss: encourage different classes to be far apart
        neg_loss = (mask * torch.exp(sim_matrix)).sum(1) / (mask.sum(1) + 1e-8)
        neg_loss = -torch.log(1 / (1 + neg_loss)).mean()
        
        total_loss = pos_loss + neg_loss
        
        return self.lambda_rec * total_loss


if __name__ == "__main__":
    # Test denoising loss
    print("Testing Denoising Loss...")
    
    denoising_loss = DenoisingLoss(lambda_rec=0.5, loss_type='mse')
    
    # Create dummy features
    features_clean = {
        'layer3': torch.randn(4, 1024, 12, 12),
        'layer4': torch.randn(4, 2048, 6, 6)
    }
    features_denoised = {
        'layer3': torch.randn(4, 1024, 12, 12),
        'layer4': torch.randn(4, 2048, 6, 6)
    }
    
    # Compute loss
    loss = denoising_loss(features_clean, features_denoised)
    print(f"Denoising loss (MSE): {loss.item():.4f}")
    
    # Test with different loss types
    for loss_type in ['l1', 'smooth_l1', 'cosine']:
        denoising_loss_type = DenoisingLoss(lambda_rec=0.5, loss_type=loss_type)
        loss = denoising_loss_type(features_clean, features_denoised)
        print(f"Denoising loss ({loss_type}): {loss.item():.4f}")
    
    # Test with layer weights
    print("\nTesting with layer weights...")
    layer_weights = {'layer3': 0.3, 'layer4': 0.7}
    weighted_loss = DenoisingLoss(lambda_rec=0.5, loss_type='mse', layer_weights=layer_weights)
    loss = weighted_loss(features_clean, features_denoised)
    print(f"Weighted denoising loss: {loss.item():.4f}")
    
    # Test contrastive loss
    print("\nTesting Contrastive Denoising Loss...")
    contrastive_loss = ContrastiveDenoisingLoss(lambda_rec=0.5)
    feat_clean = torch.randn(8, 512)
    feat_denoised = torch.randn(8, 512)
    labels = torch.randint(0, 2, (8,))
    loss = contrastive_loss(feat_clean, feat_denoised, labels)
    print(f"Contrastive denoising loss: {loss.item():.4f}")