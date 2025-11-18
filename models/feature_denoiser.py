"""
Feature Denoising Module for CAM-FD.
Reconstructs clean features from adversarially perturbed features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AutoencoderDenoiser(nn.Module):
    """
    Autoencoder-based feature denoiser.
    Learns to reconstruct clean features from noisy/adversarial features.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        spatial_size: int = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        
        # Encoder: compress features
        encoder_layers = []
        current_channels = in_channels
        for i in range(num_blocks):
            out_channels = hidden_dim if i == num_blocks - 1 else min(hidden_dim, current_channels * 2)
            encoder_layers.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: reconstruct features
        decoder_layers = []
        for i in range(num_blocks):
            out_channels = in_channels if i == num_blocks - 1 else max(in_channels, current_channels // 2)
            decoder_layers.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise adversarial features.
        
        Args:
            x: Adversarial features (B, C, H, W)
        
        Returns:
            Reconstructed clean features (B, C, H, W)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class UNetDenoiser(nn.Module):
    """
    U-Net style feature denoiser with skip connections.
    Better preserves spatial information.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 512,
        num_blocks: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        
        # Encoder with skip connections
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        self.channel_sizes = [current_channels]
        
        for i in range(num_blocks):
            out_channels = min(hidden_dim, current_channels * 2)
            self.enc_blocks.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels
            self.channel_sizes.append(current_channels)
        
        # Bottleneck
        self.bottleneck = ConvBlock(current_channels, current_channels)
        
        # Decoder with skip connections
        self.dec_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Account for skip connection concatenation
            in_ch = current_channels + self.channel_sizes[-(i + 2)]
            out_channels = in_channels if i == num_blocks - 1 else max(in_channels, current_channels // 2)
            self.dec_blocks.append(ConvBlock(in_ch, out_channels))
            current_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise with skip connections.
        
        Args:
            x: Adversarial features (B, C, H, W)
        
        Returns:
            Reconstructed clean features (B, C, H, W)
        """
        # Encoder with skip connections
        skips = []
        h = x
        for enc_block in self.enc_blocks:
            h = enc_block(h)
            skips.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, dec_block in enumerate(self.dec_blocks):
            skip = skips[-(i + 2)]  # Get corresponding skip connection
            h = torch.cat([h, skip], dim=1)  # Concatenate skip
            h = dec_block(h)
        
        return h


class MultiScaleDenoiser(nn.Module):
    """
    Denoises features from multiple layers simultaneously.
    Each layer has its own denoiser module.
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, Tuple[int, ...]],
        hidden_dim: int = 512,
        num_blocks: int = 3,
        denoiser_type: str = "autoencoder"
    ):
        super().__init__()
        self.feature_layers = list(feature_dims.keys())
        self.denoisers = nn.ModuleDict()
        
        for layer_name, dims in feature_dims.items():
            if len(dims) == 3:  # Conv features (C, H, W)
                in_channels = dims[0]
            elif len(dims) == 1:  # Transformer features (C,)
                in_channels = dims[0]
            else:
                raise ValueError(f"Unsupported feature dimension: {dims}")
            
            # Create denoiser for this layer
            if denoiser_type == "autoencoder":
                denoiser = AutoencoderDenoiser(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks
                )
            elif denoiser_type == "unet":
                denoiser = UNetDenoiser(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks
                )
            else:
                raise ValueError(f"Unknown denoiser type: {denoiser_type}")
            
            self.denoisers[layer_name] = denoiser
    
    def forward(
        self, 
        adversarial_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise features from multiple layers.
        
        Args:
            adversarial_features: Dict of adversarial features from each layer
        
        Returns:
            Dict of denoised features
        """
        denoised_features = {}
        for layer_name, adv_feat in adversarial_features.items():
            if layer_name in self.denoisers:
                denoised_features[layer_name] = self.denoisers[layer_name](adv_feat)
        
        return denoised_features


if __name__ == "__main__":
    # Test AutoencoderDenoiser
    print("Testing AutoencoderDenoiser...")
    denoiser = AutoencoderDenoiser(in_channels=512, hidden_dim=256, num_blocks=3)
    x = torch.randn(4, 512, 12, 12)
    out = denoiser(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test UNetDenoiser
    print("\nTesting UNetDenoiser...")
    unet_denoiser = UNetDenoiser(in_channels=512, hidden_dim=256, num_blocks=3)
    out = unet_denoiser(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test MultiScaleDenoiser
    print("\nTesting MultiScaleDenoiser...")
    feature_dims = {
        "layer3": (1024, 12, 12),
        "layer4": (2048, 6, 6)
    }
    multi_denoiser = MultiScaleDenoiser(
        feature_dims=feature_dims,
        hidden_dim=512,
        num_blocks=3,
        denoiser_type="autoencoder"
    )
    adv_features = {
        "layer3": torch.randn(4, 1024, 12, 12),
        "layer4": torch.randn(4, 2048, 6, 6)
    }
    denoised = multi_denoiser(adv_features)
    print(f"Denoised layers: {list(denoised.keys())}")
    for name, feat in denoised.items():
        print(f"  {name}: {feat.shape}")