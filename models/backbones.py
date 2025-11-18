"""
Backbone architectures for CAM-FD framework.
Supports ResNet, ConvNeXt, and ViT with feature extraction hooks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, List, Tuple
from collections import OrderedDict


class BackboneWrapper(nn.Module):
    """Base wrapper for extracting intermediate features from backbones."""
    
    def __init__(self, backbone: nn.Module, feature_layers: List[str]):
        super().__init__()
        self.backbone = backbone
        self.feature_layers = feature_layers
        self.features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        for name, module in self.backbone.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_activation(name))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with feature extraction.
        
        Returns:
            output: Final model output
            features: Dict of intermediate features
        """
        self.features = {}
        output = self.backbone(x)
        return output, self.features


class ResNetBackbone(BackboneWrapper):
    """ResNet backbone with feature extraction."""
    
    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        feature_layers: List[str] = None
    ):
        # Load ResNet
        if arch == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
        elif arch == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
        elif arch == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        elif arch == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet architecture: {arch}")
        
        # Modify final layer for num_classes
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        
        # Default feature layers
        if feature_layers is None:
            feature_layers = ["layer3", "layer4"]
        
        super().__init__(backbone, feature_layers)
        self.num_classes = num_classes
        self.arch = arch


class ConvNeXtBackbone(BackboneWrapper):
    """ConvNeXt backbone with feature extraction."""
    
    def __init__(
        self,
        arch: str = "convnext_base",
        num_classes: int = 2,
        pretrained: bool = True,
        feature_layers: List[str] = None
    ):
        # Load ConvNeXt using timm
        backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Default feature layers for ConvNeXt
        if feature_layers is None:
            feature_layers = ["stages.2", "stages.3"]
        
        super().__init__(backbone, feature_layers)
        self.num_classes = num_classes
        self.arch = arch


class ViTBackbone(BackboneWrapper):
    """Vision Transformer backbone with feature extraction."""
    
    def __init__(
        self,
        arch: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        feature_layers: List[str] = None,
        img_size: int = 96
    ):
        # Load ViT using timm
        backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )
        
        # Default feature layers for ViT
        if feature_layers is None:
            feature_layers = ["blocks.9", "blocks.11"]  # Later transformer blocks
        
        super().__init__(backbone, feature_layers)
        self.num_classes = num_classes
        self.arch = arch


def get_backbone(
    backbone_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    feature_layers: List[str] = None,
    img_size: int = 96
) -> BackboneWrapper:
    """
    Factory function to create backbone models.
    
    Args:
        backbone_name: Name of backbone (resnet50, convnext_base, vit_base, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        feature_layers: List of layer names to extract features from
        img_size: Input image size (for ViT)
    
    Returns:
        BackboneWrapper instance
    """
    if backbone_name.startswith("resnet"):
        return ResNetBackbone(
            arch=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            feature_layers=feature_layers
        )
    elif backbone_name.startswith("convnext"):
        return ConvNeXtBackbone(
            arch=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            feature_layers=feature_layers
        )
    elif backbone_name.startswith("vit"):
        return ViTBackbone(
            arch=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            feature_layers=feature_layers,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def get_feature_dims(backbone: BackboneWrapper, img_size: int = 96) -> Dict[str, Tuple[int, ...]]:
    """
    Get the dimensions of intermediate features by running a dummy forward pass.
    
    Args:
        backbone: Backbone model
        img_size: Input image size
    
    Returns:
        Dict mapping layer names to feature dimensions (C, H, W)
    """
    backbone.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size)
        _, features = backbone(dummy_input)
    
    feature_dims = {}
    for name, feat in features.items():
        # Handle different feature shapes (conv vs transformer)
        if len(feat.shape) == 4:  # Conv features: (B, C, H, W)
            feature_dims[name] = feat.shape[1:]
        elif len(feat.shape) == 3:  # Transformer features: (B, N, C)
            feature_dims[name] = (feat.shape[2],)  # Just channel dimension
        else:
            feature_dims[name] = feat.shape[1:]
    
    return feature_dims


if __name__ == "__main__":
    # Test backbone creation
    print("Testing ResNet50...")
    resnet = get_backbone("resnet50", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 96, 96)
    output, features = resnet(x)
    print(f"Output shape: {output.shape}")
    for name, feat in features.items():
        print(f"Feature {name}: {feat.shape}")
    
    print("\nTesting ConvNeXt...")
    convnext = get_backbone("convnext_base", num_classes=2, pretrained=False)
    output, features = convnext(x)
    print(f"Output shape: {output.shape}")
    for name, feat in features.items():
        print(f"Feature {name}: {feat.shape}")
    
    print("\nGetting feature dimensions...")
    dims = get_feature_dims(resnet, img_size=96)
    print(f"Feature dimensions: {dims}")