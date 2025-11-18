"""
PCam (PatchCamelyon) dataset implementation.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional
from .data_loader import PCamDataLoader


class PCamDataset(Dataset):
    """
    PCam (PatchCamelyon) dataset for histopathology image classification.
    
    Dataset consists of 96x96 RGB patches from histopathologic scans of lymph node sections.
    Task: Binary classification of metastatic tissue.
    
    - Train: 262,144 samples
    - Validation: 32,768 samples  
    - Test: 32,768 samples
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Args:
            images: Image array (N, H, W, C) with values in [0, 255]
            labels: Label array (N,) with values {0, 1}
            transform: Optional transforms to apply
            normalize: Whether to normalize images
            mean: Mean for normalization
            std: Std for normalization
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Build normalization transform
        self.norm_transform = transforms.Normalize(mean=mean, std=std) if normalize else None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape (C, H, W)
            label: Tensor with single value {0, 1}
        """
        # Get image and label
        image = self.images[idx]  # (H, W, C) in [0, 255]
        label = self.labels[idx]  # scalar
        
        # Convert to PIL Image for transforms
        image = transforms.ToPILImage()(image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            image = transforms.ToTensor()(image)
        
        # Normalize if needed
        if self.normalize and self.norm_transform is not None:
            image = self.norm_transform(image)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


def get_pcam_transforms(
    img_size: int = 96,
    augment: bool = True,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get transforms for PCam dataset.
    
    Args:
        img_size: Image size (PCam is 96x96 by default)
        augment: Whether to apply data augmentation
        normalize: Whether to normalize
        mean: Mean for normalization
        std: Std for normalization
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        # Data augmentation for training
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def create_pcam_dataloaders(
    data_config: dict,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PCam dataloaders for train, validation, and test sets.
    
    Args:
        cloud_config: Cloud storage configuration
        data_config: Data configuration
        download: Whether to download data if not cached
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Initialize cloud loader
    if download:
        cloud_loader = PCamDataLoader(
            data_dir=data_config.get('data_dir', 'datasets/pcam')
        )
        
        # Download all splits
        print("Downloading PCam dataset...")
        cloud_loader.download_pcam(split='all')
        
        # Load data
        print("Loading data into memory...")
        x_train, y_train = cloud_loader.load_split('train')
        x_val, y_val = cloud_loader.load_split('valid')
        x_test, y_test = cloud_loader.load_split('test')
    else:
        raise ValueError("Local data loading not implemented. Set download=True")
    
    # Get transform parameters
    img_size = data_config.get('image_size', 96)
    aug_config = data_config.get('augmentation', {})
    mean = aug_config.get('mean', [0.485, 0.456, 0.406])
    std = aug_config.get('std', [0.229, 0.224, 0.225])
    normalize = aug_config.get('normalize', True)
    
    # Create transforms
    train_transform = get_pcam_transforms(
        img_size=img_size,
        augment=True,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    val_transform = get_pcam_transforms(
        img_size=img_size,
        augment=False,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    # Create datasets
    train_dataset = PCamDataset(
        images=x_train,
        labels=y_train,
        transform=train_transform,
        normalize=False,  # Already in transform
        mean=mean,
        std=std
    )
    
    val_dataset = PCamDataset(
        images=x_val,
        labels=y_val,
        transform=val_transform,
        normalize=False,
        mean=mean,
        std=std
    )
    
    test_dataset = PCamDataset(
        images=x_test,
        labels=y_test,
        transform=val_transform,
        normalize=False,
        mean=mean,
        std=std
    )
    
    # Create dataloaders
    batch_size = data_config.get('batch_size', 128)
    num_workers = data_config.get('num_workers', 8)
    pin_memory = data_config.get('pin_memory', True)
    prefetch_factor = data_config.get('prefetch_factor', 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\n✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test PCam dataset
    print("Testing PCam Dataset...")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    dummy_images = np.random.randint(0, 256, (100, 96, 96, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 2, (100,))
    
    # Create dataset
    dataset = PCamDataset(
        images=dummy_images,
        labels=dummy_labels,
        transform=get_pcam_transforms(augment=True),
        normalize=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test getitem
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label.item()}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    
    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
        if batch_idx >= 2:
            break
    
    print("\nPCam dataset test successful!")