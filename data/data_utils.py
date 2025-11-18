"""
Data utilities and helper functions.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from torchvision import transforms
import matplotlib.pyplot as plt


def compute_mean_std(dataset, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of dataset for normalization.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to use for computation
    
    Returns:
        mean: (3,) array of channel means
        std: (3,) array of channel stds
    """
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Accumulate pixel values
    pixel_sum = np.zeros(3)
    pixel_sum_sq = np.zeros(3)
    num_pixels = 0
    
    for idx in indices:
        img, _ = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # Reshape to (3, H*W)
        img = img.reshape(3, -1)
        pixel_sum += img.sum(axis=1)
        pixel_sum_sq += (img ** 2).sum(axis=1)
        num_pixels += img.shape[1]
    
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sum_sq / num_pixels - mean ** 2)
    
    return mean, std


def denormalize(
    tensor: torch.Tensor, 
    mean: Tuple[float, float, float], 
    std: Tuple[float, float, float]
) -> torch.Tensor:
    """
    Denormalize a normalized tensor.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    device = tensor.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def normalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float]
) -> torch.Tensor:
    """
    Normalize a tensor.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        mean: Mean for normalization
        std: Std for normalization
    
    Returns:
        Normalized tensor
    """
    device = tensor.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return (tensor - mean) / std


def save_image_grid(
    images: torch.Tensor, 
    path: str, 
    nrow: int = 8, 
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None
):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images (B, C, H, W)
        path: Path to save image
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
        value_range: Range for normalization
    """
    from torchvision.utils import save_image
    save_image(images, path, nrow=nrow, normalize=normalize, value_range=value_range)


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    num_images: int = 8,
    class_names: List[str] = None
):
    """
    Visualize a batch of images with labels and predictions.
    
    Args:
        images: Image tensor (B, C, H, W)
        labels: Label tensor (B,)
        predictions: Optional prediction tensor (B,)
        mean: Mean for denormalization
        std: Std for denormalization
        num_images: Number of images to show
        class_names: List of class names
    """
    num_images = min(num_images, len(images))
    
    # Denormalize
    images = denormalize(images[:num_images], mean, std)
    images = torch.clamp(images, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    
    for idx in range(num_images):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        
        # Create title
        label = labels[idx].item()
        if class_names:
            label_str = class_names[label]
        else:
            label_str = str(label)
        
        if predictions is not None:
            pred = predictions[idx].item()
            if class_names:
                pred_str = class_names[pred]
            else:
                pred_str = str(pred)
            
            color = 'green' if pred == label else 'red'
            title = f"True: {label_str}\nPred: {pred_str}"
            axes[idx].set_title(title, color=color)
        else:
            axes[idx].set_title(f"Label: {label_str}")
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


class Cutout:
    """
    Cutout data augmentation.
    Randomly mask out square regions of the image.
    
    Reference: DeVries & Taylor, "Improved Regularization of Convolutional 
    Neural Networks with Cutout", 2017
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Args:
            n_holes: Number of holes to cut
            length: Side length of holes
        """
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image (C, H, W)
        
        Returns:
            Image with cutout applied
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.0
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


class RandomErasing:
    """
    Random Erasing augmentation.
    Similar to Cutout but with variable size and aspect ratio.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        min_area: float = 0.02,
        max_area: float = 0.4,
        min_aspect: float = 0.3,
        max_aspect: float = 3.3
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.probability:
            return img
        
        for _ in range(100):  # Try 100 times
            area = img.size(1) * img.size(2)
            
            target_area = np.random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = np.random.uniform(self.min_aspect, self.max_aspect)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size(2) and h < img.size(1):
                x1 = np.random.randint(0, img.size(1) - h)
                y1 = np.random.randint(0, img.size(2) - w)
                
                img[:, x1:x1+h, y1:y1+w] = 0
                return img
        
        return img


def split_dataset(
    dataset_size: int,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into train/val/test indices.
    
    Args:
        dataset_size: Total dataset size
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = list(range(dataset_size))
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Calculate split points
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    print("Testing data utilities...")
    
    # Test denormalization
    img = torch.randn(3, 96, 96)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    normalized = normalize(img, mean, std)
    denormalized = denormalize(normalized, mean, std)
    
    assert torch.allclose(img, denormalized, atol=1e-6)
    print("✓ Normalize/denormalize test passed")
    
    # Test Cutout
    cutout = Cutout(n_holes=1, length=16)
    img_cutout = cutout(img)
    print(f"✓ Cutout test passed: {img_cutout.shape}")
    
    # Test split
    train_idx, val_idx, test_idx = split_dataset(1000, 0.2, 0.1)
    print(f"✓ Split test passed: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    print("\n✓ Data utilities test complete!")