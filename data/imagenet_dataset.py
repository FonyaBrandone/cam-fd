import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class ImageNetDataset(Dataset):
    """
    ImageNet Dataset loader for ILSVRC2012
    
    Expected directory structure:
    root/
        train/
            n01440764/
                n01440764_10026.JPEG
                ...
            n01443537/
                ...
        val/
            n01440764/
                ILSVRC2012_val_00000293.JPEG
                ...
            n01443537/
                ...
        imagenet_class_index.json (optional, for class name mapping)
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of ImageNet dataset
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set split directory
        self.split_dir = os.path.join(root_dir, split)
        
        # Get all class folders (synsets)
        self.classes = sorted([d for d in os.listdir(self.split_dir) 
                              if os.path.isdir(os.path.join(self.split_dir, d))])
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build image paths and labels
        self.samples = []
        self._build_dataset()
        
        # Load class names if available
        self.class_names = self._load_class_names()
        
    def _build_dataset(self):
        """Build list of (image_path, class_index) tuples"""
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def _load_class_names(self):
        """Load human-readable class names from JSON file"""
        json_path = os.path.join(self.root_dir, 'imagenet_class_index.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                class_index = json.load(f)
                # Convert to dict mapping synset to name
                return {v[0]: v[1] for k, v in class_index.items()}
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, class_idx):
        """Get human-readable class name for a given index"""
        synset = self.classes[class_idx]
        if self.class_names:
            return self.class_names.get(synset, synset)
        return synset


def get_imagenet_transforms(split='train', input_size=224):
    """
    Get standard ImageNet transforms
    
    Args:
        split (str): 'train' or 'val'
        input_size (int): Input image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_imagenet_dataloaders(root_dir, batch_size=256, num_workers=4, 
                             input_size=224, pin_memory=True):
    """
    Create ImageNet train and validation dataloaders
    
    Args:
        root_dir (str): Root directory of ImageNet dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        input_size (int): Input image size
        pin_memory (bool): Whether to pin memory for faster GPU transfer
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create transforms
    train_transform = get_imagenet_transforms('train', input_size)
    val_transform = get_imagenet_transforms('val', input_size)
    
    # Create datasets
    train_dataset = ImageNetDataset(root_dir, split='train', transform=train_transform)
    val_dataset = ImageNetDataset(root_dir, split='val', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
    print(f"Val dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes")
    
    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    root_dir = "/path/to/imagenet"  # Update this path
    
    # Create dataset
    train_transform = get_imagenet_transforms('train')
    dataset = ImageNetDataset(root_dir, split='train', transform=train_transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"First few classes: {dataset.classes[:5]}")
    
    # Test loading a sample
    img, label = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")
    print(f"Class name: {dataset.get_class_name(label)}")
    
    # Test dataloader
    train_loader, val_loader = get_imagenet_dataloaders(
        root_dir, 
        batch_size=32, 
        num_workers=2
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")