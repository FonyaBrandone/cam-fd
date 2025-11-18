"""
Data loader for PCam dataset - Google Colab compatible.
Downloads data directly from public sources.
"""

import os
import h5py
import gdown
import numpy as np
from pathlib import Path
from tqdm import tqdm
import urllib.request
from typing import Tuple, Optional


class PCamDataLoader:
    """
    PCam dataset loader - downloads from public URLs.
    Compatible with Google Colab and local environments.
    """
    
    # PCam file IDs from Google Drive or direct URLs
    FILE_URLS = {
        'train_x': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz',
        'train_y': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz',
        'val_x': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz',
        'val_y': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz',
        'test_x': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz',
        'test_y': 'https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz'
    }
    
    def __init__(self, data_dir: str = "pcam_data"):
        """
        Initialize PCam data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        print(f"Data directory: {self.data_dir}")
    
    def download_file(self, url: str, filename: str, force: bool = False) -> Path:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            filename: Local filename
            force: Force re-download
        
        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename
        
        # Check if already exists
        if filepath.exists() and not force:
            print(f"✓ {filename} already exists")
            return filepath
        
        print(f"Downloading {filename}...")
        
        # Download with progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
        
        print(f"✓ Downloaded {filename}")
        return filepath
    
    def decompress_gz(self, gz_path: Path) -> Path:
        """
        Decompress .gz file.
        
        Args:
            gz_path: Path to .gz file
        
        Returns:
            Path to decompressed file
        """
        import gzip
        import shutil
        
        output_path = gz_path.with_suffix('')  # Remove .gz extension
        
        if output_path.exists():
            print(f"✓ {output_path.name} already decompressed")
            return output_path
        
        print(f"Decompressing {gz_path.name}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"✓ Decompressed to {output_path.name}")
        return output_path
    
    def download_pcam(self, split: str = 'all', force: bool = False) -> dict:
        """
        Download PCam dataset files.
        
        Args:
            split: Which split to download ('train', 'val', 'test', 'all')
            force: Force re-download
        
        Returns:
            Dictionary mapping split names to file paths
        """
        print("\n" + "="*70)
        print("Downloading PCam Dataset")
        print("="*70)
        
        files_to_download = {}
        
        # Determine which files to download
        if split == 'all':
            files_to_download = self.FILE_URLS
        else:
            files_to_download = {
                k: v for k, v in self.FILE_URLS.items()
                if k.startswith(split)
            }
        
        downloaded_files = {}
        
        # Download and decompress each file
        for key, url in files_to_download.items():
            # Download .gz file
            gz_filename = url.split('/')[-1]
            gz_path = self.download_file(url, gz_filename, force=force)
            
            # Decompress
            h5_path = self.decompress_gz(gz_path)
            
            downloaded_files[key] = str(h5_path)
        
        print("\n" + "="*70)
        print("Download Complete!")
        print("="*70)
        
        return downloaded_files
    
    def load_split(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load PCam data split into memory.
        
        Args:
            split: Split to load ('train', 'valid', 'test')
        
        Returns:
            Tuple of (images, labels)
                images: (N, 96, 96, 3) uint8 array
                labels: (N,) int array
        """
        x_filename = f"camelyonpatch_level_2_split_{split}_x.h5"
        y_filename = f"camelyonpatch_level_2_split_{split}_y.h5"
        
        x_path = self.data_dir / x_filename
        y_path = self.data_dir / y_filename
        
        # Check if files exist
        if not x_path.exists() or not y_path.exists():
            print(f"Files not found. Downloading {split} split...")
            self.download_pcam(split=split)
        
        print(f"Loading {split} data into memory...")
        
        # Load images
        with h5py.File(x_path, 'r') as f:
            x = f['x'][:]
            print(f"  Images: {x.shape}")
        
        # Load labels
        with h5py.File(y_path, 'r') as f:
            y = f['y'][:]
            if len(y.shape) > 1:
                y = y.squeeze()
            print(f"  Labels: {y.shape}")
        
        print(f"✓ Loaded {split} split: {x.shape[0]:,} samples")
        
        return x, y
    
    def get_dataset_info(self) -> dict:
        """Get information about downloaded dataset."""
        info = {
            'train': None,
            'valid': None,
            'test': None
        }
        
        for split in ['train', 'valid', 'test']:
            x_file = self.data_dir / f"camelyonpatch_level_2_split_{split}_x.h5"
            if x_file.exists():
                with h5py.File(x_file, 'r') as f:
                    info[split] = {
                        'num_samples': f['x'].shape[0],
                        'image_shape': f['x'].shape[1:],
                        'file_size_mb': x_file.stat().st_size / (1024 * 1024)
                    }
        
        return info
    
    def verify_data(self, split: str = 'train') -> bool:
        """
        Verify data integrity.
        
        Args:
            split: Split to verify
        
        Returns:
            True if data is valid
        """
        try:
            x, y = self.load_split(split)
            
            # Check shapes
            assert x.shape[1:] == (96, 96, 3), f"Invalid image shape: {x.shape}"
            assert len(y.shape) == 1, f"Invalid label shape: {y.shape}"
            assert x.shape[0] == y.shape[0], "Mismatch between images and labels"
            
            # Check data ranges
            assert x.dtype == np.uint8, f"Invalid image dtype: {x.dtype}"
            assert x.min() >= 0 and x.max() <= 255, "Images out of valid range"
            assert np.all((y == 0) | (y == 1)), "Labels not binary"
            
            # Check class distribution
            num_positive = (y == 1).sum()
            num_negative = (y == 0).sum()
            print(f"\n{split.capitalize()} split statistics:")
            print(f"  Total samples: {len(y):,}")
            print(f"  Positive samples: {num_positive:,} ({num_positive/len(y)*100:.1f}%)")
            print(f"  Negative samples: {num_negative:,} ({num_negative/len(y)*100:.1f}%)")
            
            print(f"✓ {split} data verification passed!")
            return True
            
        except Exception as e:
            print(f"✗ {split} data verification failed: {e}")
            return False


def download_pcam_dataset(data_dir: str = "pcam_data", verify: bool = True):
    """
    Convenience function to download complete PCam dataset.
    
    Args:
        data_dir: Directory to store data
        verify: Whether to verify downloaded data
    
    Returns:
        PCamDataLoader instance
    """
    loader = PCamDataLoader(data_dir=data_dir)
    
    # Download all splits
    loader.download_pcam(split='all')
    
    # Verify if requested
    if verify:
        print("\n" + "="*70)
        print("Verifying Downloaded Data")
        print("="*70)
        
        for split in ['train', 'valid', 'test']:
            loader.verify_data(split)
    
    # Print dataset info
    print("\n" + "="*70)
    print("Dataset Information")
    print("="*70)
    
    info = loader.get_dataset_info()
    for split, split_info in info.items():
        if split_info:
            print(f"\n{split.capitalize()}:")
            print(f"  Samples: {split_info['num_samples']:,}")
            print(f"  Shape: {split_info['image_shape']}")
            print(f"  File size: {split_info['file_size_mb']:.1f} MB")
    
    return loader


if __name__ == "__main__":
    # Test data loader
    print("Testing PCam Data Loader...")
    
    # Initialize loader
    loader = PCamDataLoader(data_dir="./test_pcam_data")
    
    # Download train split only (for testing)
    print("\nDownloading train split...")
    files = loader.download_pcam(split='train')
    
    # Load and verify
    print("\nLoading data...")
    x_train, y_train = loader.load_split('train')
    
    print(f"\nData loaded successfully!")
    print(f"Images shape: {x_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Image dtype: {x_train.dtype}")
    print(f"Label values: {np.unique(y_train)}")
    
    # Verify
    loader.verify_data('train')
    
    print("\n Data loader test complete!")