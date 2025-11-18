"""
Script to download PCam dataset - Google Colab compatible.

Usage:
    python scripts/download_data.py --data-dir /content/pcam_data --verify
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import PCamDataLoader, download_pcam_dataset


def main(args):
    """Download PCam dataset."""
    
    print("="*70)
    print("PCam Dataset Downloader")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Force re-download: {args.force}")
    print(f"Verify: {args.verify}")
    print("="*70)
    
    # Initialize loader
    loader = PCamDataLoader(data_dir=args.data_dir)
    
    # Download dataset
    if args.split == 'all':
        loader = download_pcam_dataset(
            data_dir=args.data_dir,
            verify=args.verify
        )
    else:
        # Download specific split
        files = loader.download_pcam(split=args.split, force=args.force)
        
        print("\n" + "="*70)
        print("Download Complete!")
        print("="*70)
        print(f"\nDownloaded files:")
        for key, path in files.items():
            print(f"  {key}: {path}")
        
        # Verify if requested
        if args.verify:
            print("\n" + "="*70)
            print("Verifying Data...")
            print("="*70)
            loader.verify_data(args.split)
    
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
    
    print("\n✓ Download script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PCam dataset")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/content/pcam_data',
        help='Directory to store downloaded data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to download'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify downloaded data by loading and checking it'
    )
    
    args = parser.parse_args()
    main(args)