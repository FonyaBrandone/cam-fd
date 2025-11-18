"""
Evaluation script for trained CAM-FD models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_checkpoint.pt --config configs/default_config.yaml
"""

import argparse
import yaml
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.cam_fd_model import create_cam_fd_model
from data.pcam_dataset import create_pcam_dataloaders
from training.evaluator import RobustnessEvaluator
from utils.checkpoint import CheckpointManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main evaluation function."""
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create data loaders
    print("\nLoading test data...")
    cloud_config = config.get('data', {}).get('cloud_storage', {})
    data_config = config.get('data', {})
    
    _, _, test_loader = create_pcam_dataloaders(
        cloud_config=cloud_config,
        data_config=data_config,
        download=False  # Assume data already downloaded
    )
    
    # Create model
    print("\nCreating model...")
    model = create_cam_fd_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.load(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device
    )
    
    # Create evaluator
    print("\nCreating evaluator...")
    evaluator = RobustnessEvaluator(
        model=model,
        device=device,
        config=config
    )
    
    # Run evaluation
    print("\n" + "="*70)
    print("Starting Evaluation")
    print("="*70)
    
    results = evaluator.evaluate_all(
        dataloader=test_loader,
        max_batches_autoattack=args.max_batches_autoattack
    )
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    for attack_name, metrics in results.items():
        print(f"\n{attack_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAM-FD model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_results.json',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--max-batches-autoattack',
        type=int,
        default=10,
        help='Maximum batches for AutoAttack (expensive)'
    )
    
    args = parser.parse_args()
    main(args)