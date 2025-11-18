"""
Main training script for CAM-FD framework.

Usage:
    python scripts/train.py --config configs/default_config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.cam_fd_model import create_cam_fd_model
from losses.combined_loss import CAMFDLossFactory
from data.pcam_dataset import create_pcam_dataloaders
from training.trainer import CAMFDTrainer
from utils.logger import WandbLogger
from utils.checkpoint import CheckpointManager


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    opt_config = config.get('training', {}).get('optimizer', {})
    opt_type = opt_config.get('type', 'sgd').lower()

    # Ensure numeric values are floats
    lr = float(opt_config.get('lr', 0.1))
    momentum = float(opt_config.get('momentum', 0.9))
    weight_decay = float(opt_config.get('weight_decay', 5e-4))

    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=opt_config.get('nesterov', True)
        )
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    return optimizer



def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler based on configuration."""
    sched_config = config.get('training', {}).get('scheduler', {})
    sched_type = sched_config.get('type', 'cosine').lower()
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    
    if sched_type == 'cosine':
        eta_min = float(sched_config.get('min_lr', 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min
        )
    elif sched_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config.get('milestones', [50, 75]),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=sched_config.get('gamma', 0.1),
            patience=sched_config.get('patience', 10)
        )
    else:
        scheduler = None
    
    return scheduler


def setup_device(config: dict) -> torch.device:
    """Setup device for training."""
    hw_config = config.get('hardware', {})
    device_type = hw_config.get('device', 'cuda')
    
    if device_type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device_type)
    
    if device.type == 'cuda':
        # Enable cudnn benchmark for better performance
        if hw_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def main(args):
    """Main training function."""
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    
    # Set random seed
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Setup device
    device = setup_device(config)
    print(f"Device: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    cloud_config = config.get('data', {}).get('cloud_storage', {})
    data_config = config.get('data', {})
    
    train_loader, val_loader, test_loader = create_pcam_dataloaders(
        data_config=data_config,
        download=not args.no_download
    )
    
    # Create model
    print("\nCreating CAM-FD model...")
    model = create_cam_fd_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    print("\nCreating CAM-FD loss...")
    criterion = CAMFDLossFactory.from_config(model, config)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    print("Creating scheduler...")
    scheduler = create_scheduler(optimizer, config)
    
    # Create logger
    if not args.no_wandb:
        print("\nInitializing WandB logger...")
        wandb_config = config.get('logging', {}).get('wandb', {})
        logger = WandbLogger(
            config=config,
            enabled=wandb_config.get('enabled', True),
            project=wandb_config.get('project', 'cam-fd-medical-robustness'),
            entity=wandb_config.get('entity', None),
            name=config.get('experiment', {}).get('name', None),
            tags=config.get('experiment', {}).get('tags', [])
        )
        
        # Watch model
        logger.log_model_graph(model)
    else:
        logger = None
        print("WandB logging disabled")
    
    # Create checkpoint manager
    print("Creating checkpoint manager...")
    ckpt_config = config.get('logging', {}).get('checkpoint', {})
    checkpoint_manager = CheckpointManager(
        save_dir=ckpt_config.get('save_dir', './checkpoints'),
        save_frequency=config.get('evaluation', {}).get('save_frequency', 5),
        save_best_only=ckpt_config.get('save_best_only', False),
        monitor=ckpt_config.get('monitor', 'val/acc_clean'),
        mode=ckpt_config.get('mode', 'max'),
        max_checkpoints=5
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        info = checkpoint_manager.load(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = info['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = CAMFDTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        logger=logger,
        checkpoint_manager=checkpoint_manager
    )
    
    # Start training
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if logger:
            logger.finish()
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        if logger:
            logger.finish()
        raise
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAM-FD model")
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    
    # Experiment settings
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )
    
    # Training settings
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # Data settings
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Do not download data from cloud (use cached data)'
    )
    
    # Logging
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    
    # Checkpoint
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    main(args)