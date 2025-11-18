"""
Checkpoint management for saving and loading model state.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional
import shutil


class CheckpointManager:
    """
    Manages model checkpoints with automatic saving and loading.
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_frequency: int = 5,
        save_best_only: bool = False,
        monitor: str = "val/acc_clean",
        mode: str = "max",
        max_checkpoints: int = 5
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
            save_best_only: Only save when metric improves
            monitor: Metric to monitor for best checkpoint
            mode: 'max' or 'min' for metric
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_checkpoints = max_checkpoints
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metric
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.checkpoints = []
    
    def _is_better(self, metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'max':
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
            extra_state: Additional state to save
        
        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        monitor_value = metrics.get(self.monitor, None)
        
        # Check if we should save
        if self.save_best_only and not is_best:
            if monitor_value is None or not self._is_better(monitor_value):
                return None
        
        # Update best metric
        if monitor_value is not None and self._is_better(monitor_value):
            self.best_metric = monitor_value
            is_best = True
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            shutil.copy(checkpoint_path, best_path)
            print(f"✓ Best checkpoint updated: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pt"
        shutil.copy(checkpoint_path, latest_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch (oldest first)
            self.checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        load_optimizer: bool = True,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            load_optimizer: Whether to load optimizer state
            device: Device to load checkpoint to
        
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        # Load optimizer state
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded")
        
        # Update best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_metric': checkpoint.get('best_metric', self.best_metric),
            'extra_state': checkpoint.get('extra_state', {})
        }
    
    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """Load the best checkpoint."""
        best_path = self.save_dir / "best_checkpoint.pt"
        return self.load(best_path, model, optimizer, scheduler, device=device)
    
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """Load the latest checkpoint."""
        latest_path = self.save_dir / "latest_checkpoint.pt"
        return self.load(latest_path, model, optimizer, scheduler, device=device)


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing Checkpoint Manager...")
    
    # Create dummy model
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Create checkpoint manager
    manager = CheckpointManager(
        save_dir="./test_checkpoints",
        save_frequency=1,
        monitor="val/acc",
        mode="max",
        max_checkpoints=3
    )
    
    # Save some checkpoints
    for epoch in range(1, 6):
        metrics = {'val/acc': 0.5 + epoch * 0.05}
        manager.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=metrics
        )
    
    print(f"\nBest metric: {manager.best_metric:.4f}")
    print(f"Number of checkpoints: {len(manager.checkpoints)}")
    
    # Test loading
    print("\nTesting checkpoint loading...")
    info = manager.load_best(model, optimizer, device='cpu')
    print(f"Loaded epoch: {info['epoch']}")
    print(f"Loaded metrics: {info['metrics']}")
    
    # Cleanup test checkpoints
    import shutil
    shutil.rmtree("./test_checkpoints")
    print("\n✓ Checkpoint manager test complete!")