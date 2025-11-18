"""
Optimizer and scheduler utilities.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineLR(_LRScheduler):
    """
    Cosine learning rate schedule with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
            warmup_start_lr: Learning rate at start of warmup
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        after_scheduler: _LRScheduler = None,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            after_scheduler: Scheduler to use after warmup
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                if self.after_scheduler is not None:
                    self.after_scheduler.base_lrs = self.base_lrs
            
            return self.after_scheduler.get_lr() if self.after_scheduler else self.base_lrs
    
    def step(self, epoch=None):
        if self.finished_warmup and self.after_scheduler:
            self.after_scheduler.step(epoch)
        else:
            super().step(epoch)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
    no_decay_bias: bool = True,
    no_decay_bn: bool = True
) -> list:
    """
    Get parameter groups with different weight decay.
    
    Args:
        model: Model
        weight_decay: Weight decay value
        no_decay_bias: Don't apply weight decay to bias
        no_decay_bn: Don't apply weight decay to batch norm parameters
    
    Returns:
        List of parameter groups
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have no weight decay
        if no_decay_bias and 'bias' in name:
            no_decay_params.append(param)
        elif no_decay_bn and ('bn' in name or 'norm' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def adjust_learning_rate(
    optimizer: Optimizer,
    epoch: int,
    base_lr: float,
    schedule: str = 'cosine',
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 0.0
):
    """
    Adjust learning rate manually.
    
    Args:
        optimizer: Optimizer
        epoch: Current epoch
        base_lr: Base learning rate
        schedule: Schedule type ('cosine', 'step', 'linear')
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        if schedule == 'cosine':
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule == 'step':
            # Step decay
            lr = base_lr * (0.1 ** (epoch // 30))
        elif schedule == 'linear':
            # Linear decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = base_lr * (1 - progress)
        else:
            lr = base_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


if __name__ == "__main__":
    # Test warmup cosine scheduler
    print("Testing WarmupCosineLR...")
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=5,
        total_epochs=100,
        min_lr=1e-5
    )
    
    print("Learning rates for first 10 epochs:")
    for epoch in range(10):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: LR = {lr:.6f}")
        scheduler.step()
    
    print("\n✓ Optimizer utils test complete!")