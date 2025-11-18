"""
Main training loop for CAM-FD framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import time

from models.cam_fd_model import CAMFDModel
from losses.combined_loss import CAMFDLoss
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from utils.logger import WandbLogger, MetricsTracker
from utils.checkpoint import CheckpointManager


class CAMFDTrainer:
    """
    Trainer for CAM-FD framework.
    
    Handles:
    - Adversarial training loop
    - Loss computation with all components
    - Metric tracking and logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: CAMFDModel,
        criterion: CAMFDLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        logger: Optional[WandbLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Args:
            model: CAM-FD model
            criterion: CAM-FD loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            config: Configuration dictionary
            logger: WandB logger
            checkpoint_manager: Checkpoint manager
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        
        # Training config
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 100)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        
        # Adversarial training config
        adv_config = config.get('adversarial', {})
        self.adversarial_enabled = adv_config.get('enabled', True)
        self.attack_type = adv_config.get('attack_type', 'pgd')
        
        # Initialize attacker for training
        if self.adversarial_enabled:
            self.attacker = self._create_attacker(adv_config)
        
        # Logging config
        log_config = config.get('logging', {}).get('wandb', {})
        self.log_frequency = log_config.get('log_frequency', 50)
        self.log_images_enabled = log_config.get('log_images', True)
        self.num_log_images = log_config.get('num_log_images', 16)
        
        # Evaluation config
        eval_config = config.get('evaluation', {})
        self.eval_frequency = eval_config.get('eval_frequency', 1)
        
        # Metrics tracker
        self.train_tracker = MetricsTracker()
        self.val_tracker = MetricsTracker()
        
        # Best metric tracking
        self.best_val_acc = 0.0
        self.best_robust_acc = 0.0
        
        # Global step counter
        self.global_step = 0
        self.current_epoch = 0
        
        # Mixed precision training
        hw_config = config.get('hardware', {})
        self.use_mixed_precision = hw_config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
    
    def _create_attacker(self, adv_config: dict):
        """Create adversarial attacker based on config."""
        attack_type = adv_config.get('attack_type', 'pgd')
        
        if attack_type == 'fgsm':
            fgsm_config = adv_config.get('fgsm', {})
            return FGSM(
                model=self.model,
                epsilon=fgsm_config.get('epsilon', 0.031)
            )
        elif attack_type == 'pgd':
            pgd_config = adv_config.get('pgd', {})
            return PGD(
                model=self.model,
                epsilon=pgd_config.get('epsilon', 0.031),
                step_size=pgd_config.get('step_size', 0.007),
                num_steps=pgd_config.get('num_steps', 10),
                random_start=pgd_config.get('random_start', True)
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of averaged training metrics
        """
        self.model.train()
        self.train_tracker.reset()
        
        # Update epoch for loss components (e.g., AWP)
        self.criterion.update_epoch(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generate adversarial examples if enabled
            if self.adversarial_enabled:
                # with torch.no_grad():
                self.model.eval()
                images_adv = self.attacker.generate(images, labels)
                self.model.train()
            else:
                images_adv = images
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Compute CAM-FD loss
                loss, loss_dict = self.criterion(
                    x_clean=images,
                    x_adv=images_adv,
                    targets=labels,
                    apply_mixup=True,
                    return_components=True
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Compute accuracy
            with torch.no_grad():
                logits_clean, _, _ = self.model(images)
                pred_clean = logits_clean.argmax(dim=1)
                acc_clean = (pred_clean == labels).float().mean().item()
                
                if self.adversarial_enabled:
                    logits_adv, _, _ = self.model(images_adv)
                    pred_adv = logits_adv.argmax(dim=1)
                    acc_adv = (pred_adv == labels).float().mean().item()
                else:
                    acc_adv = acc_clean
            
            # Update metrics
            metrics = {
                'train/loss': loss_dict['loss_total'],
                'train/acc_clean': acc_clean,
                'train/acc_adv': acc_adv,
            }
            metrics.update({f"train/{k}": v for k, v in loss_dict.items() if k != 'loss_total'})
            
            self.train_tracker.update(metrics, count=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total']:.4f}",
                'acc': f"{acc_clean:.3f}",
                'rob_acc': f"{acc_adv:.3f}"
            })
            
            # Log to WandB
            if self.logger and (batch_idx % self.log_frequency == 0):
                log_metrics = {
                    **metrics,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch
                }
                self.logger.log(log_metrics, step=self.global_step)
                
                # Log images
                if self.log_images_enabled and batch_idx == 0:
                    self.logger.log_adversarial_examples(
                        clean_images=images[:self.num_log_images],
                        adv_images=images_adv[:self.num_log_images],
                        labels=labels[:self.num_log_images],
                        clean_preds=pred_clean[:self.num_log_images],
                        adv_preds=pred_adv[:self.num_log_images],
                        step=self.global_step
                    )
        
        # Get epoch averages
        epoch_metrics = self.train_tracker.compute()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_tracker.reset()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Clean predictions
            logits_clean, _, _ = self.model(images)
            pred_clean = logits_clean.argmax(dim=1)
            acc_clean = (pred_clean == labels).float().mean().item()
            
            # Compute loss (without mixup)
            loss, loss_dict = self.criterion(
                x_clean=images,
                x_adv=images,  # Same as clean for validation
                targets=labels,
                apply_mixup=False,
                return_components=True
            )
            
            # Update metrics
            metrics = {
                'val/loss': loss_dict['loss_total'],
                'val/acc_clean': acc_clean
            }
            
            self.val_tracker.update(metrics, count=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc_clean:.3f}"
            })
        
        # Get epoch averages
        epoch_metrics = self.val_tracker.compute()
        
        # Log to WandB
        if self.logger:
            log_metrics = {
                **epoch_metrics,
                'val/epoch': epoch
            }
            self.logger.log(log_metrics, step=self.global_step)
        
        return epoch_metrics
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting CAM-FD Training")
        print(f"{'='*60}")
        print(f"Model: {self.config.get('model', {}).get('backbone', 'unknown')}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"Adversarial Training: {self.adversarial_enabled}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} Train: {self.train_tracker.get_summary_string()}")
            
            # Validate
            if epoch % self.eval_frequency == 0:
                val_metrics = self.validate(epoch)
                print(f"Epoch {epoch} Val: {self.val_tracker.get_summary_string()}")
                
                # Save best model
                val_acc = val_metrics.get('val/acc_clean', 0.0)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    if self.checkpoint_manager:
                        self.checkpoint_manager.save(
                            epoch=epoch,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            metrics=val_metrics,
                            is_best=True
                        )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            if self.checkpoint_manager and (epoch % self.checkpoint_manager.save_frequency == 0):
                self.checkpoint_manager.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=train_metrics
                )
            
            print(f"{'='*60}\n")
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best val accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        if self.logger:
            self.logger.finish()


if __name__ == "__main__":
    print("Trainer module - use scripts/train.py to start training")