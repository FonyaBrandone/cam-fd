"""
Logging utilities with WandB integration for experiment tracking.
"""

import os
import wandb
import torch
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt


class WandbLogger:
    """
    Weights & Biases logger for experiment tracking.
    """
    
    def __init__(
        self,
        config: dict,
        enabled: bool = True,
        project: str = "cam-fd-medical-robustness",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        dir: str = "./wandb_logs",
        resume: bool = False,
        id: Optional[str] = None
    ):
        """
        Args:
            config: Configuration dictionary to log
            enabled: Whether to enable logging
            project: WandB project name
            entity: WandB entity/team name
            name: Run name
            tags: List of tags for the run
            notes: Notes about the run
            dir: Directory to store WandB logs
            resume: Whether to resume from existing run
            id: Run ID (for resuming)
        """
        self.enabled = enabled
        self.config = config
        
        if not enabled:
            print("WandB logging disabled")
            return
        
        # Initialize WandB
        os.makedirs(dir, exist_ok=True)
        
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
            dir=dir,
            resume="allow" if resume else None,
            id=id
        )
        
        print(f"✓ WandB initialized: {self.run.name}")
        print(f"  Project: {project}")
        print(f"  URL: {self.run.url}")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return
        
        wandb.log(metrics, step=step, commit=commit)
    
    def log_images(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        key: str = "images",
        step: Optional[int] = None,
        max_images: int = 16,
        denormalize: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ):
        """
        Log images to WandB.
        
        Args:
            images: Image tensor (B, C, H, W)
            labels: Label tensor (B,)
            predictions: Optional prediction tensor (B,)
            key: Key for logging
            step: Step number
            max_images: Maximum number of images to log
            denormalize: Whether to denormalize images
            mean: Mean for denormalization
            std: Std for denormalization
        """
        if not self.enabled:
            return
        
        # Limit number of images
        num_images = min(len(images), max_images)
        images = images[:num_images]
        labels = labels[:num_images]
        if predictions is not None:
            predictions = predictions[:num_images]
        
        # Denormalize if needed
        if denormalize:
            mean = torch.tensor(mean).view(3, 1, 1).to(images.device)
            std = torch.tensor(std).view(3, 1, 1).to(images.device)
            images = images * std + mean
        
        # Clip to valid range
        images = torch.clamp(images, 0, 1)
        
        # Convert to numpy
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        if predictions is not None:
            predictions = predictions.cpu().numpy()
        
        # Create WandB images
        wandb_images = []
        for i in range(num_images):
            img = np.transpose(images[i], (1, 2, 0))  # (H, W, C)
            
            # Create caption
            if predictions is not None:
                caption = f"Label: {labels[i]}, Pred: {predictions[i]}"
            else:
                caption = f"Label: {labels[i]}"
            
            wandb_images.append(wandb.Image(img, caption=caption))
        
        self.log({key: wandb_images}, step=step)
    
    def log_adversarial_examples(
        self,
        clean_images: torch.Tensor,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        clean_preds: torch.Tensor,
        adv_preds: torch.Tensor,
        step: Optional[int] = None,
        max_images: int = 8
    ):
        """
        Log clean vs adversarial examples side-by-side.
        
        Args:
            clean_images: Clean images (B, C, H, W)
            adv_images: Adversarial images (B, C, H, W)
            labels: True labels (B,)
            clean_preds: Clean predictions (B,)
            adv_preds: Adversarial predictions (B,)
            step: Step number
            max_images: Maximum number of examples to log
        """
        if not self.enabled:
            return
        
        num_images = min(len(clean_images), max_images)
        
        # Create comparison figure
        fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            # Clean image
            clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
            clean_img = np.clip(clean_img, 0, 1)
            axes[i, 0].imshow(clean_img)
            axes[i, 0].set_title(f"Clean\nTrue: {labels[i].item()}, Pred: {clean_preds[i].item()}")
            axes[i, 0].axis('off')
            
            # Adversarial image
            adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
            adv_img = np.clip(adv_img, 0, 1)
            axes[i, 1].imshow(adv_img)
            axes[i, 1].set_title(f"Adversarial\nTrue: {labels[i].item()}, Pred: {adv_preds[i].item()}")
            axes[i, 1].axis('off')
            
            # Perturbation (amplified for visibility)
            pert = (adv_images[i] - clean_images[i]).abs().cpu()
            pert = pert.mean(0)  # Average over channels
            axes[i, 2].imshow(pert, cmap='hot')
            axes[i, 2].set_title("Perturbation (amplified)")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Log to WandB
        self.log({"adversarial_examples": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list = None,
        title: str = "Confusion Matrix",
        step: Optional[int] = None
    ):
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the plot
            step: Step number
        """
        if not self.enabled:
            return
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm)),
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        self.log({title.lower().replace(' ', '_'): wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_histogram(
        self,
        data: np.ndarray,
        key: str,
        step: Optional[int] = None,
        bins: int = 50
    ):
        """Log histogram of data."""
        if not self.enabled:
            return
        
        self.log({key: wandb.Histogram(data, num_bins=bins)}, step=step)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple = (1, 3, 96, 96)):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_shape: Input shape for the model
        """
        if not self.enabled:
            return
        
        try:
            dummy_input = torch.randn(input_shape)
            wandb.watch(model, log='all', log_freq=100)
            print("✓ Model graph logged to WandB")
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_gradients(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ):
        """
        Log gradient statistics.
        
        Args:
            model: PyTorch model
            step: Step number
        """
        if not self.enabled:
            return
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[f"gradients/{name}"] = grad_norm
        
        self.log(grad_norms, step=step)
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Optional metadata dictionary
        """
        if not self.enabled:
            return
        
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metadata or {}
        )
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
        print(f"✓ Checkpoint saved to WandB: {checkpoint_path}")
    
    def finish(self):
        """Finish WandB run."""
        if not self.enabled:
            return
        
        wandb.finish()
        print("✓ WandB run finished")


class MetricsTracker:
    """
    Simple metrics tracker for averaging metrics over batches/epochs.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            count: Number of samples (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        averaged = {}
        for key in self.metrics.keys():
            if self.counts[key] > 0:
                averaged[key] = self.metrics[key] / self.counts[key]
            else:
                averaged[key] = 0.0
        return averaged
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def get_summary_string(self) -> str:
        """Get formatted summary string."""
        averaged = self.compute()
        summary = ", ".join([f"{k}: {v:.4f}" for k, v in averaged.items()])
        return summary


if __name__ == "__main__":
    # Test WandB logger
    print("Testing WandB Logger...")
    
    # Create dummy config
    config = {
        'model': {'backbone': 'resnet50', 'num_classes': 2},
        'training': {'num_epochs': 100, 'lr': 0.1},
        'loss': {'lambda_trades': 6.0, 'lambda_rec': 0.5}
    }
    
    # Initialize logger (with enabled=False for testing)
    logger = WandbLogger(
        config=config,
        enabled=False,  # Set to True to actually log to WandB
        project="cam-fd-test",
        name="test-run",
        tags=["test", "pcam"]
    )
    
    # Test logging metrics
    print("\nTesting metric logging...")
    logger.log({
        'train/loss': 0.5,
        'train/acc': 0.85,
        'train/robust_acc': 0.70
    }, step=1)
    
    # Test logging images
    print("Testing image logging...")
    dummy_images = torch.randn(4, 3, 96, 96)
    dummy_labels = torch.tensor([0, 1, 0, 1])
    dummy_preds = torch.tensor([0, 1, 1, 1])
    
    logger.log_images(
        images=dummy_images,
        labels=dummy_labels,
        predictions=dummy_preds,
        key="train/samples",
        step=1,
        max_images=4
    )
    
    # Test metrics tracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker()
    
    # Simulate batch updates
    for i in range(5):
        tracker.update({
            'loss': np.random.rand(),
            'accuracy': np.random.rand()
        }, count=32)
    
    # Compute averages
    averaged = tracker.compute()
    print(f"Averaged metrics: {tracker.get_summary_string()}")
    
    # Reset
    tracker.reset()
    print("Metrics reset")
    
    print("\n✓ Logger test complete!")