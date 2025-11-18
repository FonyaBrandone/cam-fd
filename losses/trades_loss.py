"""
TRADES Loss: KL divergence between clean and adversarial predictions.
Enforces consistency for adversarial robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TRADESLoss(nn.Module):
    """
    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization) Loss.
    
    Balances natural accuracy and robustness by:
    1. Minimizing cross-entropy on clean examples
    2. Minimizing KL divergence between clean and adversarial predictions
    
    Reference: Zhang et al., "Theoretically Principled Trade-off between 
    Robustness and Accuracy", ICML 2019
    """
    
    def __init__(self, lambda_trades: float = 6.0):
        """
        Args:
            lambda_trades: Weight for KL divergence term. 
                          Higher = more emphasis on robustness
        """
        super().__init__()
        self.lambda_trades = lambda_trades
    
    def kl_divergence(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        reduction: str = 'batchmean'
    ) -> torch.Tensor:
        """
        Compute KL divergence between clean and adversarial predictions.
        
        KL(P_clean || P_adv) encourages adversarial predictions to match clean predictions.
        
        Args:
            logits_clean: Clean logits (B, num_classes)
            logits_adv: Adversarial logits (B, num_classes)
            reduction: Reduction method ('batchmean', 'mean', 'sum')
        
        Returns:
            KL divergence loss
        """
        # Convert logits to log probabilities
        log_prob_clean = F.log_softmax(logits_clean, dim=1)
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        
        # KL(clean || adv) = sum(P_clean * log(P_clean / P_adv))
        kl_div = F.kl_div(
            log_prob_adv,
            log_prob_clean,
            reduction=reduction,
            log_target=True
        )
        
        return kl_div
    
    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        targets: torch.Tensor = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute TRADES loss.
        
        Args:
            logits_clean: Clean predictions (B, num_classes)
            logits_adv: Adversarial predictions (B, num_classes)
            targets: Ground truth labels (B,) - optional, for natural loss
            return_components: Whether to return loss components separately
        
        Returns:
            loss: Combined TRADES loss (or tuple of components if return_components=True)
        """
        # KL divergence term (robustness)
        kl_loss = self.kl_divergence(logits_clean, logits_adv)
        
        if return_components:
            # Natural accuracy term
            natural_loss = F.cross_entropy(logits_clean, targets) if targets is not None else 0.0
            return natural_loss, self.lambda_trades * kl_loss
        
        # Combined loss
        if targets is not None:
            # Full TRADES: natural loss + lambda * KL
            natural_loss = F.cross_entropy(logits_clean, targets)
            total_loss = natural_loss + self.lambda_trades * kl_loss
        else:
            # Just KL term (when natural loss computed separately)
            total_loss = self.lambda_trades * kl_loss
        
        return total_loss


class AdaptiveTRADESLoss(nn.Module):
    """
    TRADES loss with adaptive lambda based on training progress.
    Gradually increases robustness emphasis during training.
    """
    
    def __init__(
        self,
        lambda_start: float = 1.0,
        lambda_end: float = 6.0,
        warmup_epochs: int = 10,
        total_epochs: int = 100
    ):
        super().__init__()
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_lambda = lambda_start
        
    def update_lambda(self, epoch: int):
        """Update lambda based on current epoch."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
            self.current_lambda = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        else:
            self.current_lambda = self.lambda_end
    
    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        targets: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute TRADES loss with current lambda."""
        kl_loss = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.log_softmax(logits_clean, dim=1),
            reduction='batchmean',
            log_target=True
        )
        
        if targets is not None:
            natural_loss = F.cross_entropy(logits_clean, targets)
            return natural_loss + self.current_lambda * kl_loss
        
        return self.current_lambda * kl_loss


class MARTLoss(nn.Module):
    """
    MART (Misclassification Aware adveRsarial Training) Loss.
    
    Alternative to TRADES that focuses on misclassified examples.
    Can be used as a drop-in replacement.
    
    Reference: Wang et al., "Improving Adversarial Robustness Requires 
    Revisiting Misclassified Examples", ICLR 2020
    """
    
    def __init__(self, lambda_mart: float = 6.0):
        super().__init__()
        self.lambda_mart = lambda_mart
    
    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_adv: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MART loss with boosted loss on misclassified examples.
        
        Args:
            logits_clean: Clean predictions (B, num_classes)
            logits_adv: Adversarial predictions (B, num_classes)
            targets: Ground truth labels (B,)
        
        Returns:
            MART loss
        """
        # Natural loss
        natural_loss = F.cross_entropy(logits_clean, targets)
        
        # Check which examples are correctly classified
        pred_clean = logits_clean.argmax(dim=1)
        correct_mask = (pred_clean == targets).float()
        
        # KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.log_softmax(logits_clean, dim=1),
            reduction='none',
            log_target=True
        ).sum(dim=1)
        
        # Boost loss on misclassified examples
        # correct examples: KL loss
        # incorrect examples: KL loss + CE loss
        adv_loss = correct_mask * kl_loss + (1 - correct_mask) * (
            kl_loss + F.cross_entropy(logits_adv, targets, reduction='none')
        )
        
        return natural_loss + self.lambda_mart * adv_loss.mean()


if __name__ == "__main__":
    # Test TRADES loss
    print("Testing TRADES Loss...")
    
    trades_loss = TRADESLoss(lambda_trades=6.0)
    
    # Create dummy data
    batch_size = 8
    num_classes = 2
    logits_clean = torch.randn(batch_size, num_classes)
    logits_adv = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss
    loss = trades_loss(logits_clean, logits_adv, targets)
    print(f"TRADES loss: {loss.item():.4f}")
    
    # Get components
    nat_loss, rob_loss = trades_loss(logits_clean, logits_adv, targets, return_components=True)
    print(f"Natural loss: {nat_loss.item():.4f}")
    print(f"Robustness loss: {rob_loss.item():.4f}")
    
    # Test adaptive TRADES
    print("\nTesting Adaptive TRADES...")
    adaptive_trades = AdaptiveTRADESLoss(lambda_start=1.0, lambda_end=6.0, warmup_epochs=10)
    
    for epoch in [0, 5, 10, 20]:
        adaptive_trades.update_lambda(epoch)
        loss = adaptive_trades(logits_clean, logits_adv, targets)
        print(f"Epoch {epoch}: lambda={adaptive_trades.current_lambda:.2f}, loss={loss.item():.4f}")
    
    # Test MART loss
    print("\nTesting MART Loss...")
    mart_loss = MARTLoss(lambda_mart=6.0)
    loss = mart_loss(logits_clean, logits_adv, targets)
    print(f"MART loss: {loss.item():.4f}")