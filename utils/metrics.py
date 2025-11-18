"""
Metrics computation utilities.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from typing import Dict, Tuple


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy."""
    return (predictions == labels).float().mean().item()


def compute_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    _, topk_pred = logits.topk(k, dim=1, largest=True, sorted=True)
    labels_expanded = labels.view(-1, 1).expand_as(topk_pred)
    correct = (topk_pred == labels_expanded).any(dim=1).float()
    return correct.mean().item()


def compute_precision_recall_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        Dictionary with precision, recall, f1
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(labels, predictions, labels=range(num_classes))


def compute_auc_roc(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray = None
) -> float:
    """
    Compute AUC-ROC score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        probabilities: Predicted probabilities (optional)
    
    Returns:
        AUC-ROC score
    """
    if probabilities is not None:
        return roc_auc_score(labels, probabilities)
    else:
        return roc_auc_score(labels, predictions)


def compute_calibration_error(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probabilities: Predicted probabilities (B, num_classes)
        predictions: Predicted labels (B,)
        labels: True labels (B,)
        num_bins: Number of bins for calibration
    
    Returns:
        Expected Calibration Error
    """
    confidences = probabilities.max(dim=1)[0]
    accuracies = (predictions == labels).float()
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        
        if bin_size > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += (bin_size / len(confidences)) * torch.abs(bin_accuracy - bin_confidence)
    
    return ece.item()


def compute_adversarial_metrics(
    clean_predictions: torch.Tensor,
    adv_predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute adversarial robustness metrics.
    
    Args:
        clean_predictions: Clean predictions
        adv_predictions: Adversarial predictions
        labels: True labels
    
    Returns:
        Dictionary of adversarial metrics
    """
    clean_correct = (clean_predictions == labels)
    adv_correct = (adv_predictions == labels)
    
    # Robust accuracy (correct on adversarial examples)
    robust_acc = adv_correct.float().mean().item()
    
    # Clean accuracy
    clean_acc = clean_correct.float().mean().item()
    
    # Attack success rate (clean correct but adversarial wrong)
    attack_success = (clean_correct & ~adv_correct).float().mean().item()
    
    # Robust accuracy on clean-correct samples
    if clean_correct.sum() > 0:
        robust_acc_on_correct = adv_correct[clean_correct].float().mean().item()
    else:
        robust_acc_on_correct = 0.0
    
    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'attack_success_rate': attack_success,
        'robust_accuracy_on_correct': robust_acc_on_correct
    }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# if __name__ == "__main__":
#     # Test metrics
#     print("Testing metrics...")
    
#     predictions = torch.tensor([0, 1, 1, 0, 1])
#     labels = torch.tensor([0, 1, 0, 0, 1])
    
#     acc = compute_accuracy(predictions, labels)
#     print(f"Accuracy: {acc:.4f}")
    
#     # Test precision/recall/F1
#     metrics = compute_precision_recall_f1(
#         predictions.numpy(),
#         labels.numpy(),
#         average='binary'
#     )
#     print(f"Precision: {metrics['precision']:.4f}")
#     print(f"Recall: {metrics['recall']:.4f}")
#     print(f"F1: {metrics['f1']:.4f}")
    
#     print("\nMetrics test complete!")