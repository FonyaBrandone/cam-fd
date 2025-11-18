"""
Evaluation module for testing adversarial robustness.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import numpy as np

from ..attacks.fgsm import FGSM
from ..attacks.pgd import PGD
from ..attacks.autoattack import AutoAttackWrapper
from ..utils.logger import MetricsTracker


class RobustnessEvaluator:
    """
    Evaluator for adversarial robustness.
    
    Tests model against multiple attacks:
    - Clean accuracy
    - FGSM
    - PGD with various steps
    - AutoAttack (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict
    ):
        """
        Args:
            model: Model to evaluate
            device: Device to run on
            config: Configuration dictionary
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Get evaluation config
        eval_config = config.get('evaluation', {})
        self.attacks_config = eval_config.get('attacks', [])
        
        # Initialize attacks
        self.attacks = self._create_attacks()
    
    def _create_attacks(self) -> Dict[str, object]:
        """Create attack objects based on configuration."""
        attacks = {}
        
        for attack_config in self.attacks_config:
            if not attack_config.get('enabled', True):
                continue
            
            attack_name = attack_config.get('name')
            
            if attack_name == 'fgsm':
                attacks['fgsm'] = FGSM(
                    model=self.model,
                    epsilon=attack_config.get('epsilon', 0.031)
                )
            
            elif attack_name == 'pgd':
                attacks['pgd'] = PGD(
                    model=self.model,
                    epsilon=attack_config.get('epsilon', 0.031),
                    step_size=attack_config.get('step_size', 0.007),
                    num_steps=attack_config.get('num_steps', 20),
                    random_start=True
                )
            
            elif attack_name == 'autoattack':
                attacks['autoattack'] = AutoAttackWrapper(
                    model=self.model,
                    epsilon=attack_config.get('epsilon', 0.031),
                    version=attack_config.get('version', 'standard'),
                    device=str(self.device)
                )
        
        return attacks
    
    @torch.no_grad()
    def evaluate_clean(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate clean accuracy.
        
        Args:
            dataloader: Data loader
        
        Returns:
            Dictionary with clean accuracy metrics
        """
        self.model.eval()
        tracker = MetricsTracker()
        
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Clean Evaluation"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'backbone'):
                logits, _, _ = self.model(images)
            else:
                logits = self.model(images)
            
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            
            tracker.update({'accuracy': acc}, count=images.size(0))
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        metrics = tracker.compute()
        
        # Compute per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        for class_id in np.unique(all_labels):
            mask = all_labels == class_id
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            metrics[f'accuracy_class_{class_id}'] = class_acc
        
        return metrics
    
    def evaluate_attack(
        self,
        attack_name: str,
        dataloader: DataLoader,
        max_batches: int = None
    ) -> Dict[str, float]:
        """
        Evaluate robustness against specific attack.
        
        Args:
            attack_name: Name of attack
            dataloader: Data loader
            max_batches: Maximum number of batches (for AutoAttack)
        
        Returns:
            Dictionary with robustness metrics
        """
        if attack_name not in self.attacks:
            raise ValueError(f"Attack {attack_name} not configured")
        
        self.model.eval()
        attacker = self.attacks[attack_name]
        tracker = MetricsTracker()
        
        all_clean_preds = []
        all_adv_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Evaluating {attack_name.upper()}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Get clean predictions
            with torch.no_grad():
                if hasattr(self.model, 'backbone'):
                    logits_clean, _, _ = self.model(images)
                else:
                    logits_clean = self.model(images)
                pred_clean = logits_clean.argmax(dim=1)
            
            # Generate adversarial examples
            if attack_name == 'autoattack':
                # AutoAttack handles batching internally
                images_adv = attacker.run(images, labels, batch_size=images.size(0))
            else:
                images_adv = attacker.generate(images, labels)
            
            # Get adversarial predictions
            with torch.no_grad():
                if hasattr(self.model, 'backbone'):
                    logits_adv, _, _ = self.model(images_adv)
                else:
                    logits_adv = self.model(images_adv)
                pred_adv = logits_adv.argmax(dim=1)
            
            # Compute metrics
            robust_acc = (pred_adv == labels).float().mean().item()
            clean_acc = (pred_clean == labels).float().mean().item()
            
            tracker.update({
                'robust_accuracy': robust_acc,
                'clean_accuracy': clean_acc
            }, count=images.size(0))
            
            all_clean_preds.extend(pred_clean.cpu().numpy())
            all_adv_preds.extend(pred_adv.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'clean_acc': f"{clean_acc:.3f}",
                'robust_acc': f"{robust_acc:.3f}"
            })
        
        metrics = tracker.compute()
        
        # Compute attack success rate
        all_clean_preds = np.array(all_clean_preds)
        all_adv_preds = np.array(all_adv_preds)
        all_labels = np.array(all_labels)
        
        clean_correct = (all_clean_preds == all_labels)
        adv_correct = (all_adv_preds == all_labels)
        attack_success = clean_correct & ~adv_correct
        
        metrics['attack_success_rate'] = attack_success.mean()
        
        return metrics
    
    def evaluate_all(
        self,
        dataloader: DataLoader,
        max_batches_autoattack: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all attacks.
        
        Args:
            dataloader: Data loader
            max_batches_autoattack: Max batches for AutoAttack (expensive)
        
        Returns:
            Dictionary mapping attack names to metrics
        """
        results = {}
        
        # Clean accuracy
        print("\n" + "="*60)
        print("Evaluating Clean Accuracy")
        print("="*60)
        results['clean'] = self.evaluate_clean(dataloader)
        print(f"Clean Accuracy: {results['clean']['accuracy']:.4f}")
        
        # Adversarial robustness
        for attack_name in self.attacks.keys():
            print("\n" + "="*60)
            print(f"Evaluating {attack_name.upper()}")
            print("="*60)
            
            max_batches = max_batches_autoattack if attack_name == 'autoattack' else None
            results[attack_name] = self.evaluate_attack(
                attack_name,
                dataloader,
                max_batches=max_batches
            )
            
            print(f"Robust Accuracy: {results[attack_name]['robust_accuracy']:.4f}")
            print(f"Attack Success Rate: {results[attack_name]['attack_success_rate']:.4f}")
        
        return results


if __name__ == "__main__":
    print("Evaluator module - use scripts/evaluate.py for evaluation")