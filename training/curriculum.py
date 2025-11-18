"""
Curriculum learning strategies for adversarial training.
"""

import numpy as np


class CurriculumScheduler:
    """
    Base class for curriculum learning schedulers.
    Gradually increases difficulty of adversarial examples during training.
    """
    
    def __init__(self, start_value: float, end_value: float, total_epochs: int):
        """
        Args:
            start_value: Starting value
            end_value: Final value
            total_epochs: Total number of epochs
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_value = start_value
    
    def step(self, epoch: int = None):
        """
        Update curriculum for current epoch.
        
        Args:
            epoch: Current epoch (if None, increment internal counter)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        self.current_value = self._compute_value()
    
    def _compute_value(self) -> float:
        """Compute value for current epoch (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_value(self) -> float:
        """Get current curriculum value."""
        return self.current_value


class LinearCurriculum(CurriculumScheduler):
    """Linear curriculum: linearly increase from start to end value."""
    
    def __init__(self, start_value: float, end_value: float, total_epochs: int, warmup_epochs: int = 0):
        """
        Args:
            start_value: Starting value
            end_value: Final value
            total_epochs: Total number of epochs
            warmup_epochs: Number of epochs to remain at start value
        """
        super().__init__(start_value, end_value, total_epochs)
        self.warmup_epochs = warmup_epochs
    
    def _compute_value(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return self.start_value
        
        progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = np.clip(progress, 0.0, 1.0)
        
        return self.start_value + (self.end_value - self.start_value) * progress


class CosineCurriculum(CurriculumScheduler):
    """Cosine curriculum: smoothly increase using cosine schedule."""
    
    def __init__(self, start_value: float, end_value: float, total_epochs: int, warmup_epochs: int = 0):
        super().__init__(start_value, end_value, total_epochs)
        self.warmup_epochs = warmup_epochs
    
    def _compute_value(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return self.start_value
        
        progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = np.clip(progress, 0.0, 1.0)
        
        # Cosine schedule
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
        
        return self.start_value + (self.end_value - self.start_value) * cosine_progress


class StepCurriculum(CurriculumScheduler):
    """Step curriculum: increase in discrete steps."""
    
    def __init__(self, start_value: float, end_value: float, total_epochs: int, num_steps: int = 5):
        """
        Args:
            start_value: Starting value
            end_value: Final value
            total_epochs: Total number of epochs
            num_steps: Number of discrete steps
        """
        super().__init__(start_value, end_value, total_epochs)
        self.num_steps = num_steps
        self.step_size = (end_value - start_value) / num_steps
        self.epochs_per_step = total_epochs // num_steps
    
    def _compute_value(self) -> float:
        step = min(self.current_epoch // self.epochs_per_step, self.num_steps)
        return self.start_value + step * self.step_size


class ExponentialCurriculum(CurriculumScheduler):
    """Exponential curriculum: exponentially increase difficulty."""
    
    def __init__(self, start_value: float, end_value: float, total_epochs: int, gamma: float = 2.0):
        """
        Args:
            start_value: Starting value
            end_value: Final value
            total_epochs: Total number of epochs
            gamma: Exponential growth rate
        """
        super().__init__(start_value, end_value, total_epochs)
        self.gamma = gamma
    
    def _compute_value(self) -> float:
        progress = self.current_epoch / self.total_epochs
        progress = np.clip(progress, 0.0, 1.0)
        
        # Exponential schedule
        exp_progress = (np.power(self.gamma, progress) - 1) / (self.gamma - 1)
        
        return self.start_value + (self.end_value - self.start_value) * exp_progress


class AdversarialCurriculumManager:
    """
    Manages curriculum learning for multiple adversarial training parameters.
    """
    
    def __init__(self, config: dict, total_epochs: int):
        """
        Args:
            config: Curriculum configuration
            total_epochs: Total number of training epochs
        """
        self.schedulers = {}
        
        if not config.get('enabled', False):
            return
        
        # Epsilon (perturbation magnitude) curriculum
        if 'start_epsilon' in config and 'end_epsilon' in config:
            self.schedulers['epsilon'] = LinearCurriculum(
                start_value=config['start_epsilon'],
                end_value=config['end_epsilon'],
                total_epochs=total_epochs,
                warmup_epochs=config.get('warmup_epochs', 0)
            )
        
        # Number of attack steps curriculum
        if 'start_num_steps' in config and 'end_num_steps' in config:
            self.schedulers['num_steps'] = StepCurriculum(
                start_value=config['start_num_steps'],
                end_value=config['end_num_steps'],
                total_epochs=total_epochs,
                num_steps=5
            )
        
        # Attack strength curriculum
        if 'start_step_size' in config and 'end_step_size' in config:
            self.schedulers['step_size'] = LinearCurriculum(
                start_value=config['start_step_size'],
                end_value=config['end_step_size'],
                total_epochs=total_epochs
            )
    
    def step(self, epoch: int):
        """Update all curriculum schedulers."""
        for scheduler in self.schedulers.values():
            scheduler.step(epoch)
    
    def get_values(self) -> dict:
        """Get current values for all curriculum parameters."""
        return {
            name: scheduler.get_value()
            for name, scheduler in self.schedulers.items()
        }
    
    def update_attacker(self, attacker):
        """Update attacker parameters based on curriculum."""
        values = self.get_values()
        
        if 'epsilon' in values:
            attacker.epsilon = values['epsilon']
        if 'num_steps' in values and hasattr(attacker, 'num_steps'):
            attacker.num_steps = int(values['num_steps'])
        if 'step_size' in values and hasattr(attacker, 'step_size'):
            attacker.step_size = values['step_size']


if __name__ == "__main__":
    # Test curriculum schedulers
    print("Testing Curriculum Schedulers...")
    
    total_epochs = 100
    
    # Test linear curriculum
    print("\nLinear Curriculum:")
    linear = LinearCurriculum(start_value=0.01, end_value=0.031, total_epochs=total_epochs, warmup_epochs=10)
    for epoch in [0, 10, 25, 50, 75, 100]:
        linear.step(epoch)
        print(f"Epoch {epoch}: {linear.get_value():.4f}")
    
    # Test cosine curriculum
    print("\nCosine Curriculum:")
    cosine = CosineCurriculum(start_value=0.01, end_value=0.031, total_epochs=total_epochs, warmup_epochs=10)
    for epoch in [0, 10, 25, 50, 75, 100]:
        cosine.step(epoch)
        print(f"Epoch {epoch}: {cosine.get_value():.4f}")
    
    # Test curriculum manager
    print("\nCurriculum Manager:")
    config = {
        'enabled': True,
        'start_epsilon': 0.01,
        'end_epsilon': 0.031,
        'warmup_epochs': 10
    }
    manager = AdversarialCurriculumManager(config, total_epochs)
    
    for epoch in [0, 10, 25, 50, 100]:
        manager.step(epoch)
        values = manager.get_values()
        print(f"Epoch {epoch}: epsilon={values.get('epsilon', 0):.4f}")
    
    print("\n✓ Curriculum test complete!")