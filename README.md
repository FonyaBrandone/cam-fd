# CAM-FD: Improving Adversarial Robustness without Sacrificing Generalization
`Abstract`
Vision models achieve state-of-the-art performance but remain vulnerable to adversarial perturbations. While existing adversarial training methods improve robustness, they often sacrifice accuracy on clean data or fail to generalize across domains. We introduce CAM-FD, a Curriculum Adversarial Mixup with Feature Denoising, a unified training framework that simultaneously optimizes decision boundary smoothness, output-level robustness, representation level stability, and weight-space flatness. By strategically combining mixup regularization, TRADES-style KL smoothing, feature denoising, and Adversarial Weight Perturbation (AWP) into a single joint loss function, CAM-FD addresses distinct failure modes of single-component defenses. We are demonstrating through extensive experiments on ImageNet across multiple network architecture backbones that our approach establishes a new baseline. We aim to achieve superior adversarial robustness while maintaining strong generalization on clean out-of-sample data and robustness under distribution shifts. Unlike prior methods that optimize robustness and generalization separately or sequentially, CAM-FD provides a principled solution to the adversarial robustness-generalization trade-off, making deep learning models more reliable for safety-critical applications in healthcare, security and beyond.


## ✅ Configuration Files

- [x] `configs/default_config.yaml` - Main configuration with all parameters
- [x] `configs/pcam_config.yaml` 
- [x] `configs/model_configs.yaml`

## ✅ Data Module (`data/`)

- [x] `data/__init__.py` - Package initialization
- [x] `data/pcam_dataset.py` - PCam dataset class and dataloaders
- [x] `data/imagenet_dataset.py` - PCam dataset class and dataloaders
- [x] `data/data_utils.py` - Data utilities and transforms

## ✅ Models Module (`models/`)

- [x] `models/__init__.py` - Package initialization
- [x] `models/backbones.py` - ResNet, ConvNeXt, ViT with feature extraction
- [x] `models/feature_denoiser.py` - AutoEncoder and U-Net denoisers
- [x] `models/cam_fd_model.py` - Complete CAM-FD model

## ✅ Losses Module (`losses/`)

- [x] `losses/__init__.py` - Package initialization
- [x] `losses/mixup_loss.py` - Cross-entropy with Mixup (L_CE(mix))
- [x] `losses/trades_loss.py` - TRADES KL divergence (λ_TR·KL)
- [x] `losses/denoising_loss.py` - Feature reconstruction (λ_rec·||F_adv - F_clean||²)
- [x] `losses/awp_loss.py` - Adversarial Weight Perturbation (λ_AWP·L_AWP)
- [x] `losses/combined_loss.py` - Unified CAM-FD loss function

## ✅ Attacks Module (`attacks/`)

- [x] `attacks/__init__.py` - Package initialization
- [x] `attacks/fgsm.py` - Fast Gradient Sign Method
- [x] `attacks/pgd.py` - Projected Gradient Descent (with L2, Adaptive variants)
- [x] `attacks/autoattack.py` - AutoAttack wrapper for evaluation
- [x] `attacks/attack_utils.py` - Attack utility functions

## ✅ Training Module (`training/`)

- [x] `training/__init__.py` - Package initialization
- [x] `training/trainer.py` - Main training loop with all loss components
- [x] `training/evaluator.py` - Robustness evaluation against multiple attacks
- [x] `training/curriculum.py` - Curriculum learning schedulers
- [x] `training/optimizer_utils.py` - Optimizer and LR scheduler utilities

## ✅ Utils Module (`utils/`)

- [x] `utils/__init__.py` - Package initialization
- [x] `utils/logger.py` - WandB logger with image/metric logging
- [x] `utils/checkpoint.py` - Checkpoint manager for saving/loading
- [x] `utils/metrics.py` - Metrics computation (accuracy, precision, recall, etc.)
- [x] `utils/distributed.py` - Multi-GPU distributed training utilities

## ✅ Scripts (`scripts/`)

- [x] `scripts/train.py` - Main training script with argument parsing
- [x] `scripts/evaluate.py` - Evaluation script for trained models
- [x] `scripts/download_data.py` - Data download from cloud storage
- [x] `scripts/verify_installation.py` - Installation verification script

## ✅ Root Files

- [x] `requirements.txt` - Python dependencies
- [x] `setup.py` - Package setup for installation
- [x] `README.md` - Complete documentation

## 📊 Component Verification

### Loss Function Components 

1. **L_CE(mix)** - Cross-Entropy with Mixup
   - File: `losses/mixup_loss.py`
   - Class: `MixupLoss`
   - Features: Beta distribution sampling, label mixing

2. **λ_TR·KL** - TRADES KL Divergence
   - File: `losses/trades_loss.py`
   - Class: `TRADESLoss`
   - Features: Adaptive lambda, MART variant

3. **λ_rec·||F_adv - F_clean||²** - Feature Denoising
   - File: `losses/denoising_loss.py`
   - Class: `DenoisingLoss`
   - Features: Multi-layer, MSE/L1/cosine losses

4. **λ_AWP·L_AWP** - Adversarial Weight Perturbation
   - File: `losses/awp_loss.py`
   - Class: `AWPLoss`
   - Features: Weight perturbation, SAM variant

5. **Combined Loss** - Unified CAM-FD Loss
   - File: `losses/combined_loss.py`
   - Class: `CAMFDLoss`
   - Features: All components integrated, ablation support

### Model Architecture 

1. **Backbones**
   - ResNet (18, 34, 50, 101)
   - ConvNeXt (base, large)
   - Vision Transformer (base, large)
   - Feature extraction hooks

2. **Feature Denoiser**
   - AutoEncoder architecture
   - U-Net architecture
   - Multi-scale denoising

3. **CAM-FD Model**
   - Integrated backbone + denoiser
   - Feature extraction and denoising
   - Compatible with all loss components

### Attack Implementations 

1. **FGSM** - Single-step attack
2. **PGD** - Multi-step iterative attack
   - L-inf norm
   - L2 norm variant
   - Adaptive variant
3. **AutoAttack** - Ensemble attack wrapper

### Training Infrastructure 

1. **Trainer** - Main training loop
   - Adversarial training
   - Mixed precision (FP16/BF16)
   - Gradient accumulation
   - Metric tracking

2. **Evaluator** - Robustness evaluation
   - Multiple attack evaluation
   - Accuracy metrics
   - Attack success rates

3. **Data Pipeline**
   - PCam and ImageNet dataset
   - Data augmentation
   - Efficient dataloaders

4. **Logging**
   - WandB integration
   - Image logging
   - Metric tracking
   - Checkpoint management

## 🔧 Configuration Parameters

All parameters are configurable in `configs/default_config.yaml`:

- [x] `lambda_tr` (TRADES weight)
- [x] `lambda_rec` (Denoising weight)
- [x] `lambda_awp` (AWP weight)
- [x] `mixup_alpha` (Mixup parameter)
- [x] `epsilon` (Adversarial budget)
- [x] `num_steps` (PGD iterations)
- [x] `step_size` (PGD step size)
- [x] Curric


