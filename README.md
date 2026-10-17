# Modern InceptionV1 across Varying Scales (up to 300x)

This repository modernizes **InceptionV1** (originally GoogLeNet, Szegedy et al., 2014) by replacing its 2014-era training and architectural bottlenecks with modern deep learning practices and provides a continuous scaling suite spanning a **300x parameter range** (from ~92k to ~28M parameters).

It is specifically structured to empirically test and peer-review hypotheses surrounding circuit cleanliness, polysemanticity, and superposition across model capacities—directly addressing themes in modern mechanistic interpretability (such as Nick Cammarata's *"Neural Networks Are Already Clean"*).

---

## 1. Architectural Modernizations

The original 2014 InceptionV1 architecture suffered from severe gradient dispersion, vanishing gradients (which necessitated auxiliary loss heads), and un-normalized activations. This implementation makes the following key upgrades:

| Component | Classic InceptionV1 (2014) | Modern InceptionV1 (This Repo) | Rationale |
| :--- | :--- | :--- | :--- |
| **Normalization** | None (or Local Response Norm) | **BatchNorm2d** after all convolutions | Eliminates internal covariate shift; stabilizes deep gradient flow. |
| **Activation** | ReLU | **GELU (Gaussian Error Linear Unit)** | Smooth non-linearity preventing dead neurons and improving gradient flow. |
| **$5\times5$ Convolutions** | Single $5\times5$ filter | **Factorized two $3\times3$ convolutions** | Retains identical $5\times5$ receptive field with fewer parameters and added non-linearity. |
| **Residual Shortcuts** | None | **Inception-ResNet Skip Connections** | Provides gradient highways directly from output to stem. |
| **Auxiliary Classifiers** | Two auxiliary loss heads | **Removed** | Residual shortcuts and modern normalization render auxiliary heads obsolete. |
| **Regularization** | Standard dropout (0.4) | **Stochastic Depth (DropPath) + Dropout** | Linearly scaled drop-path regularizes deeper blocks. |

---

## 2. The 300x Scaling Suite

Model capacity is controlled via a continuous width multiplier $w$. Six calibrated presets span a **300x parameter ratio** (matching the parameter range from 92k to 28M tested in modern convnet interpretability audits):

| Preset | Width Mult ($w$) | Backbone Params | Total Params (1000 cls) | FLOPs / MACs | Scale Factor | Role / Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`pico`** | `0.088` | **98,120** | 195,120 | 37.9 M | **1.0x** | Sub-100k parameter baseline |
| **`nano`** | `0.155` | **242,064** | 403,064 | 92.8 M | **2.5x** | Quarter-million parameter regime |
| **`micro`** | `0.290` | **764,232** | 1,069,232 | 233.5 M | **7.8x** | ~1M total parameter scale |
| **`mini`** | `0.520` | **2,167,232** | 2,696,232 | 542.7 M | **22.1x** | Intermediate compact model |
| **`base`** | `0.950` | **7,709,568** | 8,686,568 | 1.89 G | **78.6x** | Standard Inception capacity |
| **`large`** | `1.660` | **24,237,512** | 25,950,512 | 5.68 G | **247.0x** - **300x** | Full capacity modern convnet |

*Note: You can also specify any arbitrary continuous width multiplier (e.g. `--width-mult 1.8` for exact 300x parameter count).*

---

## 3. Modern Training Recipe

- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 0.05$).
- **Weight Decay Decoupling:** Biases and 1D normalization parameters (BatchNorm weights/biases) are excluded from weight decay.
- **Learning Rate Schedule:** Linear warmup (5 epochs) followed by Cosine Annealing decay down to $10^{-6}$.
- **Data Augmentation:**
  - `RandomResizedCrop(224, scale=(0.08, 1.0))`
  - `RandomHorizontalFlip()`
  - `RandAugment(num_ops=2, magnitude=9)`
  - `ColorJitter`
  - `Mixup` ($\alpha=0.8$) + `CutMix` ($\alpha=1.0$)
- **Loss:** Soft-Target CrossEntropy with Label Smoothing ($0.1$).
- **Acceleration:** PyTorch AMP (`torch.amp.autocast`) with support for CUDA, Apple Silicon MPS, and CPU.

---

## 4. Usage

### A. Quick Scaling Benchmark
To run an automated throughput, memory, and parameter benchmark across all 6 scales:
```bash
.venv/bin/python benchmark.py
```

### B. Training on ImageNet-1k
Ensure your ImageNet directory is arranged with standard subfolders:
```
/path/to/imagenet/
  ├── train/
  └── val/
```
Run training for any scale preset:
```bash
# Train Base scale (~9M params)
.venv/bin/python train.py --scale base --data-dir /path/to/imagenet --batch-size 128

# Train Pico scale (1x baseline ~98k params)
.venv/bin/python train.py --scale pico --data-dir /path/to/imagenet --batch-size 256

# Train Large scale (300x scale ~28M params)
.venv/bin/python train.py --scale large --data-dir /path/to/imagenet --batch-size 64
```

### C. Synthetic Benchmarking (Without downloading ImageNet)
You can verify the entire training pipeline, throughput, and GPU/MPS memory on synthetic data:
```bash
.venv/bin/python train.py --scale pico --synthetic --epochs 5 --batch-size 64
```

### D. Mechanistic Interpretability & Circuit Cleanliness
To extract activations and compute neuron kurtosis & lifetime sparsity:
```python
from models import create_modern_inception_v1
from interpretability import FeatureExtractor, compute_activation_kurtosis, compute_lifetime_sparsity
import torch

model = create_modern_inception_v1("base")
extractor = FeatureExtractor(model, target_layers=["mixed3b", "mixed4c", "mixed5b"])

images = torch.randn(32, 3, 224, 224)
_ = model(images)

activations = extractor.activations["mixed4c"]
kurt = compute_activation_kurtosis(activations)
print("Mean neuron kurtosis:", kurt.mean().item())
```
