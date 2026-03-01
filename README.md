<div align="center">

# 🌿 Semantic Segmentation — CODE CRUNCH Hackathon

### *Nature Scene Parsing with DINOv2 + Custom Segmentation Head*

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2-purple?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> 🏆 **Hackathon Submission** — Achieving **0.2849 Mean IoU** on a 10-class nature scene segmentation dataset using a frozen DINOv2 backbone with a lightweight custom decoder head.

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [License](#-license)

---

## 🔭 Overview

This project tackles **multi-class semantic segmentation** of natural outdoor scenes containing classes like trees, bushes, grass, rocks, sky, and more. The pipeline leverages the power of **Meta's DINOv2** — a self-supervised Vision Transformer — as a frozen feature extractor, topped with a **custom lightweight segmentation head** as the decoder.

**Key Idea:** By freezing the DINOv2 backbone, we exploit rich pretrained visual representations while only training the decoder head, making the approach efficient and fast to converge.

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT IMAGE                       │
│               (with augmentations)                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              🧊 DINOv2 BACKBONE (Frozen)            │
│         Self-Supervised Vision Transformer          │
│            All layers FROZEN — no grad              │
└──────────────────────┬──────────────────────────────┘
                       │  Feature Maps
                       ▼
┌─────────────────────────────────────────────────────┐
│         🎯 CUSTOM SEGMENTATION HEAD (Decoder)       │
│            Learnable Decoder Layers                 │
│         Upsamples & classifies each pixel           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         📊 SEGMENTATION MAP (10 Classes)            │
│   Background · Trees · Lush Bushes · Dry Grass     │
│   Dry Bushes · Ground Clutter · Logs · Rocks       │
│            Landscape · Sky                          │
└─────────────────────────────────────────────────────┘
```

| Component | Details |
|---|---|
| **Backbone** | DINOv2 (Vision Transformer) — **Frozen** |
| **Decoder** | Custom Segmentation Head |
| **Classes** | 10 (Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky) |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | `1e-4` |
| **LR Scheduler** | Cosine Annealing (`max_lr=3e-4`) |
| **Epochs** | 10 |
| **Loss Function** | Cross-Entropy Loss |
| **Backbone Layers** | 🧊 Frozen (no gradient updates) |
| **Data Augmentation** | Basic augmentations aligned with train images & masks |

### 📈 Data Augmentation Pipeline

Basic augmentations applied **consistently** to both images and their corresponding masks:
- Random horizontal/vertical flips
- Random rotations
- Color jitter (image only)
- Normalization

---

## 📊 Results

### 🏅 Final Training Metrics

| Metric | Train | Validation |
|---|---|---|
| **Loss** | 0.4248 | 0.4242 |
| **IoU** | 0.5291 | 0.5222 |
| **Dice Score** | 0.6684 | 0.6609 |
| **Accuracy** | 83.47% | 83.55% |

### 🏆 Best Checkpoints

| Metric | Value | Epoch |
|---|---|---|
| **Best Val IoU** | 0.5226 | 9 |
| **Best Val Dice** | 0.6618 | 9 |
| **Best Val Accuracy** | 83.55% | 10 |
| **Lowest Val Loss** | 0.4242 | 10 |

### 📉 Training Progress (Per Epoch)

```
Epoch   Train Loss   Val Loss   Train IoU   Val IoU   Train Dice   Val Dice   Train Acc   Val Acc
─────   ──────────   ────────   ─────────   ───────   ──────────   ────────   ─────────   ───────
  1       0.7508      0.5206     0.4352     0.4395     0.5641      0.5698     81.09%     81.44%
  2       0.5124      0.4802     0.4674     0.4715     0.6016      0.6066     81.83%     82.22%
  3       0.4789      0.4608     0.4866     0.4882     0.6255      0.6262     82.31%     82.61%
  4       0.4625      0.4502     0.4944     0.4939     0.6323      0.6316     82.70%     82.94%
  5       0.4511      0.4414     0.5045     0.5025     0.6432      0.6401     82.98%     83.21%
  6       0.4431      0.4348     0.5139     0.5095     0.6523      0.6474     83.17%     83.32%
  7       0.4337      0.4302     0.5213     0.5163     0.6608      0.6551     83.24%     83.38%
  8       0.4269      0.4267     0.5262     0.5199     0.6645      0.6587     83.31%     83.45%
  9       0.4233      0.4243     0.5288     0.5226     0.6683      0.6618     83.40%     83.50%
 10       0.4248      0.4242     0.5291     0.5222     0.6684      0.6609     83.47%     83.55%
```

### 🎯 Test Evaluation — Per-Class IoU (Mean IoU: **0.2849**)

| Class | IoU | Performance |
|---|---|---|
| **Sky** | 0.9540 | 🟢 Excellent |
| **Landscape** | 0.5931 | 🟢 Good |
| **Dry Grass** | 0.4277 | 🟡 Moderate |
| **Trees** | 0.2261 | 🟡 Fair |
| **Dry Bushes** | 0.1298 | 🔴 Low |
| **Rocks** | 0.0475 | 🔴 Low |
| **Lush Bushes** | 0.0002 | 🔴 Minimal |
| **Background** | 0.0000 | ⚫ None |
| **Ground Clutter** | 0.0000 | ⚫ None |
| **Logs** | 0.0000 | ⚫ None |

> **Note:** The model excels at large, well-defined regions (Sky, Landscape, Dry Grass) while struggling with smaller or underrepresented classes — a common challenge in semantic segmentation with class imbalance.

### 📊 Per-Class Metrics Visualization

![Per-Class Metrics](per_class_metrics.png)

---

## 📂 Repository Structure

```
SEMANTIC_SEGMENTATION_CODE_CRUNCH/
│
├── 📓 TRAIN_CODE.ipynb            # Main training notebook (model, training loop, evaluation)
├── 📓 final-test-results.ipynb    # Test inference & evaluation notebook
├── 📄 evaluation_metrics.txt      # Test set evaluation metrics (Mean IoU + per-class)
├── 📊 per_class_metrics.png       # Per-class IoU visualization chart
├── 📜 LICENSE                     # MIT License
│
├── 📁 TRAIN RESULS/               # Training logs & metrics
│   └── evaluation_metrics.txt     # Detailed per-epoch training history
│
└── 📁 TEST_IMAGES_Results/        # Sample test images & predictions
    └── Test_Images                # Test image samples
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision transformers
```

### Training

1. Open `TRAIN_CODE.ipynb` in Jupyter / Google Colab
2. Configure dataset paths to your train images and masks
3. Run all cells — the DINOv2 backbone loads automatically with frozen layers
4. Training runs for **10 epochs** with AdamW + Cosine Scheduler

### Inference & Evaluation

1. Open `final-test-results.ipynb`
2. Load the trained model checkpoint
3. Run inference on the test set to reproduce the **0.2849 Mean IoU** result

---

## 💡 Key Takeaways

- **Frozen DINOv2** provides powerful feature representations with **zero training cost** on the backbone
- **10 epochs** were sufficient to achieve convergence thanks to the pretrained features
- **Sky (0.95 IoU)** and **Landscape (0.59 IoU)** are segmented with high fidelity
- Class imbalance remains the main bottleneck — future work could explore:
  - Weighted loss functions
  - Advanced augmentations (CutMix, MixUp)
  - Unfreezing top DINOv2 layers for fine-tuning
  - Multi-scale decoder heads (FPN / UPerNet)

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🔥 PyTorch & 🦕 DINOv2 for the Code Crunch Hackathon**

</div>
