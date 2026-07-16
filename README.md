# Audio Deepfake Detection using CNNs and Transformers

A PyTorch implementation comparing convolutional neural networks and transformer-based architectures for audio deepfake detection using Mel spectrogram classification.

This project evaluates seven deep learning architectures under a unified training pipeline and investigates optimization strategies including architecture-specific fine-tuning, layer-wise learning rate decay, cosine annealing warm restarts, gradient clipping, and Equal Error Rate (EER)-driven evaluation.

## Features

- Mel Spectrogram preprocessing with offline feature caching
- RAM-cached dataset loading for fast training
- ResNet18
- ResNet50
- CNN-Transformer Hybrid
- Vision Transformer (ViT)
- DeiT (Hard & Soft Distillation)
- Custom Swin Transformer
- Layer-wise Learning Rate Decay (LLRD)
- Cosine Annealing Warm Restarts
- AdamW optimizer
- Gradient Clipping
- Checkpoint System
- Early Stopping
- ROC Curve
- Equal Error Rate (EER)
- Confusion Matrix
- F1 Score
- Recall

## Dataset

Dataset:
ASVspoof2019 Logical Access (LA)

The dataset is not included in this repository because of licensing restrictions.

Directory structure:

Datasets/
    ASVspoof2019_LA_train/
    ASVspoof2019_LA_dev/
    ASVspoof2019_LA_eval/

## 📚 Implemented Architectures

| Architecture | Description | Status |
|:-------------|:------------|:------:|
| **ResNet18** | Residual Convolutional Neural Network | ✅ |
| **ResNet50** | Deep Residual Convolutional Neural Network | ✅ |
| **CNN-Transformer** | CNN feature extractor with Transformer encoder | ✅ |
| **Vision Transformer (ViT)** | Pure Transformer-based image classifier | ✅ |
| **DeiT-Hard** | Data-efficient Image Transformer (Hard Distillation) | ✅ |
| **DeiT-Soft** | Data-efficient Image Transformer (Soft Distillation) | ✅ |
| **Swin Transformer** | Hierarchical Window-based Vision Transformer | ✅ |

## 📂 Data Pipeline

```text
ASVspoof2019 Audio (.flac)
            │
            ▼
      Librosa Loader
            │
            ▼
   Mel Spectrogram (128 bins)
            │
            ▼
      Power-to-dB
            │
            ▼
 Pad / Crop to 128 × 128
            │
            ▼
 Save as NumPy (.npy)
            │
            ▼
      RAM Feature Cache
```

## 🚀 Training Pipeline

```text
RAM Cached Features
        │
        ▼
 PyTorch DataLoader
        │
        ▼
 Data Augmentation
 (Gaussian Noise)
        │
        ▼
      Model
        │
        ▼
 BCEWithLogitsLoss
        │
        ▼
 AdamW Optimizer
        │
        ▼
 Gradient Clipping
        │
        ▼
 Cosine Annealing
 Warm Restarts
        │
        ▼
 Checkpointing
        │
        ▼
 Evaluation
```
## Experimental Pipeline

Phase I
- Unified baseline training

Phase II
- Architecture comparison

Phase III
- Architecture-specific fine-tuning
- Layer-wise learning rate decay
- Cosine Annealing Warm Restarts
- AdamW
- Gradient Clipping
- EER-based checkpoint selection

## 📊 Results

The final models were evaluated on the **ASVspoof2019 Logical Access (LA) evaluation set** using **Test Accuracy**, **ROC-AUC**, and **Equal Error Rate (EER)**. Lower EER indicates better spoof detection performance.

| Rank | Model | Test Accuracy | ROC-AUC | EER ↓ |
|:---:|:------------------|:------------:|:-------:|:------:|
| 🥇 | **Swin Transformer** | **92.85%** | 0.9767 | **7.14%** |
| 🥈 | **ResNet50** | **92.85%** | **0.9797** | 7.15% |
| 🥉 | **ResNet18** | 92.34% | **0.9825** | 7.65% |
| 4 | **CNN-Transformer** | 92.11% | 0.9800 | 7.89% |
| 5 | **Vision Transformer (ViT)** | 91.67% | 0.9689 | 8.32% |
| 6 | **DeiT-Hard** | 90.04% | 0.9620 | 9.97% |
| 7 | **DeiT-Soft** | 89.25% | 0.9535 | 10.75% |

> **Note:** Results correspond to the **best coherent evaluation checkpoints** obtained after architecture-specific fine-tuning. Model selection was based on agreement between **Equal Error Rate (EER)**, **ROC-AUC**, **classification accuracy**, and additional diagnostic analyses to avoid misleading results caused by optimization instability.

## Key Findings

- Swin Transformer achieved the lowest observed EER after architecture correction.
- ResNet50 remained the strongest CNN baseline.
- Aggressive fine-tuning produced misleadingly low EER values in some checkpoints.
- Cross-validation using ROC-AUC, confusion matrices, and activation statistics proved essential for reliable model evaluation.
- Layer-wise activation tracing identified numerical instability within the custom Swin implementation.

## Future Work

- BEATs
- AST (Audio Spectrogram Transformer)
- Self-supervised Audio Transformers
