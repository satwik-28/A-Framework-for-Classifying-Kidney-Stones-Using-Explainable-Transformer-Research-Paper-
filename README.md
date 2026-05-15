# 🫘 Kidney Disease Classification Using Explainable Transformers

[![KIIT](https://img.shields.io/badge/Institution-KIIT%20University-green)](https://kiit.ac.in)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.37%25-brightgreen)]()

A deep learning framework for **multi-class kidney disease classification** from CT scan images using Vision Transformers (ViT) and explainability via Grad-CAM. Developed as a minor project at KIIT University, Bhubaneswar.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Explainability](#explainability)
- [Team](#team)
- [Citation](#citation)

---

## 🔍 Overview

Kidney disease affects over 10% of the global population, and the global shortage of nephrologists makes automated diagnosis critical. This project builds an AI-driven diagnostic framework that classifies CT kidney images into four categories:

| Class | Description |
|-------|-------------|
| **Normal** | Healthy kidney tissue |
| **Cyst** | Fluid-filled sac on the kidney surface |
| **Stone** | Crystalline deposits (nephrolithiasis) |
| **Tumor** | Renal cell carcinoma |

Key contributions:
- Benchmarking of **6 deep learning models** (CNN-based and Transformer-based)
- **Grad-CAM** explainability adapted for Vision Transformers and DINO
- **99.37% accuracy** achieved with Vision Transformer (ViT)
- Publicly available CT dataset with 12,446 annotated images

---

## 📊 Dataset

**CT KIDNEY DATASET: Normal–Cyst–Tumor and Stone**

| Split | Proportion | Samples |
|-------|-----------|---------|
| Training | 80% | ~9,957 |
| Validation | 10% | ~1,245 |
| Test | 10% | ~1,244 |

| Class | Count |
|-------|-------|
| Normal | 5,077 |
| Cyst | 3,709 |
| Tumor | 2,283 |
| Stone | 1,377 |
| **Total** | **12,446** |

- Images sourced from hospital PACS systems in Dhaka, Bangladesh
- Converted from DICOM to JPEG; all patient-identifiable metadata removed
- Annotated by a doctor and medical technologist
- Available on [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

---

## 🧠 Models

Six architectures were implemented and compared:

| Model | Type | Total Params | Trainable Params | Accuracy |
|-------|------|-------------|-----------------|----------|
| **Vision Transformer (ViT)** | Transformer | 50,688 | 50,688 | **99.37%** |
| VGG16 | CNN | 14,747,780 | 4,752,708 | 98.47% |
| EANet | Transformer | 600,907 | 600,900 | 98.35% |
| DINO | Self-supervised | 76,032 | 76,032 | 98.00% |
| Swin Transformer | Transformer | 412,788 | 396,372 | 97.27% |
| ResNet50 | CNN | 23,719,108 | 135,492 | 94.02% |

### Preprocessing Pipeline
1. Directory-based data collection → Pandas DataFrame
2. Duplicate removal & missing value inspection
3. Label encoding (alphabetical: Cyst=0, Normal=1, Stone=2, Tumor=3)
4. Stratified 80/10/10 train/val/test split
5. Random oversampling on training set (1,300 samples/class)
6. Resize to 224×224, normalize to [0, 1], RGB conversion
7. Batch size: 16 | Optimizer: Adam (lr=1e-5) | Early stopping (patience=5)

---

## 📈 Results

### Performance Summary

| Model | Class | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| **ViT** | Cyst | 1.00 | 1.00 | 1.00 |
| | Normal | — | — | — |
| | Stone | — | — | — |
| | Tumor | — | — | — |
| DINO | Cyst | 1.00 | 0.99 | 0.99 |
| | Stone | 0.96 | 1.00 | 0.98 |
| VGG16 | Cyst | 0.98 | 0.99 | 0.98 |
| | Tumor | 1.00 | 0.95 | 0.97 |

Key findings:
- ViT achieves near-perfect recall across all classes — critical for reducing missed diagnoses
- DINO and VGG16 produce the most clinically meaningful Grad-CAM heatmaps
- ResNet50 shows higher misclassification rates and poor lesion localization
- Transformer models converge faster and generalize better than CNNs

---

## 📁 Project Structure

```
kidney-disease-classifier/
│
├── data/
│   ├── raw/                    # Original CT images (Cyst, Normal, Stone, Tumor)
│   └── processed/              # Preprocessed DataFrames
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb  # Data pipeline
│   ├── 03_CNN_Models.ipynb     # VGG16, ResNet50
│   ├── 04_ViT_Model.ipynb      # Vision Transformer
│   ├── 05_DINO_EANet_Swin.ipynb# Advanced transformer models
│   └── 06_GradCAM.ipynb        # Explainability analysis
│
├── src/
│   ├── preprocessing.py        # Data loading & preprocessing
│   ├── models/
│   │   ├── vgg16.py
│   │   ├── resnet50.py
│   │   ├── vit.py
│   │   ├── dino.py
│   │   ├── eanet.py
│   │   └── swin.py
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Metrics & confusion matrix
│   └── gradcam.py              # Grad-CAM visualization
│
├── outputs/
│   ├── models/                 # Saved model weights
│   ├── plots/                  # Accuracy/loss curves, confusion matrices
│   └── gradcam/                # Heatmap visualizations
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/kidney-disease-classifier.git
cd kidney-disease-classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10
keras
torch
torchvision
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
opencv-python
Pillow
```

---

## 🚀 Usage

### 1. Prepare the dataset
Download the [CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) from Kaggle and place it under `data/raw/`.

### 2. Preprocess
```python
from src.preprocessing import build_dataframe, split_and_oversample

df = build_dataframe("data/raw/")
train_df, val_df, test_df = split_and_oversample(df)
```

### 3. Train a model
```python
from src.train import train_model

model, history = train_model(
    model_name="vit",       # Options: vgg16, resnet50, vit, dino, eanet, swin
    train_df=train_df,
    val_df=val_df,
    epochs=100,
    batch_size=16
)
```

### 4. Evaluate
```python
from src.evaluate import evaluate_model

evaluate_model(model, test_df)
# Outputs: accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
```

### 5. Generate Grad-CAM heatmaps
```python
from src.gradcam import generate_gradcam

generate_gradcam(model, image_path="sample_ct.jpg", target_class="Stone")
```

---

## 🔎 Explainability

Grad-CAM is adapted for both CNN-based and Transformer-based architectures:

- **CNNs (VGG16, ResNet50):** Gradients computed from the last convolutional layer
- **ViT / DINO:** Gradients computed from token-level embeddings of the final Transformer encoder block; attention maps upsampled to 224×224
- DINO self-supervised attention maps provide richer semantic localization, helping highlight clinically relevant kidney regions

Heatmaps are superimposed on original CT scans for visual verification that the model focuses on anatomically meaningful structures.

---

## 👥 Team

Developed at the **School of Computer Engineering, KIIT University** (2025–2026)

| Name | Roll No | Contribution |
|------|---------|-------------|
| Satwik Gupta | 2305075 | Data preprocessing, ViT implementation |
| Saurabh Sharma | 2305076 | Literature review, project pipeline design |
| Shirsh Mohan | 2305245 | Data preprocessing, ViT training & tuning |
| Soham Dey | 2305250 | Dataset organization, pipeline integration |
| Swayam Kashyap | 2305258 | VGG16 & ResNet50 implementation, evaluation |

**Supervisor:** Prof. Roshini Pradhan

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{mohan2025kidney,
  title={A Framework for Classifying Kidney Stones Using Explainable Transformer},
  author={Mohan, Shirsh and Gupta, Satwik and Kashyap, Swayam and Sharma, Saurabh and Dey, Soham},
  institution={Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **Disclaimer:** This system is intended as a decision-support tool and should not be used as a standalone diagnostic authority. Real-world clinical deployment requires validated datasets, ethical approvals, and regulatory clearance.
