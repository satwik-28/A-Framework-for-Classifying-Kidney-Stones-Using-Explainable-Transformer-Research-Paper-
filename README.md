# 🫘 Kidney Disease Classification Using Explainable Transformers

[![Institution](https://img.shields.io/badge/Institution-KIIT%20University-green)](https://kiit.ac.in)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)](https://colab.research.google.com/)
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.37%25-brightgreen)]()

A deep learning framework for **multi-class kidney disease classification** from CT scan images, benchmarking six architectures — CNNs and Vision Transformers — with **Grad-CAM explainability**. Published as a research paper and minor project at KIIT University, Bhubaneswar (2025–2026).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Notebooks](#notebooks)
- [Models & Architecture](#models--architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Explainability — Grad-CAM](#explainability--grad-cam)
- [Team](#team)
- [Citation](#citation)

---

## 🔍 Overview

Kidney disease affects over 10% of the global population. The severe shortage of nephrologists — approximately 1 per million in South Asia versus 25.3 per million in Europe — makes automated AI-assisted diagnosis critical. This project builds an end-to-end diagnostic pipeline that classifies abdominal CT scans into four renal categories:

| Label | Encoded | Description |
|-------|---------|-------------|
| Cyst | 0 | Fluid-filled sac on kidney surface |
| Normal | 1 | Healthy kidney tissue |
| Stone | 2 | Crystalline deposits (nephrolithiasis) |
| Tumor | 3 | Renal cell carcinoma |

Six deep learning models are implemented and benchmarked. Grad-CAM is adapted for both convolutional and transformer architectures to provide visual explanations for clinical validation.

---

## 📊 Dataset

**CT KIDNEY DATASET: Normal–Cyst–Tumor and Stone**  
Source: [Kaggle — nazmul0087](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

| Class | Images |
|-------|--------|
| Normal | 5,077 |
| Cyst | 3,709 |
| Tumor | 2,283 |
| Stone | 1,377 |
| **Total** | **12,446** |

- Collected from hospital PACS systems in Dhaka, Bangladesh
- Coronal and axial slices from contrast-enhanced and non-contrast CT exams
- Converted from DICOM → JPEG; all patient-identifiable metadata removed
- Each image reviewed and labeled by a doctor and medical technologist

**Dataset split and balancing:**

| Subset | Proportion | Shuffle | Notes |
|--------|-----------|---------|-------|
| Training | 80% | Yes | Balanced via `RandomOverSampler` |
| Validation | 10% | Yes | Original distribution preserved |
| Test | 10% | No | Original distribution preserved |

`stratify=category_encoded` and `random_state=42` are used throughout for reproducibility. After oversampling, all four classes are equally represented in the training set.

---

## 📁 Repository Structure

```
kidney-disease-classifier/
│
├── Dino.ipynb                  # DINOViT custom implementation + Grad-CAM
├── EAnet.ipynb                 # External Attention Network (EANet)
├── ResNet50.ipynb              # ResNet50 transfer learning
├── Swin.ipynb                  # Swin Transformer (PyTorch)
├── VGG_kidney_stone.ipynb      # VGG16 transfer learning
├── ViT.ipynb                   # Custom Vision Transformer
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

All notebooks are designed for **Google Colab** with **Google Drive** mounted. The dataset must be uploaded to Drive before running.

### For TensorFlow notebooks (VGG16, ResNet50, ViT, DINO, EANet)

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn opencv-python pillow transformers
```

### For the PyTorch notebook (Swin Transformer)

```bash
pip install torch torchvision tqdm scikit-learn matplotlib seaborn
```

### Mount Drive (first cell in each notebook)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set your dataset base path:

```python
base_path = '/content/drive/MyDrive/<your_folder>/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
categories = ["Cyst", "Normal", "Stone", "Tumor"]
```

---

## 📓 Notebooks

| Notebook | Framework | Key Libraries |
|----------|-----------|---------------|
| `ViT.ipynb` | TensorFlow / Keras | `tensorflow`, `sklearn`, `imbalanced-learn` |
| `Dino.ipynb` | TensorFlow / Keras | `tensorflow`, `transformers`, `cv2` |
| `EAnet.ipynb` | TensorFlow / Keras | `tensorflow`, `sklearn` |
| `VGG_kidney_stone.ipynb` | TensorFlow / Keras | `tensorflow`, `sklearn` |
| `ResNet50.ipynb` | TensorFlow / Keras | `tensorflow`, `sklearn`, `PIL` |
| `Swin.ipynb` | PyTorch | `torch`, `torchvision`, `tqdm` |

Each TF/Keras notebook follows the same shared pipeline:

```
Mount Drive
  → Build DataFrame (image_path, label)
  → Duplicate & null checks
  → LabelEncoder (alphabetical: Cyst=0, Normal=1, Stone=2, Tumor=3)
  → train_test_split(80/20, stratify) → second split(50/50) → val / test
  → RandomOverSampler on training set
  → Cast encoded labels to str (Keras generator compatibility)
  → ImageDataGenerator(rescale=1./255) + flow_from_dataframe(224×224, batch=16)
  → Build model → compile → fit with EarlyStopping + ModelCheckpoint
  → Evaluate: classification_report, confusion matrix, accuracy/loss plots
```

The PyTorch Swin notebook uses `torchvision.datasets.ImageFolder` + `random_split` instead.

---

## 🧠 Models & Architecture

### VGG16 — `VGG_kidney_stone.ipynb` (TensorFlow/Keras)

```python
base = tf.keras.applications.VGG16(
    include_top=False, input_shape=(224,224,3),
    pooling='avg', weights='imagenet'
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')
], name="VGG16")

model.compile(
    optimizer=Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### ResNet50 — `ResNet50.ipynb` (TensorFlow/Keras)

```python
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# All base layers frozen

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64,  activation='relu')(x)
x = Dense(4,   activation='softmax')(x)

model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)
```

- **Data augmentation** (training only): rotation 20°, width/height shift 0.2, zoom 0.2, horizontal flip
- **Callbacks:** `ReduceLROnPlateau(monitor='val_recall', factor=0.1, patience=5)`, `EarlyStopping(monitor='val_recall', patience=5)`
- **Batch size:** 32

---

### Vision Transformer (ViT) — `ViT.ipynb` (TensorFlow/Keras, custom)

Built from scratch with custom Keras layers:

```python
# 1. Patch Embedding — Conv2D projection (patch_size=16)
class PatchEmbedding(layers.Layer): ...

# 2. Multi-Head Self-Attention
class MultiHeadSelfAttention(layers.Layer):
    self.attention = layers.MultiHeadAttention(...)

# 3. Transformer Encoder Block
#    LayerNorm → MHSA → residual → LayerNorm → MLP(GELU) → residual

# 4. CLS token + positional embedding prepended to patch sequence
# 5. Classification head: Dense(4, softmax) on CLS token output

model.compile(
    optimizer=Adam(lr=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

- **Epochs:** 4
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, `ModelCheckpoint('best_vit_model.keras')`

---

### DINOViT — `Dino.ipynb` (TensorFlow/Keras, custom)

```python
dino_model = DINOViT(
    image_size  = (224, 224),
    patch_size  = 16,
    embed_dim   = 384,
    num_heads   = 6,
    num_blocks  = 12,
    mlp_dim     = 1536,
    num_classes = 4
)

dino_model.compile(
    optimizer=Adam(lr=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

- **Two-phase training:** initial run (5 epochs) → save → reload weights into fresh instance → second run (5 epochs)
- Saved as `best_dino_model.keras` → `best_dino_model_v2.keras`
- Custom Grad-CAM computed on token-level embeddings from the last transformer block (see [Explainability](#explainability--grad-cam))

---

### EANet — `EAnet.ipynb` (TensorFlow/Keras, custom)

Channel attention via squeeze-and-excitation blocks:

```python
def efficient_attention_block(x, reduction_ratio=8):
    channels = x.shape[-1]
    avg_pool = GlobalAveragePooling2D()(x)
    dense1   = Dense(channels // reduction_ratio, activation='relu')(avg_pool)
    dense2   = Dense(channels, activation='sigmoid')(dense1)
    scale    = Multiply()([x, Reshape((1,1,channels))(dense2)])
    return scale

# Architecture: Conv2D stem (stride=2) → stacked attention blocks → classifier
```

- **Epochs:** 5
- **Saved as:** `best_eanet_model.keras`

---

### Swin Transformer — `Swin.ipynb` (PyTorch)

```python
class Config:
    model_name  = "swin_base"   # torchvision.models.swin_b
    pretrained  = True
    num_classes = 4
    image_size  = 224
    batch_size  = 16
    epochs      = 5
    lr          = 1e-4
    weight_decay = 1e-4

# Partial fine-tuning: layers.0 and layers.1 frozen
# layers.2, layers.3, norm, head → trainable

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=config.lr, weight_decay=config.weight_decay)
scaler    = torch.amp.GradScaler("cuda")  # mixed-precision training
```

- **Transforms:** `Resize(224)` → `RandomHorizontalFlip` → `RandomRotation(15°)` → `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])`
- **Saved as:** `swin_base_kidney_ct.pth`

---

## 🏋️ Training Configuration

| Setting | Value |
|---------|-------|
| Input image size | 224 × 224 |
| Color mode | RGB (3 channels) |
| Pixel normalization (TF) | `rescale = 1./255` |
| Pixel normalization (PyTorch) | ImageNet mean/std |
| Batch size | 16 (all models) / 32 (ResNet50) |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 5 epochs |
| `restore_best_weights` | True |
| `random_state` / seed | 42 |
| Training hardware | Google Colab Pro — 26.3 GB RAM, 16 GB GPU, CUDA 11.2 |

---

## 📈 Results

| Model | Framework | Accuracy |
|-------|-----------|----------|
| **Vision Transformer (ViT)** | TF / Keras | **99.37%** |
| VGG16 | TF / Keras | 98.47% |
| EANet | TF / Keras | 98.35% |
| DINO | TF / Keras | 98.00% |
| Swin Transformer | PyTorch | 97.27% |
| ResNet50 | TF / Keras | 94.02% |

### Per-Class Metrics

| Model | Class | Precision | Recall | F1 |
|-------|-------|:---------:|:------:|:--:|
| EANet | Cyst | 0.96 | 0.98 | 0.97 |
| | Normal | 0.99 | 0.98 | 0.98 |
| | Stone | 1.00 | 0.84 | 0.91 |
| | Tumor | 0.92 | 1.00 | 0.96 |
| Swin | Cyst | 0.94 | 0.99 | 0.97 |
| | Normal | 1.00 | 0.99 | 0.99 |
| | Stone | 0.98 | 0.86 | 0.92 |
| | Tumor | 0.96 | 0.98 | 0.97 |
| DINO | Cyst | 1.00 | 0.99 | 0.99 |
| | Normal | 1.00 | 0.95 | 0.98 |
| | Stone | 0.96 | 1.00 | 0.98 |
| | Tumor | 0.92 | 1.00 | 0.96 |
| VGG16 | Cyst | 0.98 | 0.99 | 0.98 |
| | Normal | 0.97 | 0.99 | 0.98 |
| | Stone | 0.98 | 0.97 | 0.97 |
| | Tumor | 1.00 | 0.95 | 0.97 |
| **ViT** | **Cyst** | **1.00** | **1.00** | **1.00** |

Key observations:
- ViT achieves near-perfect scores across all classes with minimal false negatives — critical for clinical safety
- ResNet50 struggles most on the Stone class due to weak lesion localization
- Transformer models converge faster and generalize better than CNN baselines
- DINO and VGG16 produce the most interpretable and anatomically precise Grad-CAM heatmaps

---

## 🔎 Explainability — Grad-CAM

Grad-CAM is adapted separately for CNN and Transformer architectures.

**Mathematical formulation:**

```
Relevance weights:   w_k^c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_ij
Heatmap:             L_GradCAM = ReLU( Σ_k  w_k^c · A^k )
```

**CNN models (VGG16, ResNet50):** Gradients computed from the last convolutional layer's spatial feature maps → pooled → weighted sum → ReLU.

**Transformer models (ViT, DINO, EANet):** Gradients computed on token embeddings from the final encoder block. The 196 patch tokens (14×14 spatial grid) are reshaped and upsampled back to 224×224.

**DINO Grad-CAM — `compute_dino_gradcam()`:**

```python
with tf.GradientTape() as tape:
    x = inner_model.patch_embed(img_array)          # patch embeddings
    x = tf.concat([cls_tokens, x], axis=1)          # prepend CLS token
    x = x + inner_model.pos_embed                   # add positional encoding
    for block in inner_model.transformer_blocks:
        x = block(x, training=False)                # final x = target_block_output
    preds = inner_model.classifier(norm(x)[:, 0])  # classify on CLS token
    loss  = preds[:, class_index]

grads   = tape.gradient(loss, target_block_output)
weights = tf.reduce_mean(grads, axis=-1)            # pool over embedding dim
heatmap = tf.einsum('btp,btp->bt', weights, target_block_output)

# Reshape: (196,) → (14,14) → cv2.resize → (224,224)
# Superimpose: cv2.addWeighted(heatmap_color, 0.5, original_img, 0.5, 0)
```

`plot_gradcam_grid()` renders a 10×3 grid of original CT scans paired with their heatmaps across all test classes for batch qualitative review.

---

## 👥 Team

School of Computer Engineering, KIIT Deemed to be University (2025–2026)  
**Supervisor:** Prof. Roshini Pradhan

| Name | Roll No | Primary Contribution |
|------|---------|---------------------|
| Satwik Gupta | 2305075 | Data preprocessing, ViT implementation & tuning |
| Saurabh Sharma | 2305076 | Literature review, project pipeline design |
| Shirsh Mohan | 2305245 | Data preprocessing, ViT training & hyperparameter tuning |
| Soham Dey | 2305250 | Dataset organization, pipeline integration & testing |
| Swayam Kashyap | 2305258 | VGG16 & ResNet50 implementation, evaluation metrics |

---

## 📄 Citation

If you use this code or findings in your work, please cite:

```bibtex
@article{mohan2025kidney,
  title       = {A Framework for Classifying Kidney Stones Using Explainable Transformer},
  author      = {Mohan, Shirsh and Gupta, Satwik and Kashyap, Swayam and Sharma, Saurabh and Dey, Soham},
  institution = {Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar, Odisha, India},
  year        = {2025}
}
```

Dataset citation:

```bibtex
@dataset{islam2021ctkidney,
  author    = {Islam, M.},
  title     = {CT Kidney Dataset: Normal-Cyst-Tumor and Stone},
  year      = {2021},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **⚕️ Clinical Disclaimer:** This system is a research-grade decision-support tool and is not approved for standalone clinical diagnosis. Real-world deployment requires validated multi-center datasets, regulatory approval, and integration under qualified medical supervision.
