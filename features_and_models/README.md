# Features and Models

This directory contains all **extracted features**, **trained models**, and **experimental results**
used in the study.  
It includes **classical machine learning**, **deep learning**, and **state-of-the-art pretrained
speech representation models** applied to the MELD dataset.

Most of the feature extraction and model training steps were executed on **Google Colab**
due to GPU requirements. Generated artifacts were downloaded and organized here
in a reproducible and structured manner.

---

## Directory Overview

### 1. autoencoders/
Autoencoder-based dimensionality reduction and feature selection artifacts.

**Contents:**
- Trained encoder weights (`.pth`)
- Feature scalers (`.pkl`)
- Boruta + Autoencoder selected feature lists

Used for:
- Feature compression
- Hybrid feature selection pipelines

---

### 2. balanced_1/
Balanced versions of feature sets used for ML experiments.

Includes:
- Cleaned feature matrices
- Feature-selected datasets (Boruta, RF, RFE, SFFS, KBest, Autoencoder, hybrids)

All files are **class-balanced** versions of the corresponding ML feature sets.

---

### 3. dl_features/
Deep learning input representations.

#### Subfolders include:
- **bilstm/** – Sequential features for BiLSTM
- **cnn_bilstm_mel128/** – Mel-spectrogram-based CNN-BiLSTM inputs
- **keras_bilstm/** – Keras-based MFCC/BiLSTM inputs
- **spectrogram_images/** – Image-based representations:
  - Mel
  - MFCC
  - RGB
  - 224x224 variants
  - Train / validation splits

Each spectrogram directory is organized by **emotion class**.

---

### 4. dl_models/
Trained deep learning models.

Includes:
- BiLSTM (PyTorch)
- CNN-BiLSTM
- BiLSTM (Keras)
- CNN-based spectrogram classifiers
    - Mel
    - MFCC
    - Multi (Mel + MFCC)
    - RGB
- EfficientNet / ResNet models
- Pytorch MLP
- Keras MLP

Also contains:
- Summary CSV files with **performance comparisons across feature sets**

---

### 5. ml_features/
Raw and selected **classical ML feature sets**.

Includes:
- Baseline feature matrices
- Autoencoder-selected features
- Boruta, RF, RFE, KBest, SFFS selections

Used as input for traditional ML classifiers.

---

### 6. ml_models/
Classical machine learning models trained on different feature sets.

Algorithms include:
- KNN
- Random Forest
- Gradient Boosting
- SVM
- XGBoost
- Bagging
- Voting / Soft Voting
- Sklearn MLP

Each subfolder contains:
- Best trained model
- Corresponding scaler
- Label encoder

Performance summary CSV files are provided per algorithm.

---

### 7. sota_features_pretrained/
Extracted features from **pretrained speech models**:

- HuBERT
- Wav2Vec 2.0
- WavLM

Includes:
- Raw features
- Balanced feature versions
- Label encoders

---

### 8. sota_models_pretrained/
MLP-based classifiers trained on pretrained speech representations.

Each model directory contains:
- Trained model weights
- Scaler
- Label encoder
- Classification report

---

### 9. sota_models_transformer/
Transformer-based fine-tuned models.

Includes:
- HuBERT
- Wav2Vec 2.0
- WavLM

Each model directory contains:
- Fine-tuned model weights
- Configuration files
- Training metadata

Associated Colab notebooks are included for reproducibility.

---

## Notes

- This directory is **excluded from GitHub** due to size and licensing constraints.
- All files are preserved locally for **experimental reproducibility**.
- Paths and naming conventions are kept consistent across experiments.

---

## Purpose

This folder serves as a **complete experimental archive**, supporting:
- Feature extraction analysis
- Model comparison
- Reproducibility
- Thesis evaluation and future extensions
