# Speech-Based Emotion Recognition System

This repository contains the implementation of a **speech emotion recognition system**
developed as a **senior graduation project**.

The project focuses on extracting advanced acoustic features from speech signals and
training state-of-the-art machine learning models for emotion classification.
Special attention is given to robustness, feature diversity, and real-world applicability.

---

## 📌 Project Scope

The system includes:

- Advanced **audio feature extraction** (time-domain, frequency-domain, cepstral, wavelet, and prosodic features)
- **Speech transcription and diarization** using Whisper and PyAnnote (GPU-based, Colab)
- Training and evaluation of **state-of-the-art emotion recognition models**
- A **GUI-based inference system** for practical usage
- Experiments conducted on the **MELD dataset**

---

## 🗂️ Repository Structure

```text
senior-project/
├── src/
│   ├── gui/                  # GUI and inference scripts
│   └── main/                 # Core Jupyter notebooks
│
├── archive_meld/             # MELD dataset structure (no raw data included)
│   ├── audio/
│   ├── video/
│   ├── text/
│   ├── colab_inputs/
│   └── colab_outputs/
│
├── features_and_models/      # Extracted features and trained models (not included due to size)
│
├── results/                  # Experimental results and evaluations
│   ├── ml/
│   ├── dl/
│   └── sota/
│
├── reports/                  # Final report and presentation
│
├── extra_files/              # Auxiliary and temporary files
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Information

This project uses the **MELD (Multimodal EmotionLines Dataset)**.

Due to licensing and size constraints, **raw audio/video files and extracted datasets are NOT included**
in this repository.

Please refer to:

archive_meld/README.md

for detailed instructions on how to obtain and organize the dataset.

---

## 🔄 Overall System Pipeline

This project follows a **multi-stage end-to-end pipeline** for speech-based emotion recognition,
designed to ensure robustness, reproducibility, and real-world applicability.

The overall workflow is summarized below:

- Dataset Overview (MELD)
- Dataset Cleaning & Filtering
- Whisper Transcription + PyAnnote (Speaker Diarization)
- Audio Preprocessing
- Data Augmentation
- Feature Extraction
  - ML-based Handcrafted Features
  - DL-based Representations
  - SOTA Pretrained Embeddings
- Feature Selection & Dimensionality Reduction
- Undersampling
  - KMeans-based Undersampling
  - PCA-based Dimensionality Reduction
- Models Used
  - Machine Learning (ML)
  - Deep Learning (DL)
  - State-of-the-Art (SOTA) Models
- Model Training and Evaluation
- Prediction and Experimental Results
- GUI-based Inference

---

## 📌 Model Training and Evaluation Protocol

All experiments in this project follow a **strict Train / Development / Test split strategy**
to prevent data leakage and ensure fair evaluation.

- **Train set** is used for model training.
- **Development (Dev) set** is used for:
  - Hyperparameter tuning
  - Model selection
  - Early stopping
  - Threshold optimization
- **Test set** is used **only once** for final evaluation.

All quantitative results reported in this repository and in the thesis
are computed exclusively on the **test split** and are provided under the `results/` directory.

---

## 📌 Reproducibility Note

The project is structured such that:

- Executing the notebooks in order reconstructs intermediate datasets and outputs
- Large-scale data, pretrained models, and raw multimedia files are excluded
- Google Colab is required for GPU-intensive stages

> **Note:**  
> The main experimental development was conducted in Jupyter notebooks under `src/main/`.  
> Due to their exploratory and research-oriented nature, these notebooks are not publicly included.  
> Detailed documentation is provided within the corresponding directory.

---

## 🧠 Feature Extraction

The project employs **multiple feature extraction strategies** depending on the model category.
This design enables a fair comparison between traditional, deep learning, and state-of-the-art approaches.

### 🔹 Machine Learning (ML) Features

For classical machine learning models, **handcrafted acoustic features** are extracted, including:

- MFCCs and delta coefficients
- Spectral features (centroid, bandwidth, roll-off, contrast)
- Chroma-based features
- Wavelet-based representations
- Prosodic features (pitch, energy, jitter, shimmer, formants)
- openSMILE feature sets (eGeMAPS)

The primary implementation for ML-based feature extraction is located in:

src/gui/extract_advanced_ml_features.py

### ⚙️ External Dependencies (Manual Installation)

Some components of this project rely on external tools that must be installed manually:

- **openSMILE**  
  Used for extracting eGeMAPS acoustic feature sets.  
  openSMILE is not included in `requirements.txt` and must be installed separately.

  https://audeering.github.io/opensmile/

- **Praat** (via Parselmouth)  
  Used for prosodic feature extraction such as jitter, shimmer, HNR, and formants.

These tools are optional but required for full feature extraction functionality.


### 🔹 Deep Learning (DL) Representations

For deep learning models, feature extraction follows **representation learning** principles rather than
manual feature engineering. These include:

- Time-frequency representations (e.g., Mel-spectrograms, MFCC maps)
- CNN-compatible 2D inputs
- Sequence-based representations for RNN and BiLSTM architectures
- Hybrid CNN–RNN feature pipelines

Feature extraction and preprocessing for DL models are implemented within the corresponding
training and inference notebooks.


### 🔹 State-of-the-Art (SOTA) Pretrained Embeddings

For SOTA models, **pretrained self-supervised speech models** are used as feature extractors, including:

- HuBERT
- wav2vec 2.0
- WavLM

These models generate high-level audio embeddings that are used:
- Directly in end-to-end classification
- Or as inputs to downstream classifiers (e.g., MLP)

SOTA feature extraction and fine-tuning steps were executed on **Google Colab with GPU support**,
and only scripts and evaluation outputs are included in this repository.

---

## 🖥️ GPU-Based Processing (Colab)

Some components of this project require GPU acceleration, including:

- Whisper-based speech transcription
- PyAnnote-based speaker diarization
- Training of deep learning models

These processes were executed on **Google Colab**.
Only scripts and documentation are provided here; large outputs and pretrained models are excluded.

---

## 🪟 GUI-Based Inference

The project includes a lightweight graphical user interface (GUI) for emotion prediction
from speech recordings.

The GUI can be launched locally using:

```bash
python src/gui/gui.py
```

The interface allows users to load an audio file, extract acoustic features,
and obtain emotion predictions using pretrained models.

---

## 📄 Reports

The final project report and presentation slides are provided under:

reports/

This folder includes:
- Final project report (PDF)
- Project presentation slides (PPTX)

These documents are shared **for academic reference only**.

---

## ⚠️ Notes

- This repository is intended for **academic and research purposes**
- Large files, datasets, and pretrained model weights are intentionally excluded
- The project structure is designed for **reproducibility**, not direct execution out-of-the-box

---

## License
This project is licensed under the MIT License.

---

## 👤 Author

**Tuncay KÖSE**  
Senior Graduation Project  
