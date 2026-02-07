# Experimental Pipeline (src/main)

This directory documents the **core experimental development pipeline** of the senior graduation project.
All major stages of dataset preparation, feature extraction, model training, and evaluation were originally
implemented and executed within a comprehensive Jupyter notebook (`senior.ipynb`).

Due to its **research-oriented, exploratory, and extensive nature**, the notebook itself is **not included** in this repository.
Instead, this README provides a **complete and structured description** of what was implemented,
how intermediate artifacts were produced, and how other directories in the repository were generated.

---

## 🎯 Problem Statement and Objectives

The primary objective of this project is to design and evaluate a **robust speech-based emotion recognition system**
using classical machine learning, deep learning, and state-of-the-art pretrained speech models.

Key goals include:

* Investigating the effectiveness of different acoustic feature representations
* Comparing ML, DL, and SOTA approaches under a unified evaluation protocol
* Ensuring reproducibility and fair benchmarking
* Demonstrating real-world usability via a GUI-based inference system

---

## 📊 Dataset Overview (MELD)

The project is based on the **MELD (Multimodal EmotionLines Dataset)**, a dialogue-level emotion recognition dataset
derived from the TV series *Friends*.

* Utterance-level emotion labels
* Multi-speaker conversational setting
* Audio, text, and video modalities

Due to licensing constraints, **raw data is not included**.
Dataset organization and preparation steps are documented under:

`archive_meld/README.md`

---

## 🧹 Dataset Cleaning & Filtering

Within the main notebook, extensive preprocessing was performed, including:

* Removal of corrupted or missing samples
* Speaker-based filtering (e.g., Joey-specific subsets)
* Alignment of audio, text, and annotation files
* Consistency checks across train/dev/test splits

These steps generated multiple intermediate CSV files and filtered datasets
that were stored under `archive_meld/` and `extra_files/`.

---

## 🗣️ Whisper Transcription & PyAnnote Diarization

GPU-based speech processing stages were executed on **Google Colab**, including:

* Automatic speech transcription using **OpenAI Whisper**
* Speaker diarization using **PyAnnote**
* Segment-level alignment and validation

Inputs and outputs of these stages were archived locally under:

* `archive_meld/colab_inputs_*`
* `archive_meld/colab_outputs_*`

---

## 🔊 Audio Preprocessing

Audio preprocessing steps included:

* Resampling to **16 kHz**
* Channel normalization (mono conversion)
* Silence trimming and normalization
* Segmentation aligned with utterance metadata

Processed audio files served as inputs for both feature extraction and model training stages.

---

## 🔁 Data Augmentation

To address class imbalance and improve generalization, multiple augmentation strategies were applied:

* Noise injection
* Pitch shifting
* Time stretching
* ..
* ..

Augmented samples were generated programmatically and stored in hidden or ignored directories
(excluded from version control).

---

## 🧠 Feature Extraction

Multiple feature extraction paradigms were explored:

### 🔹 Machine Learning (ML) Features

* MFCCs (+ deltas)
* Spectral features (centroid, bandwidth, roll-off, contrast)
* Prosodic features (pitch, energy, jitter, shimmer, formants)
* Wavelet-based features
* openSMILE (eGeMAPS)
* ..
* ..

### 🔹 Deep Learning (DL) Representations

* Time-frequency representations (Mel-spectrograms)
* CNN-compatible 2D feature maps
* Sequence-based inputs for RNN/BiLSTM models
* ..
* ..

### 🔹 State-of-the-Art (SOTA) Embeddings

* HuBERT
* wav2vec 2.0
* WavLM

Extracted features and embeddings were saved under:

`features_and_models/`

(Note: large feature files are excluded from this repository.)

---

## 📉 Feature Selection & Dimensionality Reduction

To reduce redundancy and improve efficiency:

* Statistical feature selection methods were applied
* PCA-based dimensionality reduction was used
* Feature normalization and scaling pipelines were established

---

## ⚖️ Undersampling (KMeans + PCA)

To mitigate class imbalance:

* KMeans-based undersampling strategies were explored
* PCA was used to preserve informative variance
* Balanced datasets were generated for fair comparison

---

## 🤖 Models Used

### 🔹 Machine Learning (ML)

* SVM
* Random Forest
* Gradient Boosting
* ..
* ..

### 🔹 Deep Learning (DL)

* CNN
* RNN / BiLSTM
* Hybrid architectures
* ..
* ..


### 🔹 State-of-the-Art (SOTA)

* End-to-end HuBERT fine-tuning
* Pretrained embeddings + MLP classifiers
* ..
* ..

---

## 📈 Model Training & Evaluation

All models were trained under a **strict Train / Dev / Test protocol**:

* Dev set used for hyperparameter tuning
* Test set used only for final evaluation

Metrics included:

* Accuracy
* F1-score
* Confusion matrices

Results are documented under:

`results/`

---

## 🔮 Prediction & Experimental Results

Prediction pipelines were implemented both:

* Offline (batch evaluation)
* Online (GUI-based inference)

Dedicated prediction scripts are located under:

`src/gui/`

---

## 🖥️ GUI Interface (Real-Time Deployment)

A lightweight GUI was developed to demonstrate real-world usability:

* Audio file loading
* Feature extraction
* Emotion prediction using pretrained models

The GUI can be launched via:

```bash
python src/gui/gui.py
```

---

## 🛠️ Tools, Libraries, and IDEs Used

* Python
* PyTorch, torchaudio
* HuggingFace Transformers
* scikit-learn
* Whisper, PyAnnote
* openSMILE, Praat (Parselmouth)
* Google Colab (GPU)
* Jupyter Notebook
* VS Code

---

## 🔚 Conclusion & Future Work

This experimental pipeline enabled a comprehensive comparison of emotion recognition approaches
and demonstrated the effectiveness of SOTA speech representations.

Future work may include:

* Multimodal fusion (audio + text + video)
* Larger-scale datasets
* Real-time streaming inference

---

## 🙏 Acknowledgements

* MELD Dataset authors
* OpenAI Whisper
* HuggingFace community

This README serves as a **conceptual and methodological replacement** for the excluded notebook
and ensures transparency, reproducibility, and academic completeness.

---

## 📬 Access to the Full Experimental Notebook

The complete experimental notebook (`senior.ipynb`) contains the full implementation of
all data preparation, feature extraction, model training, and evaluation stages described above.

Due to its **exploratory nature**, **extensive length**, and the presence of
dataset-dependent paths and intermediate artifacts, the notebook is **not publicly included**
in this repository.

However, the notebook can be shared **upon request for academic or research purposes**,
such as reproducibility verification, thesis evaluation, or potential collaboration.

Please note that the notebook is not self-contained and depends on
dataset-specific paths, large intermediate artifacts, and external execution environments.

