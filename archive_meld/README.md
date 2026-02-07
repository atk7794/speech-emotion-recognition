# MELD Dataset Organization (archive_meld)

This directory contains the **organized, processed, and intermediate representations**
of the **MELD (Multimodal EmotionLines Dataset)** used throughout this project.

Due to **dataset licensing restrictions and file size limitations**, the original raw
audio and video files are **NOT included** in this repository.
This folder documents the **intended directory structure** and data flow used during
dataset preparation and experimentation.

---

## 📌 Dataset Overview

MELD is a multimodal emotion recognition dataset derived from the TV series *Friends*.
It contains:
- Dialogue-level utterances
- Emotion and sentiment annotations
- Multimodal inputs (text, audio, video)

Official dataset source:
- https://affective-meld.github.io/

Users must obtain the dataset directly from the official source and organize it
according to the structure described below.

---

## 🗂️ Directory Structure and Purpose

### 🔹 video/
Contains the **original MELD video utterances**, organized into predefined splits:

- `train_splits/`
- `test_splits/`
- `dev_splits/`

Each video corresponds to a single dialogue utterance (e.g., `diaX_uttY.mp4`).

---

### 🔹 text/
Contains the official MELD annotation files:

- `train_sent_emo.csv`
- `test_sent_emo.csv`
- `dev_sent_emo.csv`

These files provide emotion and sentiment labels aligned with utterance-level samples.

---

### 🔹 joey_video / joey_video_final
Derived video subsets focusing on a **single speaker (Joey)**.
These folders contain filtered and cleaned utterances used for speaker-specific experiments.

- `joey_video/` → intermediate filtering stages
- `joey_video_final/` → finalized video utterances used in experiments

---

### 🔹 joey_text / joey_text_final
Text annotation files aligned with Joey-only utterances.

- Intermediate CSV files are stored in `joey_text/`
- Final cleaned and aligned annotation files are stored in `joey_text_final/`

These CSV files serve as the primary metadata source for audio and video processing.

---

### 🔹 joey_audio/
Audio representations extracted from Joey video utterances.

Includes:
- Resampled audio at **16 kHz**
- Processed and advanced processed audio
- Augmented versions used for data expansion

Hidden folders (`.augmented_*`) contain augmentation outputs and are intentionally excluded
from version control.

---

## ☁️ Google Colab Inputs & Outputs

Some processing stages required **GPU acceleration** and were executed on **Google Colab**.
To maintain reproducibility, all inputs and outputs were archived and stored locally.

### 🔹 colab_inputs_1 / colab_inputs_2
Contain compressed inputs uploaded to Colab, including:
- Filtered video and text data
- Preprocessed audio
- Model-ready datasets

---

### 🔹 colab_outputs_1
Outputs from:
- Whisper-based transcription
- PyAnnote-based speaker diarization
- Audio cropping and alignment
- Transcription comparison and validation

Includes:
- Aligned CSV files
- JSON transcription outputs
- Processed video utterances
- Colab notebooks used for these steps

---

### 🔹 colab_outputs_2
Contains **SOTA feature extraction and model training outputs**, including:
- HuBERT, wav2vec 2.0, and WavLM feature embeddings
- Balanced and unbalanced feature sets
- Trained MLP classifiers
- Model evaluation reports
- Fine-tuning notebooks

These experiments were conducted on GPU-enabled Colab environments.

---

## ⚠️ Important Notes

- Raw MELD audio/video files are **not included**
- This directory documents the **intended data organization**
- Some subfolders are excluded via `.gitignore` due to size constraints
- Paths referenced in notebooks assume this directory structure

---

## 📌 Reproducibility

By following this directory structure and executing the provided notebooks in order,
the dataset preparation pipeline can be **reconstructed** by users who have
legitimate access to the MELD dataset.

This design ensures:
- Transparency
- Reproducibility
- Compliance with dataset licensing terms
