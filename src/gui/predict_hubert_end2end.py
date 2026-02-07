# -*- coding: utf-8 -*-
"""
predict_hubert_end2end.py

End-to-end speech emotion recognition using a fine-tuned HuBERT model.
Raw audio waveforms are directly processed without handcrafted feature
extraction.

Used by the GUI layer.


Created on Sat Jul 26 23:50:51 2025
@author: tunca
"""

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# === LOGGING & WARNINGS ===
from transformers.utils import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")


# === BASE PROJECT DIRECTORY (robust for src/ structure) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === MODEL DIRECTORY ===
model_dir = os.path.join(BASE_DIR, "features_and_models/sota_models_transformer/hubert_model")

# === DEVICE CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD FEATURE EXTRACTOR & MODEL (ONCE) ===
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
model = HubertForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

# === PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(file_path):
    """
    Predict emotion label from a WAV file using an end-to-end HuBERT model.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.

    Returns
    -------
    dict
        Dictionary containing:
        - pred_label: predicted emotion label
        - probs: class probabilities
        - label_encoder: class names
    """
    waveform, sr = torchaudio.load(file_path)

    # Resample if necessary
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Feature extraction
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000 * 16,
        return_attention_mask=True
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_id = np.argmax(probs)

    label_map = model.config.id2label
    pred_label = label_map[pred_id]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": list(label_map.values())
    }
