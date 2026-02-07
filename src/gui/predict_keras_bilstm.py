# -*- coding: utf-8 -*-
"""
predict_keras_bilstm.py

Inference module for Keras-based BiLSTM speech emotion recognition.
MFCC + delta + delta² features are extracted from audio signals and
classified using a trained Bidirectional LSTM model.

Used by the GUI layer.


Created on Sun Jul 27 11:48:22 2025
@author: tunca
"""

import numpy as np
import librosa
import joblib
import os
from tensorflow.keras.models import load_model

# === BASE PROJECT DIRECTORY (robust for src/ structure) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === GLOBAL SETTINGS ===
SR = 16000
N_MFCC = 40
MAX_LEN = 200

# === FEATURE EXTRACTION (MFCC + delta + delta-delta) ===
def extract_features(wav_path):
    """
    Extract MFCC-based sequential features for BiLSTM input.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.

    Returns
    -------
    np.ndarray
        Feature tensor of shape [MAX_LEN, 120].
    """
    y, sr = librosa.load(wav_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # [T, 120]

    if features.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:MAX_LEN, :]
    return features.astype(np.float32)

# === PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using a Keras BiLSTM model.

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
    # === PATH DEFINITIONS ===
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/keras_bilstm/bilstm_keras_mfcc120_model.h5")
    scaler_path = os.path.join(BASE_DIR, "features_and_models/dl_features/keras_bilstm/scaler.pkl")
    label_encoder_path = os.path.join(BASE_DIR, "features_and_models/dl_features/keras_bilstm/label_encoder.pkl")

    # === LOAD MODEL COMPONENTS ===
    scaler = joblib.load(scaler_path)
    le = joblib.load(label_encoder_path)
    class_names = le.classes_
    
    model = load_model(model_path)

    # === FEATURE EXTRACTION ===
    features = extract_features(wav_path)
    features_scaled = scaler.transform(features)
    features_scaled = np.expand_dims(features_scaled, axis=0)  # [1, T, F]

    # === MODEL INFERENCE ===
    probs = model.predict(features_scaled)[0]
    pred_idx = np.argmax(probs)
    pred_label = class_names[pred_idx]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
