# -*- coding: utf-8 -*-
"""
predict_bilstm.py

Inference module for BiLSTM-based speech emotion recognition.
This script loads a pre-trained BiLSTM model and predicts emotions
from a given WAV audio file.

Used by the GUI layer.


Created on Sun Jul 27 11:23:43 2025
@author: tunca
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import joblib
import os

# Base project directory (robust against src/ relocation)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === 1. BiLSTM MODEL DEFINITION ===
class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim=120, hidden_dim1=128, hidden_dim2=64, num_classes=7):
        super(BiLSTMEmotionClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.pool = nn.AdaptiveMaxPool1d(1)  # Global Max Pooling
        self.fc1 = nn.Linear(hidden_dim2 * 2, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out.permute(0, 2, 1)  # [B, F, T]
        out = self.pool(out).squeeze(-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        return self.out(out)

# === 2. FEATURE EXTRACTION (MFCC + delta + delta-delta) ===
SR = 16000
N_MFCC = 40
MAX_FRAMES = 200

def extract_features(wav_path):
    """
    Extract MFCC-based features from an audio file.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (MAX_FRAMES, 120).
    """
    y, sr = librosa.load(wav_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # [T, 120]

    # Pad or truncate to fixed length
    if features.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:MAX_FRAMES, :]
    return features

# === 3. PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using a trained BiLSTM model.

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 3.1 Load the model, scaler, and label encoder
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/bilstm/bilstm_best_model.pth")
    scaler_path = os.path.join(BASE_DIR, "features_and_models/dl_features/bilstm/scaler_bilstm.pkl")
    label_encoder_path = os.path.join(BASE_DIR, "features_and_models/dl_features/bilstm/label_encoder_bilstm.pkl")

    scaler = joblib.load(scaler_path)
    le = joblib.load(label_encoder_path)
    class_names = le.classes_

    model = BiLSTMEmotionClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 3.2 Feature extraction
    features = extract_features(wav_path)
    features_scaled = scaler.transform(features).astype(np.float32)
    features_tensor = torch.tensor(features_scaled).unsqueeze(0).to(device)  # [1, T, F]

    # === 3.3 Predict
    with torch.no_grad():
        output = model(features_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()
        pred_index = probs.argmax()
        pred_label = class_names[pred_index]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
