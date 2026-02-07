# -*- coding: utf-8 -*-
"""
predict_cnn_bilstm.py

Inference module for CNN + BiLSTM based speech emotion recognition.
This script extracts Mel-spectrogram features from a WAV file and
predicts the corresponding emotion label using a pre-trained model.

Used by the GUI layer.


Created on Sun Jul 27 11:59:05 2025
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

# === 1. CONFIGURATION ===
SAMPLE_RATE = 16000
N_MELS = 128
MAX_LEN = 128

model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/cnn_bilstm/cnn_bilstm_best_model.pt")
scaler_path = os.path.join(BASE_DIR, "features_and_models/dl_features/cnn_bilstm_mel128/scaler.pkl")
label_encoder_path = os.path.join(BASE_DIR, "features_and_models/dl_features/cnn_bilstm_mel128/label_encoder.pkl")

# === 2. MEL FEATURE EXTRACTION ===
def extract_mel(file_path):
    """
    Extract fixed-length Mel-spectrogram features from an audio file.

    Parameters
    ----------
    file_path : str
        Path to WAV audio file.

    Returns
    -------
    np.ndarray
        Mel-spectrogram feature matrix of shape (MAX_LEN, N_MELS).
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel).T  # [T, 128]

    if mel_db.shape[0] < MAX_LEN:
        pad = np.zeros((MAX_LEN - mel_db.shape[0], N_MELS))
        mel_db = np.vstack([mel_db, pad])
    else:
        mel_db = mel_db[:MAX_LEN]
    return mel_db  # [128,128]

# === 3. MODEL CLASS ===
class SimpleCNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bilstm = nn.LSTM(
            input_size=32 * 32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)                        # [B, 32, 32, 32]
        x = x.permute(0, 2, 1, 3)              # [B, T=32, C=32, F=32]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, 32, 1024]
        lstm_out, _ = self.bilstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.classifier(last_step)
        return out

# === 4. PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using a trained CNN-BiLSTM model.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.

    Returns
    -------
    dict
        Dictionary containing prediction results:
        - pred_label: predicted emotion label
        - probs: class probabilities
        - label_encoder: class names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model, scaler, and label encoder
    le = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    class_names = le.classes_
    num_classes = len(class_names)

    model = SimpleCNNBiLSTM(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Feature extraction and transformations
    mel = extract_mel(wav_path)  # [128, 128]
    mel_flat = mel.reshape(1, -1)
    mel_scaled = scaler.transform(mel_flat)
    mel_scaled = mel_scaled.reshape(1, 1, 128, 128)  # [B, C, H, W]

    # Predict
    inputs = torch.tensor(mel_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
        pred_index = np.argmax(probs)
        pred_label = class_names[pred_index]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
