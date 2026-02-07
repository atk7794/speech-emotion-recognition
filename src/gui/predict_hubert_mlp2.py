# -*- coding: utf-8 -*-
"""
predict_hubert_mlp2.py

Speech emotion recognition using HuBERT-base embeddings
classified with a PyTorch MLP model.

Used by the GUI layer.


Created on Sat Jul 26 22:40:01 2025
@author: tunca
"""


import torch
import torchaudio
import numpy as np
import joblib
import pandas as pd
import os
from transformers import AutoFeatureExtractor, AutoModel
from huggingface_hub import login

# === OPTIONAL HUGGINGFACE LOGIN (ENV VAR BASED) ===
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# === LOGGING & WARNINGS ===
from transformers.utils import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "none"
warnings.filterwarnings("ignore", category=UserWarning)


# === BASE PROJECT DIRECTORY ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === MODEL PATHS & CONFIGURATION ===
model_dir = os.path.join(BASE_DIR, "features_and_models/sota_models_pretrained/hubert_model2")
model_name = "facebook/hubert-base-ls960"

# === DEVICE CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD FEATURE EXTRACTOR & HUBERT MODEL ===
print("📥 Feature extractor loading...")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

print("📥 HuBert model loading...")
hubert = AutoModel.from_pretrained(model_name).to(device)
hubert.eval()

# === LOAD SCALER & LABEL ENCODER ===
print("📦 Scaler and LabelEncoder loading...")
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))


# === MLP CLASSIFIER DEFINITION ===
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# === LOAD MLP MODEL ===
input_dim = 768 
num_classes = len(label_encoder.classes_)if hasattr(label_encoder, "classes_") else len(label_encoder.classes)

print("🏗️ The MLP model structure is being established...")
mlp_model = MLPClassifier(input_dim, num_classes).to(device)
mlp_model.load_state_dict(torch.load(os.path.join(model_dir, "hubert_mlp_model.pt"), map_location=device))
mlp_model.eval()

print("✅ The model and embedding structures were successfully loaded.")


# === PREDICTION FUNCTION ===
def predict_emotion(file_path):
    """
    Predict emotion label from a WAV file using
    HuBERT-base embeddings + MLP classifier.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.

    Returns
    -------
    dict
        Prediction results.
    """
    
    print(f"\n🎧 The prediction has been initiated: {file_path}")
    waveform, sr = torchaudio.load(file_path)
    
    print("📥 Audio loaded.")
    
    # Resample if necessary
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        print("🔁 Resampling was done.")

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("🎚️ It was converted to a mono channel.")

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    print("📊 Feature extraction is complete.")

    with torch.no_grad():
        outputs = hubert(**inputs.to(device))
        print("🧠 HuBert forward pass completed.")
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

    embedding_scaled = scaler.transform([embedding])
    embedding_tensor = torch.tensor(embedding_scaled, dtype=torch.float32).to(device)
    print("📐 Embedding has been scaled.")

    with torch.no_grad():
        logits = mlp_model(embedding_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        
    print(f"🔮 Prediction: {predicted_label} — Probabilities: {probs}")

    return {
        "pred_label": predicted_label,
        "probs": probs,
        "label_encoder": label_encoder
    }

