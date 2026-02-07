# -*- coding: utf-8 -*-
"""
predict_pytorch_mlp.py

Inference module for PyTorch-based MLP speech emotion recognition.
Advanced handcrafted audio features are extracted and classified
using a deep multilayer perceptron trained in PyTorch.

Supports feature-selection and autoencoder-based pipelines.

Used by the GUI layer.


Created on Sun Jul 27 12:09:56 2025
@author: tunca
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings

from extract_advanced_ml_features import extract_advanced_ml_features

warnings.filterwarnings("ignore")

# === BASE PROJECT DIRECTORY (robust for src/ structure) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === PATH DEFINITIONS ===
FEATURE_BASE = os.path.join(BASE_DIR, "features_and_models/balanced_1")
MODEL_BASE = os.path.join(BASE_DIR, "features_and_models/dl_models/torch_mlp")
AE_MODEL_BASE = os.path.join(BASE_DIR, "features_and_models/autoencoders")

# === MODEL DEFINITION ===
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path, selected_model_name):
    """
    Predict emotion label from a WAV file using a PyTorch MLP model.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.
    selected_model_name : str
        Feature/model configuration name selected from GUI.

    Returns
    -------
    dict
        Dictionary containing:
        - pred_label: predicted emotion label
        - probs: class probabilities
        - label_encoder: class names
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === FEATURE EXTRACTION ===
    feats_dict = extract_advanced_ml_features(wav_path)
    if feats_dict is None:
        return {"pred_label": "ERROR", "probs": [], "label_encoder": []}

    feats_df = pd.DataFrame([feats_dict]).fillna(0)

    try:
        # === NON-AUTOENCODER PIPELINE ===
        if "autoencoder" not in selected_model_name:
            feat_path = f"{FEATURE_BASE}/{selected_model_name}.csv"
            selected_feats = pd.read_csv(feat_path).drop(columns=["label"]).columns.tolist()
            X_input = feats_df[selected_feats].values

        # === AUTOENCODER PIPELINE ===
        else:
            if "boruta" in selected_model_name:
                boruta_feats = pd.read_csv(f"{FEATURE_BASE}/meld_features_selected_boruta_1_balanced.csv").drop(columns=["label"]).columns.tolist()
                scaler_ae = joblib.load(f"{AE_MODEL_BASE}/scaler_boruta_autoencoder_128.pkl")
                encoder = nn.Sequential(nn.Linear(len(boruta_feats), 512), nn.ReLU(), nn.Linear(512, 128))
                encoder.load_state_dict(torch.load(f"{AE_MODEL_BASE}/encoder_boruta_autoencoder_128.pth", map_location=device))
                encoder.eval()
                X_scaled = scaler_ae.transform(feats_df[boruta_feats])
                z = encoder(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy()
                df_z = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])

                if "rf" in selected_model_name:
                    rf_feats = joblib.load(f"{AE_MODEL_BASE}/top_100_rf_boruta_autoencoder_features.pkl")
                    X_input = df_z[rf_feats]
                else:
                    X_input = df_z
            else:
                all_feats = feats_df.columns.tolist()
                scaler_ae = joblib.load(f"{AE_MODEL_BASE}/scaler_autoencoder_128.pkl")
                encoder = nn.Sequential(nn.Linear(len(all_feats), 512), nn.ReLU(), nn.Linear(512, 128))
                encoder.load_state_dict(torch.load(f"{AE_MODEL_BASE}/encoder_autoencoder_128.pth", map_location=device))
                encoder.eval()
                X_scaled = scaler_ae.transform(feats_df[all_feats])
                z = encoder(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy()
                X_input = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])

        # === LOAD CLASSIFIER COMPONENTS ===
        model_dir = os.path.join(MODEL_BASE, selected_model_name)

        # === Model, scaler, encoder yükle ===
        model_path = os.path.join(model_dir, "best_model.pth")
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        
        class_names = label_encoder.classes_
        num_classes = len(class_names)

        X_final = scaler.transform(X_input)
        input_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)

        # === MODEL INFERENCE ===
        input_dim = input_tensor.shape[1]
        model = DeepMLP(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred_idx = np.argmax(probs)
            pred_label = class_names[pred_idx]

        return {
            "pred_label": pred_label,
            "probs": probs,
            "label_encoder": class_names
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"pred_label": "ERROR", "probs": [], "label_encoder": []}
