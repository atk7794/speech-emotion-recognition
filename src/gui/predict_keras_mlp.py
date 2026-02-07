# -*- coding: utf-8 -*-
"""
predict_keras_mlp.py

Inference module for classical ML & Keras-MLP based speech emotion recognition.
Advanced handcrafted audio features are extracted and classified using
pre-trained Keras MLP models, optionally combined with autoencoders.

Used by the GUI layer.


Created on Sun Jul 27 12:51:47 2025
@author: tunca
"""

import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import torch
import torch.nn as nn
import warnings
import logging
import absl.logging
from cryptography.utils import CryptographyDeprecationWarning

# === LOG & WARNING SUPPRESSION ===
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="X has feature names", category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')


from extract_advanced_ml_features import extract_advanced_ml_features

# === BASE PROJECT DIRECTORY (robust for src/ structure) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === PATH DEFINITIONS ===
FEATURE_BASE = os.path.join(BASE_DIR, "features_and_models/balanced_1")
MODEL_BASE = os.path.join(BASE_DIR, "features_and_models/dl_models/keras_mlp")
AE_MODEL_BASE = os.path.join(BASE_DIR, "features_and_models/autoencoders")

# === PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path, selected_model_name):
    """
    Predict emotion label from a WAV file using Keras MLP models.

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

    try:
        # === FEATURE EXTRACTION ===
        feat_dict = extract_advanced_ml_features(wav_path)
        if feat_dict is None:
            return {"pred_label": "ERROR", "probs": [], "label_encoder": []}

        features_df = pd.DataFrame([feat_dict]).fillna(0)

        # === FEATURE SELECTION / AUTOENCODER PIPELINE ===
        if "autoencoder" not in selected_model_name:
            feat_path = f"{FEATURE_BASE}/{selected_model_name}.csv"
            feats = pd.read_csv(feat_path).drop(columns=["label"]).columns.tolist()
            X_input = features_df[feats].values

        else:
            if "boruta" in selected_model_name:
                boruta_feats = pd.read_csv(f"{FEATURE_BASE}/meld_features_selected_boruta_1_balanced.csv").drop(columns=['label']).columns.tolist()
                scaler_ae = joblib.load(f"{AE_MODEL_BASE}/scaler_boruta_autoencoder_128.pkl")
                encoder_model = nn.Sequential(
                    nn.Linear(len(boruta_feats), 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)
                )
                encoder_model.load_state_dict(torch.load(f"{AE_MODEL_BASE}/encoder_boruta_autoencoder_128.pth", map_location="cpu"))
                encoder_model.eval()
                X_scaled = scaler_ae.transform(features_df[boruta_feats])
                with torch.no_grad():
                    z = encoder_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
                df_z = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])
                if "rf" in selected_model_name:
                    rf_feats = joblib.load(f"{AE_MODEL_BASE}/top_100_rf_boruta_autoencoder_features.pkl")
                    X_input = df_z[rf_feats]
                else:
                    X_input = df_z
            else:
                all_feats = features_df.columns.tolist()
                scaler_ae = joblib.load(f"{AE_MODEL_BASE}/scaler_autoencoder_128.pkl")
                encoder_model = nn.Sequential(
                    nn.Linear(len(all_feats), 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)
                )
                encoder_model.load_state_dict(torch.load(f"{AE_MODEL_BASE}/encoder_autoencoder_128.pth", map_location="cpu"))
                encoder_model.eval()
                X_scaled = scaler_ae.transform(features_df[all_feats])
                with torch.no_grad():
                    z = encoder_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
                X_input = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])

        # === LOAD MODEL & CLASSIFIER & SCALER ===
        model_dir = os.path.join(MODEL_BASE, selected_model_name)

        model_path = os.path.join(model_dir, "best_model.h5")
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

        X_final = scaler.transform(X_input)
        X_tensor = tf.convert_to_tensor(X_final, dtype=tf.float32)

        # === INFERENCE ===
        model = tf.keras.models.load_model(model_path)
        probs = model(X_tensor, training=False).numpy().squeeze()
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        return {
            "pred_label": pred_label,
            "probs": probs,
            "label_encoder": label_encoder.classes_
        } 

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"pred_label": "ERROR", "probs": [], "label_encoder": []}
