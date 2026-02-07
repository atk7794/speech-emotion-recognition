# -*- coding: utf-8 -*-
"""
predict_efficientnetb0_rgb.py

Inference module for EfficientNet-B0 based speech emotion recognition.
MFCC, delta, and delta-delta features are mapped to RGB channels and
classified using an EfficientNet-B0 architecture.

Used by the GUI layer.


Created on Sun Jul 27 11:13:17 2025
@author: tunca
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import librosa
import numpy as np
import os
import tempfile

# Base project directory (robust for src/ structure)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === 1. NORMALIZATION & RGB IMAGE CREATION ===
def normalize(x):
    """
    Normalize a feature matrix to 0–255 range.

    Parameters
    ----------
    x : np.ndarray
        Input feature matrix.

    Returns
    -------
    np.ndarray
        Normalized uint8 matrix.
    """
    x = np.nan_to_num(x)
    x_min, x_max = x.min(), x.max()
    return np.clip(255 * (x - x_min) / (x_max - x_min + 1e-8), 0, 255).astype(np.uint8)

def create_rgb_image_efficientnet(wav_path, target_size=(224, 224)):
    """
    Create an RGB image from MFCC, delta, and delta-delta features.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.
    target_size : tuple
        Output image size (default: 224x224 for EfficientNet).

    Returns
    -------
    str
        Path to the temporary RGB image file.
    """
    y, sr = librosa.load(wav_path, sr=16000)

    # Feature extraction: MFCC + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Align time dimension
    min_len = min(mfcc.shape[1], delta.shape[1], delta2.shape[1])
    mfcc, delta, delta2 = mfcc[:, :min_len], delta[:, :min_len], delta2[:, :min_len]

    r = normalize(mfcc)
    g = normalize(delta)
    b = normalize(delta2)

    # RGB fusion
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    img = img.resize(target_size, Image.BILINEAR)

    # Save temporarily (GUI-compatible approach)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img.save(tmp_img.name)
        return tmp_img.name

# === 2. PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using EfficientNet-B0.

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
    class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    # Create RGB image
    img_path = create_rgb_image_efficientnet(wav_path, target_size=(224, 224))
    image = Image.open(img_path).convert("RGB")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    os.remove(img_path)

    # Load EfficientNet-B0 model
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/spectrogram_images_rgb224_efficientnet/efficientnetb0_best.pt")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()
        pred_class = probs.argmax()
        pred_label = class_names[pred_class]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
