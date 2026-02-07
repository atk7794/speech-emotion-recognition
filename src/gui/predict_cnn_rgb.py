# -*- coding: utf-8 -*-
"""
predict_cnn_rgb.py

Inference module for RGB-based CNN speech emotion recognition.
MFCC, delta, and delta-delta features are mapped to RGB channels
and processed by a convolutional neural network.

Used by the GUI layer.


Created on Sun Jul 27 02:45:35 2025
@author: tunca
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
import os
import tempfile

# Base project directory (robust for src/ structure)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === 1. NORMALIZATION & RGB FUSION ===
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

def create_rgb_image(wav_path, target_size=(128, 128)):
    """
    Create an RGB image from MFCC, delta, and delta-delta features.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.
    target_size : tuple
        Output image size.

    Returns
    -------
    PIL.Image
        RGB image representing fused audio features.
    """
    y, sr = librosa.load(wav_path, sr=16000)

    # Feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Align time dimension
    min_len = min(mfcc.shape[1], delta.shape[1], delta2.shape[1])
    mfcc, delta, delta2 = mfcc[:, :min_len], delta[:, :min_len], delta2[:, :min_len]

    # RGB channel mapping
    r = normalize(mfcc)
    g = normalize(delta)
    b = normalize(delta2)
    rgb = np.stack([r, g, b], axis=-1)

    img = Image.fromarray(rgb)
    img = img.resize(target_size, Image.BILINEAR)
    return img

# === 2. RGB CNN MODEL ===
class RGBCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGBCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# === 3. PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using an RGB CNN model.

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
    image = create_rgb_image(wav_path)

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/spectrogram_images_rgb_model/rgb_cnn_best.pt")
    model = RGBCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()
        pred_class = np.argmax(probs)
        pred_label = class_names[pred_class]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
