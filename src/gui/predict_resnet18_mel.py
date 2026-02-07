# -*- coding: utf-8 -*-
"""
predict_resnet18_mel.py

Inference module for ResNet-18 based speech emotion recognition.
Mel-spectrograms are converted into 224x224 RGB images and classified
using a fine-tuned ResNet-18 model.

Used by the GUI layer.


Created on Sun Jul 27 11:02:08 2025
@author: tunca
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

# Base project directory (robust for src/ structure)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === MEL-SPECTROGRAM RGB IMAGE CREATION (224x224 RGB) ===
def create_mel_image_rgb(wav_path, target_sr=16000, min_duration=1.5):
    """
    Create a 224x224 RGB image from a Mel-spectrogram.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.
    target_sr : int
        Target sampling rate.
    min_duration : float
        Minimum duration in seconds.

    Returns
    -------
    str
        Path to the temporary RGB image file.
    """
    y, sr = librosa.load(wav_path, sr=target_sr)
    target_len = int(target_sr * min_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to 0–255
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = (mel_norm * 255).astype(np.uint8)

    img = Image.fromarray(mel_img).resize((224, 224), Image.BILINEAR)
    img_rgb = Image.merge("RGB", (img, img, img))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img_rgb.save(tmp_img.name)
        return tmp_img.name

# === MRESNET-18 MODEL LOADING ===
class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def load_resnet18_model():
    """
    Load fine-tuned ResNet-18 model for emotion classification.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/spectrogram_images_mel224_resnet18/resnet18_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model

# === PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using ResNet-18.

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

    # Create Mel RGB image
    mel_img_path = create_mel_image_rgb(wav_path)
    image = Image.open(mel_img_path).convert("RGB")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    os.remove(mel_img_path)

    # Load model
    model = load_resnet18_model().to(device)
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
