# -*- coding: utf-8 -*-
"""
predict_cnn_multi.py

Inference module for multi-input CNN-based speech emotion recognition.
The same audio signal is converted into both Mel spectrogram and MFCC
spectrogram images. These representations are processed by two parallel
CNN branches whose features are concatenated for final classification.

Used by the GUI layer.


Created on Sun Jul 27 02:55:43 2025
@author: tunca
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

# Base project directory (robust for src/ structure)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === 1. MEL & MFCC SPECTROGRAM IMAGE GENERATION ===
def generate_spectrogram_image(wav_path, mode="mel", target_sr=16000, min_duration=1.5):
    """
    Generate a Mel or MFCC spectrogram image from an audio file.

    Parameters
    ----------
    wav_path : str
        Path to WAV audio file.
    mode : str
        'mel' or 'mfcc'.
    target_sr : int
        Target sampling rate.
    min_duration : float
        Minimum duration in seconds.

    Returns
    -------
    str
        Path to the generated temporary image file.
    """
    y, sr = librosa.load(wav_path, sr=target_sr)
    target_len = int(target_sr * min_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    if mode == "mel":
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
    elif mode == "mfcc":
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        spec_db = librosa.power_to_db(mfcc ** 2)
    else:
        raise ValueError("mode must be 'mel' or 'mfcc'")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.axis('off')
        librosa.display.specshow(spec_db, sr=sr, cmap='gray')
        plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0)
        plt.close()
        return tmp_img.name

# === 2. MULTI-INPUT CNN MODEL ===
class MultiCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiCNN, self).__init__()

        def cnn_branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.4)
            )

        self.mel_cnn = cnn_branch()
        self.mfcc_cnn = cnn_branch()

        self.fc = nn.Sequential(
            nn.Linear(2 * 64 * 16 * 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel_x, mfcc_x):
        mel_feat = self.mel_cnn(mel_x)
        mfcc_feat = self.mfcc_cnn(mfcc_x)
        mel_flat = torch.flatten(mel_feat, start_dim=1)
        mfcc_flat = torch.flatten(mfcc_feat, start_dim=1)
        combined = torch.cat((mel_flat, mfcc_flat), dim=1)
        return self.fc(combined)

# === 3. PREDICTION FUNCTION (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using a multi-input CNN
    trained on Mel and MFCC spectrogram images.

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

    # Generate spectrogram images
    mel_img_path = generate_spectrogram_image(wav_path, mode="mel")
    mfcc_img_path = generate_spectrogram_image(wav_path, mode="mfcc")

    # Image transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mel_img = transform(Image.open(mel_img_path).convert("RGB")).unsqueeze(0).to(device)
    mfcc_img = transform(Image.open(mfcc_img_path).convert("RGB")).unsqueeze(0).to(device)

    os.remove(mel_img_path)
    os.remove(mfcc_img_path)

    # Load model
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/spectrogram_images_multi_mel_mfcc_model/multi_input_cnn_best.pt")
    model = MultiCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(mel_img, mfcc_img)
        probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()
        pred_class = np.argmax(probs)
        pred_label = class_names[pred_class]

    return {
        "pred_label": pred_label,
        "probs": probs,
        "label_encoder": class_names
    }
