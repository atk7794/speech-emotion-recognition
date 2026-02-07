# -*- coding: utf-8 -*-
"""
predict_cnn_mel.py

Inference module for CNN-based speech emotion recognition using
Mel-spectrogram images. The audio signal is converted into a
Mel-spectrogram image and classified by a pre-trained CNN model.

Used by the GUI layer.


Created on Sun Jul 27 02:15:14 2025
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
import joblib

# Base project directory (robust for src/ structure)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# === Model Definition ===
class MelCNN(nn.Module):
    def __init__(self, num_classes):
        super(MelCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout_fc(x)
        return self.fc2(x)

# === Predicition Function (called by GUI) ===
def predict_emotion(wav_path):
    """
    Predict emotion label from a WAV file using a CNN trained on
    Mel-spectrogram images.

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

    # Generate temporary Mel-spectrogram image
    y, sr = librosa.load(wav_path, sr=16000)
    target_length = int(16000 * 1.5)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.axis('off')
        librosa.display.specshow(mel_db, sr=sr, cmap='gray', fmax=8000)
        plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0)
        plt.close()
        temp_img_path = tmp_img.name

    # Load and transform image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    with Image.open(temp_img_path) as image:
        input_tensor = transform(image).unsqueeze(0).to(device)

    os.remove(temp_img_path)  # Remove temporary image

    # Load model
    model_path = os.path.join(BASE_DIR, "features_and_models/dl_models/spectrogram_images_mel_model/mel_cnn_best.pt")
    model = MelCNN(num_classes=len(class_names)).to(device)
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
