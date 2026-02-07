# -*- coding: utf-8 -*-
"""
gui.py

Unified graphical user interface for speech emotion recognition experiments.

Supports:
- Classical machine learning models
- Deep learning architectures
- State-of-the-art transformer-based models

This GUI is used for inference and qualitative evaluation only.


Created on Sat Jul 26 16:06:56 2025
@author: tunca
"""

import importlib
import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# =========================
# GLOBAL WARNINGS
# =========================
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="X has feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# =========================
# PYQT5 IMPORTS
# =========================
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit
)

from PyQt5.QtGui import QFont

# =========================
# LOCAL IMPORTS
# =========================
from gui.extract_advanced_ml_features import extract_advanced_ml_features


# =========================
# BASE DIRECTORY (repo root)
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

GUI_DIR = os.path.dirname(__file__)
if GUI_DIR not in sys.path:
    sys.path.insert(0, GUI_DIR)


# =========================
# SOFT VOTING WRAPPER
# =========================
# NOTE:
# This wrapper is kept for backward compatibility with
# soft-voting experiments during training.
from sklearn.base import BaseEstimator, ClassifierMixin

class SoftVotingWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, models, scalers, label_encoder):
        self.models = models
        self.scalers = scalers
        self.label_encoder = label_encoder

    def predict_proba(self, X):
        probas = []
        for key in self.models:
            X_scaled = self.scalers[key].transform(X)
            proba = self.models[key].predict_proba(X_scaled)
            probas.append(proba)
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        pred_num = np.argmax(avg_proba, axis=1)
        # Here we only return class indexes.
        return pred_num


# =========================
# TEST LABELS (CSV)
# =========================
TEST_CSV = os.path.join(BASE_DIR, "archive_meld", "joey_text_final", "test_sent_emo_joey_preprocess_with_filename.csv")

file2label = {}

try:
    df_test = pd.read_csv(TEST_CSV)
    file2label = dict(zip(df_test["FileName"], df_test["Emotion"]))
except Exception:
    print("⚠️ Test CSV not found. True labels will be unavailable.")

# =========================
# MODEL CONFIG
# =========================
model_types = ["svm_1", "rf_1", "xgb_1", "knn_1", "s-mlp_1", "gb_1", "bagging_knn_1", "voting_1", "soft_voting_1"]

model_type_prefix_map = {
    'svm_1': 'svm',
    'rf_1': 'rf',
    'xgb_1': 'xgb',
    'knn_1': 'knn',
    's-mlp_1': 'mlp',  
    'gb_1': 'gb',
    'bagging_knn_1': 'bagging_knn',
    'voting_1': 'voting',
    'soft_voting_1': 'soft_voting'
}

model_names = [
    "meld_features_ml_1_balanced_cleaned", 
    "meld_features_ml_1_clean_balanced", 
    "meld_features_selected_boruta_1_balanced",
    "meld_features_selected_kbest_1_balanced", 
    "meld_features_selected_rf_1_balanced", 
    "meld_features_selected_rfe_1_balanced",
    "meld_features_selected_sffs_1_balanced", 
    "meld_features_selected_autoencoder_1_balanced",
    "meld_features_selected_boruta_rf_1_balanced", 
    "meld_features_selected_boruta_autoencoder_1_balanced",
    "meld_features_selected_boruta_autoencoder_rf_1_balanced"
]


# =========================
# MAIN GUI CLASS
# =========================
class EmotionPredictorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 Emotion Prediction Interface")
        self.setGeometry(200, 200, 900, 700)

        self.selected_file = None

        # === Interface elements ===
        self.file_label = QLabel("🔘 Selected File: -")

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 11))
        self.result_text.setStyleSheet("""
            background-color: #f5f5f5;
            color: #222;
            padding: 10px;
            border-radius: 10px;
        """)

        self.select_button = QPushButton("📂 Select Audio File (.wav)")
        self.select_button.clicked.connect(self.select_file)
        
        # 1 ok
        self.genel_model_dropdown = QComboBox()
        self.genel_model_dropdown.addItems(["ML", "DL", "SOTA"])
        self.genel_model_dropdown.currentIndexChanged.connect(self.update_model_options)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(model_types)

        self.feature_dropdown = QComboBox()
        self.feature_dropdown.addItems(model_names)
        
        # === SOTA model selection ===
        self.sota_model_dropdown = QComboBox()
        self.sota_model_dropdown.addItems([
            "wavlm_mlp", "wavlm_mlp2", 
            "wav2vec2_mlp", "wav2vec2_mlp2", 
            "hubert_mlp", "hubert_mlp2",
            "wavlm_end2end", "wav2vec2_end2end", "hubert_end2end"
        ])
        self.sota_model_dropdown.setVisible(False)


        # === DL model selection ===
        self.dl_model_dropdown = QComboBox()
        self.dl_model_dropdown.addItems([
            "cnn_mel", "cnn_mfcc", 
            "cnn_rgb", "cnn_multi", 
            "resnet18_mel", "efficientnetb0_rgb",
            "bilstm", "keras_bilstm", "pytorch_mlp",
            "keras_mlp", "cnn_bilstm"
        ])
        self.dl_model_dropdown.setVisible(False)
        
        self.dl_model_dropdown.currentIndexChanged.connect(self.update_model_options)

        # === TORCH-MLP feature set dropdown ===
        self.label_dl_feature_set = QLabel("Feature Set:")
        self.dl_feature_dropdown = QComboBox()
        self.dl_feature_dropdown.addItems(model_names)
        self.label_dl_feature_set.setVisible(False)
        self.dl_feature_dropdown.setVisible(False)


        self.predict_button = QPushButton("🔮 Make Prediction")
        self.predict_button.clicked.connect(self.predict_emotion)

        # === Layout ===
        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.file_label)
        
        # 2
        layout.addWidget(QLabel("General Model Type:"))
        layout.addWidget(self.genel_model_dropdown)

        # 3
        self.label_model_type = QLabel("ML Model Type:")
        self.label_feature_set = QLabel("Feature Set:")
        
        ml_layout  = QHBoxLayout()
        ml_layout .addWidget(self.label_model_type)
        ml_layout .addWidget(self.model_dropdown)
        ml_layout .addWidget(self.label_feature_set)
        ml_layout .addWidget(self.feature_dropdown)
        layout.addLayout(ml_layout )
        
        self.label_sota_model = QLabel("SOTA Model Type:")
        layout.addWidget(self.label_sota_model)
        layout.addWidget(self.sota_model_dropdown)
        
        self.label_dl_model = QLabel("DL Model Type:")
        layout.addWidget(self.label_dl_model)
        layout.addWidget(self.dl_model_dropdown)
        
        layout.addWidget(self.label_dl_feature_set)
        layout.addWidget(self.dl_feature_dropdown)


        layout.addWidget(self.predict_button)
        layout.addWidget(QLabel("📋 Results:"))
        layout.addWidget(self.result_text)

        self.setLayout(layout)
        
        # ✅ Set visibility settings correctly when the GUI opens.
        self.update_model_options()

        
    # 4
    # =========================
    # UI VISIBILITY LOGIC
    # =========================
    def update_model_options(self):
        selected_type = self.genel_model_dropdown.currentText()
    
        is_ml = selected_type == "ML"
        is_sota = selected_type == "SOTA"
        is_dl = selected_type =="DL"
    
        # ML visibility
        self.label_model_type.setVisible(is_ml)
        self.model_dropdown.setVisible(is_ml)
        self.label_feature_set.setVisible(is_ml)
        self.feature_dropdown.setVisible(is_ml)
    
        # SOTA visibility
        self.label_sota_model.setVisible(is_sota)
        self.sota_model_dropdown.setVisible(is_sota)
        
        # DL visibility
        self.label_dl_model.setVisible(is_dl)
        self.dl_model_dropdown.setVisible(is_dl)

        # If TORCH-MLP is selected: activate the feature set dropdown.
        selected_dl_model = self.dl_model_dropdown.currentText()
        is_torch_mlp = is_dl and selected_dl_model == "pytorch_mlp"
        is_keras_mlp = is_dl and selected_dl_model == "keras_mlp"
        self.label_dl_feature_set.setVisible(is_torch_mlp or is_keras_mlp)
        self.dl_feature_dropdown.setVisible(is_torch_mlp or is_keras_mlp)

    
    # =========================
    # FILE SELECTION
    # =========================
    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav)")
        if fname:
            self.selected_file = fname
            self.file_label.setText(f"🔘 Selected file: {os.path.basename(fname)}")
            self.result_text.clear()
            

    # =========================
    # DISPLAY RESULT
    # =========================
    def display_result(self, file_name, true_label, pred_label, match, probs, classes):
        self.result_text.clear()
    
        is_correct = (pred_label == true_label)
        bg_color = "#d4edda" if is_correct else "#f8d7da"  # green / red
    
        emoji_dict = {
            "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨",
            "disgust": "🤢", "surprise": "😲", "neutral": "😐"
        }
        emoji = emoji_dict.get(pred_label, "❓")
    
        html = f"""
        <!-- First div: File and prediction information -->
        <div style="
            background-color:{bg_color};
            padding:15px;
            border-radius:12px;
            margin-bottom: 15px;
            border: 2px solid black;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        ">
            <b>📄 File:</b> {file_name}<br>
            <b>🎯 True Label:</b> {true_label}<br>
            <b>🔮 Predicted Label:</b> <span style="font-size: 16px;">{pred_label} {emoji}</span><br>
            <b>✅ Is it correct?:</b> {"<b>YES ✅</b>" if is_correct else "<b>NO ❌</b>"}
        </div>
    
        <!-- Second div: Probabilities table -->
        <div style="
            background-color:#f9f9f9;
            padding:15px;
            border-radius:12px;
            border: 2px solid black;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        ">
            <b><br><br>📊 Olasılıklar:</b>
            <table style="
                border-collapse: collapse;
                font-family: Consolas, monospace;
                font-size: 11px;
                width: 100%;
            ">
                <thead>
                    <tr>
                        <th style="text-align:left; padding:4px 10px; border-bottom: 1px solid #999; border: 1px solid #ccc;">Emotion</th>
                        <th style="text-align:right; padding:4px 10px; border-bottom: 1px solid #999; border: 1px solid #ccc;">Probability</th>
                    </tr>
                </thead>
                <tbody>
        """
    
        for c, p in zip(classes, probs):
            html += f"""
                <tr>
                    <td style="padding: 4px 10px; border: 1px solid #ccc;">{c}</td>
                    <td style="padding: 4px 10px; text-align: right; border: 1px solid #ccc;">{p:.5f}</td>
                </tr>
            """
    
        html += """
                </tbody>
            </table>
        </div>
        """
    
        self.result_text.setHtml(html)

    
    # =========================
    # PREDICTION DISPATCHER
    # =========================
    def predict_emotion(self):
        if not self.selected_file:
            self.result_text.setPlainText("⚠️ Please select an audio file first.")
            return
        
        # 5
        genel_model = self.genel_model_dropdown.currentText()
        
        # 6
        if genel_model == "ML":
            model_type = self.model_dropdown.currentText()
            model_prefix = model_type_prefix_map.get(model_type)
            feature_set = self.feature_dropdown.currentText()
        
            self.result_text.setPlainText("📡 Feature extraction is being performed....")
        
            try:
                feats_dict = extract_advanced_ml_features(self.selected_file)
                if feats_dict is None:
                    self.result_text.setPlainText("❌ The feature could not be extracted.")
                    return
                features_df = pd.DataFrame([feats_dict]).fillna(0)
            except Exception as e:
                self.result_text.setPlainText(f"❌ Feature extraction error: {e}")
                return
        
            try:
                # === Autoencoder-based feature set control ===
                if "autoencoder" in feature_set:
                    if "boruta" in feature_set:
                        feat_base = os.path.join(BASE_DIR, "features_and_models", "balanced_1")
                        ae_base = os.path.join(BASE_DIR, "features_and_models", "autoencoders")
                        boruta_feats = pd.read_csv(os.path.join(feat_base, "meld_features_selected_boruta_1_balanced.csv")).drop(columns=["label"]).columns.tolist()
                        scaler_ae = joblib.load(os.path.join(ae_base, "scaler_boruta_autoencoder_128.pkl"))
                        encoder_model = torch.nn.Sequential(
                            torch.nn.Linear(len(boruta_feats), 512),
                            torch.nn.ReLU(),
                            torch.nn.Linear(512, 128)
                        )
                        encoder_model.load_state_dict(torch.load(os.path.join(ae_base, "encoder_boruta_autoencoder_128.pth")))
                        encoder_model.eval()
                        encoder_model = encoder_model.to("cpu")
                        X_scaled = scaler_ae.transform(features_df[boruta_feats])
                        with torch.no_grad():
                            z = encoder_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
                        z_df = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])
        
                        if "rf" in feature_set:
                            rf_feats = joblib.load(os.path.join(ae_base, "top_100_rf_boruta_autoencoder_features.pkl"))
                            X_input = z_df[rf_feats]
                        else:
                            X_input = z_df
                    else:
                        all_feats = features_df.columns.tolist()
                        ae_base = os.path.join(BASE_DIR, "features_and_models", "autoencoders")
                        scaler_ae = joblib.load(os.path.join(ae_base, "scaler_autoencoder_128.pkl"))
                        encoder_model = torch.nn.Sequential(
                            torch.nn.Linear(len(all_feats), 512),
                            torch.nn.ReLU(),
                            torch.nn.Linear(512, 128)
                        )
                        encoder_model.load_state_dict(torch.load(os.path.join(ae_base, "encoder_autoencoder_128.pth")))
                        encoder_model.eval()
                        encoder_model = encoder_model.to("cpu")
                        X_scaled = scaler_ae.transform(features_df[all_feats])
                        with torch.no_grad():
                            z = encoder_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
                        X_input = pd.DataFrame(z, columns=[f"ae_{i}" for i in range(z.shape[1])])
                else:
                    feat_base = os.path.join(BASE_DIR, "features_and_models", "balanced_1")
                    df_feat = pd.read_csv(os.path.join(feat_base, f"{feature_set}.csv"))
                    feat_cols = df_feat.drop(columns=["label"]).columns.tolist()
                    X_input = features_df[feat_cols]
        
                # === Load model ===
                model_base = os.path.join(BASE_DIR, "features_and_models", "ml_models", model_type)
                model_path = os.path.join(model_base, f"{model_prefix}_best_model_{feature_set}.pkl")

                model_obj = joblib.load(model_path)
        
                if model_type == "soft_voting_1":
                    model = model_obj["model"]
                    label_encoder = model.label_encoder
                    self.result_text.append(f"Model tipi: {type(model)}")
                    probs = model.predict_proba(X_input)[0]
                    pred_label = label_encoder.inverse_transform([np.argmax(probs)])[0]
                else:
                    model = model_obj
                    scaler = joblib.load(os.path.join(model_base, f"scaler_{feature_set}.pkl"))
                    label_encoder = joblib.load(os.path.join(model_base, f"label_encoder_{feature_set}.pkl"))
                    X_scaled = scaler.transform(X_input)
        
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_scaled)[0]
                        pred_label = label_encoder.inverse_transform([np.argmax(probs)])[0]
                    else:
                        pred_num = model.predict(X_scaled)[0]
                        pred_label = label_encoder.inverse_transform([pred_num])[0]
                        probs = [1.0 if i == pred_num else 0.0 for i in range(len(label_encoder.classes_))]
        
            except Exception as e:
                self.result_text.setPlainText(f"❌ Model prediction error: {e}")
                return
            
        
            # === Print results ===
            filename = os.path.basename(self.selected_file)
            true_label = file2label.get(filename, "Unknown")
            match = (pred_label == true_label)
            
            # === Display result ===
            self.display_result(
                filename,
                true_label,
                pred_label,
                match,
                probs,
                label_encoder.classes_
            )
            
        elif genel_model == "SOTA":
            print("🧠 GUI: Initializing SOTA model inference...")
            selected_sota_model = self.sota_model_dropdown.currentText()
            print(f"📦 Selected model: {selected_sota_model}")
            module_name = f"gui.predict_{selected_sota_model}"
        
            self.result_text.setPlainText(f"📡 Executing selected SOTA model: {selected_sota_model}")
        
            try:
                sota_module = importlib.import_module(module_name)
                result = sota_module.predict_emotion(self.selected_file)
        
                pred_label = result["pred_label"]
                probs = result["probs"]
                label_encoder = result["label_encoder"]
        
                filename = os.path.basename(self.selected_file)
                true_label = file2label.get(filename, "Unknown")
                match = (pred_label == true_label)
        
                self.display_result(
                    filename,
                    true_label,
                    pred_label,
                    match,
                    probs,
                    label_encoder.classes_ if hasattr(label_encoder, "classes_") else label_encoder
                )
        
            except Exception as e:
                self.result_text.setPlainText(f"❌ SOTA model error: {e}")
                
                
        elif genel_model == "DL":
            print("🧠 GUI: Initializing DL model inference...")
            selected_dl_model = self.dl_model_dropdown.currentText()
            print(f"📦 Selected model: {selected_dl_model}")
            module_name = f"gui.predict_{selected_dl_model}"
            self.result_text.setPlainText(f"📡 Executing selected DL model: {selected_dl_model}")
        
            try:
                if selected_dl_model == "pytorch_mlp":
                    selected_feature_set = self.dl_feature_dropdown.currentText()
                    from gui.predict_pytorch_mlp import predict_emotion as torch_mlp_predict
                    result = torch_mlp_predict(self.selected_file, selected_feature_set)
                    
                elif selected_dl_model == "keras_mlp":
                    selected_feature_set = self.dl_feature_dropdown.currentText()
                    from gui.predict_keras_mlp import predict_emotion as keras_mlp_predict
                    result = keras_mlp_predict(self.selected_file, selected_feature_set)
                    
                else:
                    dl_module = importlib.import_module(module_name)
                    result = dl_module.predict_emotion(self.selected_file)
        
                pred_label = result["pred_label"]
                probs = result["probs"]
                label_encoder = result["label_encoder"]
        
                filename = os.path.basename(self.selected_file)
                true_label = file2label.get(filename, "Unknown")
                match = (pred_label == true_label)
        
                self.display_result(
                    filename,
                    true_label,
                    pred_label,
                    match,
                    probs,
                    label_encoder if isinstance(label_encoder, (list, np.ndarray)) else label_encoder.classes_
                )
        
            except Exception as e:
                self.result_text.setPlainText(f"❌ DL model error: {e}")

    

if __name__ == "__main__":
    # === OPTIONAL HUGGINGFACE LOGIN (ENV VAR BASED) ===
    from huggingface_hub import login
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    app = QApplication(sys.argv)

    gui = EmotionPredictorGUI()
    gui.show()
    sys.exit(app.exec_())
