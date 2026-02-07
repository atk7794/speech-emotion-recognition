# -*- coding: utf-8 -*-
"""
extract_advanced_feature_extraction.py

Advanced handcrafted acoustic feature extraction module
for classical machine learning-based speech emotion recognition.

This module extracts a wide range of low-level and high-level
audio features from WAV files, including:
- Time-domain statistics
- MFCCs with delta and delta-delta coefficients
- Spectral features
- Pitch estimation
- Frequency-domain features (DFT)
- Wavelet packet energies
- Teager Energy Operator (TEO)
- Prosodic features via Praat/Parselmouth
- eGeMAPS features via openSMILE (optional)

The extracted features are designed for use with
traditional ML models (SVM, Random Forest, XGBoost, etc.)
and offline dataset preprocessing pipelines.


Created on Sat Jul 26 16:04:20 2025
@author: tunca
"""

import os
import pandas as pd
import numpy as np
import librosa
import scipy.stats
import pywt
import subprocess
import tempfile
import warnings
from scipy.fft import fft
from scipy.signal import lfilter
import parselmouth  # ✅ PRAAT Python API
from parselmouth.praat import call  # ✅ PRAAT function calls
from cryptography.utils import CryptographyDeprecationWarning


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="X has feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')


# FEATURE EXTRACTION
def extract_advanced_ml_features(file_path, sr=16000, n_mfcc=40, use_opensmile=True, opensmile_path=None):
    """
    Extracts advanced handcrafted acoustic features from a single audio file
    for classical machine learning-based emotion recognition.

    Features include:
    - Time-domain statistics (RMS, ZCR, spectral flatness)
    - MFCCs with delta and delta-delta coefficients
    - Spectral features (chroma, mel, contrast, tonnetz, centroid, bandwidth)
    - Pitch estimation (YIN)
    - Frequency-domain features (DFT)
    - Wavelet packet energy
    - Teager Energy Operator (TEO)
    - Prosodic features (jitter, shimmer, HNR, formants) via Praat/Parselmouth
    - eGeMAPS features via openSMILE (optional)

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    sr : int
        Target sampling rate.
    n_mfcc : int
        Number of MFCC coefficients.
    use_opensmile : bool
        Whether to extract eGeMAPS features using openSMILE.
    opensmile_path : str or None
        Path to the openSMILE installation directory.

    Returns
    -------
    dict or None
        Dictionary of extracted features, or None if extraction fails.
    """
    def summarize_safe(feat, name):
        try:
            feat = np.array(feat)
            if len(feat) == 0 or np.all(np.isnan(feat)) or np.all(feat == 0):
                return {}
            return {
                f"{name}_mean": float(np.nanmean(feat)),
                f"{name}_std": float(np.nanstd(feat)),
                f"{name}_min": float(np.nanmin(feat)),
                f"{name}_max": float(np.nanmax(feat)),
                f"{name}_skew": float(np.nan_to_num(scipy.stats.skew(feat), nan=0.0)),
                f"{name}_kurt": float(np.nan_to_num(scipy.stats.kurtosis(feat), nan=0.0))
            }
        except:
            return {}

    try:
        y, sr = librosa.load(file_path, sr=sr)
        N_FFT = 512
        MIN_LENGTH = N_FFT * 2
        y = np.pad(y, (0, max(0, MIN_LENGTH - len(y))), mode='constant')

        if len(y) < sr * 0.5:
            return None

        features = {}

        # Time-domain
        try:
            features.update(summarize_safe(librosa.feature.rms(y=y)[0], "rms"))
            features.update(summarize_safe(librosa.feature.zero_crossing_rate(y=y)[0], "zcr"))
            features.update(summarize_safe(librosa.feature.spectral_flatness(y=y, n_fft=N_FFT)[0], "flatness"))
        except: pass

        # MFCC + delta
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            for i in range(mfcc.shape[0]):
                features.update(summarize_safe(mfcc[i], f"mfcc_{i}"))
                features.update(summarize_safe(delta[i], f"delta_{i}"))
                features.update(summarize_safe(delta2[i], f"delta2_{i}"))
        except: pass

        # Spectral
        try:
            for mat, name in [
                (librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT), "chroma"),
                (librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), "mel"),
                (librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)), "mel_db"),
                (librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT), "contrast"),
                (librosa.feature.tonnetz(y=y, sr=sr), "tonnetz"),
            ]:
                for i in range(mat.shape[0]):
                    features.update(summarize_safe(mat[i], f"{name}_{i}"))

            features.update(summarize_safe(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT)[0], "centroid"))
            features.update(summarize_safe(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT)[0], "bandwidth"))
            features.update(summarize_safe(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT)[0], "rolloff"))
        except: pass

        # Pitch
        try:
            pitch_yin = librosa.yin(y, fmin=50, fmax=300, sr=sr)
            features.update(summarize_safe(pitch_yin, "pitch_yin"))
        except: pass

        # DFT
        try:
            dft = np.abs(fft(y))[:len(y)//2]
            features.update(summarize_safe(dft, "dft"))
        except: pass

        # Wavelet
        try:
            wp = pywt.WaveletPacket(data=y, wavelet='db1', mode='symmetric', maxlevel=3)
            for i, node in enumerate(wp.get_level(3, 'freq')):
                features[f"wavelet_{i}_mean"] = float(np.mean(np.abs(node.data)))
        except: pass

        # TEO
        try:
            teo = np.array([y[i]**2 - y[i-1]*y[i+1] for i in range(1, len(y)-1)])
            features.update(summarize_safe(teo, "teo"))
        except: pass

        # Parselmouth
        try:
            snd = parselmouth.Sound(file_path)
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features["jitter_local"] = float(jitter)
            features["shimmer_local"] = float(shimmer)
            features["hnr_mean"] = float(call(hnr, "Get mean", 0, 0))
            formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
            features["formant_f1"] = float(call(formant, "Get value at time", 1, snd.duration/2, "Hertz", "Linear"))
            features["formant_f2"] = float(call(formant, "Get value at time", 2, snd.duration/2, "Hertz", "Linear"))
            features["formant_f3"] = float(call(formant, "Get value at time", 3, snd.duration/2, "Hertz", "Linear"))
        except: pass

        # openSMILE (eGeMAPSv01a)
        if use_opensmile and opensmile_path is not None:
            try:
                import csv
        
                bin_path = "SMILExtract.exe" if os.name == "nt" else "SMILExtract"
                smile_bin = os.path.abspath(os.path.join(opensmile_path, "bin", bin_path)).replace("\\", "/")
                config_path = os.path.abspath(os.path.join(opensmile_path, "config", "gemaps", "v01a", "GeMAPSv01a.conf")).replace("\\", "/")
                input_path = os.path.abspath(file_path).replace("\\", "/")
        
                if not os.path.exists(config_path):
                    print(f"❌ GeMAPSv01a.conf not found: {config_path}")
        
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    output_csv = tmp.name
        
                result = subprocess.run([
                    smile_bin,
                    "-C", config_path,
                    "-I", input_path,
                    "-O", output_csv
                ], capture_output=True, text=True)
        
                if result.returncode != 0:
                    print(f"⚠️ openSMILE stderr: {result.stderr.strip()}")

                #  # Log output
                # if result.stderr.strip():
                #     print(f"⚠️ openSMILE stderr: {result.stderr.strip()}")
                # if result.stdout.strip():
                #     print(f"📤 openSMILE stdout: {result.stdout.strip()}")
        
                if os.path.exists(output_csv):
                    with open(output_csv, 'r') as f:
                        reader = csv.reader(f)
                        row = next(reader)

                    # Example log (removable)
                    # print(f"📄 output_csv sample content:\n{','.join(row)[:100]}...")
        
                    try:
                        if len(row) > 2 and (row[0].lower() == "unknown" or row[0] == ""):
                            numeric_values = [v for v in row[1:] if v != "?"]
                            for i, val in enumerate(numeric_values):
                                features[f"egemaps_{i}"] = float(val)
                        else:
                            print(f"⚠️ openSMILE invalid output format — {input_path}")
                    except Exception as e:
                        print(f"[openSMILE] row parsing error: {e} — {input_path}")
                    finally:
                        os.remove(output_csv)
                else:
                    print(f"❌ openSMILE output file was not created: {output_csv}")
        
            except Exception as e:
                print(f"[openSMILE] general error: {e} — {file_path}")

        return features if len(features) > 0 else None

    except Exception as e:
        print(f"ERROR ({file_path}): {e}")
        return None