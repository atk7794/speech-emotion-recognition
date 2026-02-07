# Results

This directory contains the **final evaluation results** of all models used in this study.
All results reported here are obtained **exclusively from the test split** and represent
the final performance of the trained models.

The results are organized by **model category** and **experimental setting** to support
systematic comparison, analysis, and reproducibility.

All tables and figures reported in the thesis and presentation are derived from the CSV
files provided in this directory.

---

## 📌 Evaluation Protocol

The experimental pipeline follows a **strict data split strategy**:

- **Train set**: Used for model training
- **Development (Dev) set**: Used for hyperparameter tuning and model selection
- **Test set**: Used only once for final evaluation

⚠️ **Important:**  
All CSV files in this directory correspond to **test-set predictions only**.
No training or validation data is included here.

---

## 🗂️ Directory Structure

```text
results/
├── ml/        # Classical machine learning results
├── dl/        # Deep learning results
├── sota/      # Pretrained SOTA model results
```

---

# 📄 CSV File Structure

Each CSV file stores sample-level prediction outputs rather than aggregated metrics.

Typical columns include:

- FILENAME – Audio or video segment identifier
- PREDICT_LABEL – Model-predicted emotion
- TRUE_LABEL – Ground-truth emotion
- MATCHING – Whether prediction matches ground truth
- *_POSSIBILITY – Class probability or softmax output for each emotion

Example:
```text
FILENAME,PREDICT_LABEL,TRUE_LABEL,MATCHING,ANGER_POSSIBILITY,...,SURPRISE_POSSIBILITY
dia103_utt3.wav,surprise,surprise,True,0.0013,...,0.9281
```

📌 **Note:**
Classification metrics such as Accuracy, Precision, Recall, and F1-score are computed from these CSV files using evaluation scripts and notebooks.

---

# 1️⃣ Machine Learning Results (ml/)

This folder contains results for classical ML classifiers trained on handcrafted acoustic features extracted from speech.

Algorithms Included

- svm_1/ – Support Vector Machine
- rf_1/ – Random Forest
- knn_1/ – K-Nearest Neighbors
- bagging_knn_1/ – Bagged KNN
- gb_1/ – Gradient Boosting
- xgb_1/ – XGBoost
- voting_1/ – Hard Voting Ensemble
- soft_voting_1/ – Soft Voting Ensemble
- s-mlp_1/ – Sklearn-based MLP

Each subfolder corresponds to one algorithm and contains multiple CSV files representing different feature selection strategies.

---

# Feature Configurations

For each algorithm, multiple feature selection strategies were evaluated.
This is reflected directly in the CSV filenames:

| Keyword                                        | Description                                  |
| ---------------------------------------------- | -------------------------------------------- |
| `meld_features_ml_1_clean`                     | Cleaned baseline features                    |
| `meld_features_ml_1_clean_balanced`            | Cleaned and class-balanced baseline features |
| `meld_features_selected_boruta`                | Boruta-selected features                     |
| `meld_features_selected_rf`                    | Random Forest feature selection              |
| `meld_features_selected_rfe`                   | Recursive Feature Elimination                |
| `meld_features_selected_kbest`                 | K-Best feature selection                     |
| `meld_features_selected_sffs`                  | Sequential Floating Forward Selection        |
| `meld_features_selected_autoencoder`           | Autoencoder-based feature compression        |
| `meld_features_selected_boruta_rf`             | Hybrid Boruta + RF                           |
| `meld_features_selected_boruta_autoencoder`    | Hybrid Boruta + Autoencoder                  |
| `meld_features_selected_boruta_autoencoder_rf` | Hybrid Boruta + Autoencoder + RF             |


All experiments were conducted on balanced datasets to address class imbalance.

---

# 2️⃣ Deep Learning Results (dl/)

This folder contains results from deep learning architectures trained on
learned audio representations.

Architectures Included

- bilstm_results.csv
- cnn_bilstm_hybrid_results.csv
- keras_bilstm_results.csv
- melcnn_results.csv
- mfccc_results.csv
- multicnn_results.csv
- rgbcnn_results.csv
- resnet18_rgb_results.csv
- efficientnet_rgbfusion_results.csv

All results are evaluated on the test split following model selection on the dev set.

---

# MLP-Based DL Models

Subfolders:

- keras_mlp/ – Keras-based MLP classifiers
- torch_mlp/ – PyTorch-based MLP classifiers

These models were trained on the same handcrafted features used in ML, allowing a fair ML vs DL comparison.

Feature selection strategies mirror those in the ML experiments (Boruta, RF, RFE, SFFS, Autoencoder, hybrid methods).

---

# 3️⃣ State-of-the-Art Results (sota/)

This folder contains results obtained using pretrained self-supervised speech models.

Models Included

- HuBERT
- wav2vec 2.0
- WavLM

Each model is evaluated in two settings:

- End-to-end fine-tuning
- MLP classifier on extracted embeddings

Multiple experimental runs are provided (e.g. _mlp_2_) for robustness analysis.

---

# 📊 Metric Standardization

Across all ML, DL, and SOTA experiments:

- Metrics are computed on the **test split**
- Metrics are computed by aggregating prediction-level outputs across the entire test set.
- Weighted averages are used to account for class imbalance
- Evaluation protocol is consistent to ensure comparability

---

# 📌 Notes

- All CSV files are final experimental outputs
- No post-processing is applied outside this directory
- Results are directly reproducible using the scripts and notebooks in src/. (Google Colab codes are also available in src/main/senior.ipynb)

---

# 🎯 Purpose

This directory serves as the final evaluation archive of the study, ensuring:

- Transparent reporting
- Fair model comparison
- Reproducibility
- Clear separation between training, validation, and test evaluation