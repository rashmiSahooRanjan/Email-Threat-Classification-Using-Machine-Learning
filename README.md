# Email Threat Classification Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20Streamlit-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)]()

> **An End-to-End, Zero-Cost Machine Learning Capstone Solution for Email Threat & Phishing Detection Using Observable, Measurable Characteristics.**

---

## 1. Project Overview
This project presents an educational machine-learning capstone solution that investigates an aggregated corpus of 82,486 email instances, engineers 20 deterministic structural and stylistic features, trains and benchmarks foundational supervised classifiers (Logistic Regression, K-Nearest Neighbors, and Decision Tree), implements rigorous data-leakage prevention, and deploys an interactive Streamlit application for real-time risk scoring.

---

## 2. Problem Statement
Email remains the primary attack vector for over 90% of global cyber attacks, facilitating credential harvesting, malware delivery, and financial fraud. While modern deep-learning language models (e.g. BERT, LLMs) achieve high accuracy, they require significant computational power, operate as opaque black boxes, and incur substantial operational costs. This capstone addresses the challenge of building a lightweight, zero-cost, and fully explainable detection pipeline based strictly on observable email characteristics.

---

## 3. Objective
The central objective of this project is:
> *"Develop an end-to-end, zero-cost ML solution that investigates the assigned email dataset, prepares the data, builds and evaluates foundational machine learning models, interprets the results, and demonstrates the prediction through a lightweight Streamlit application."*

---

## 4. Research Direction
In strict accordance with the capstone specifications, the research direction is:
> *"Investigate which observable email characteristics contribute to threat classification."*

The model evaluates measurable email characteristics:
- Volumetric metrics (message length, subject length, body length, word count)
- Network vector indicators (hyperlink count, binary URL flag, referenced email addresses)
- Psychological and lure terminology (credential/banking keywords, urgency triggers, currency symbols, attachment references)
- Stylistic and formatting markers (capitalization ratio, exclamation marks, question marks, digit density, special characters, HTML tag counts)

---

## 5. Dataset
The project utilizes the actual, locally provided **Phishing Email Dataset** combining six well-known public cybersecurity email collections:

| Source File | File Size | Total Rows | Columns | Class 0 (Routine) | Class 1 (Threat) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CEAS_08.csv` | 64.76 MB | 39,154 | 7 | 17,312 (44.2%) | 21,842 (55.8%) |
| `Enron.csv` | 43.45 MB | 29,767 | 3 | 15,791 (53.0%) | 13,976 (47.0%) |
| `Ling.csv` | 8.91 MB | 2,859 | 3 | 2,401 (84.0%) | 458 (16.0%) |
| `Nazario.csv` | 7.45 MB | 1,565 | 7 | 0 (0.0%) | 1,565 (100.0%) |
| `Nigerian_Fraud.csv` | 8.77 MB | 3,332 | 7 | 0 (0.0%) | 3,332 (100.0%) |
| `SpamAssasin.csv` | 14.19 MB | 5,809 | 7 | 4,091 (70.4%) | 1,718 (29.6%) |
| `phishing_email.csv` | 101.67 MB | 82,486 | 2 | 39,595 (48.0%) | 42,891 (52.0%) |
| **Cleaned Unique Corpus** | **~250 MB** | **82,391** | **20 Features** | **39,588 (48.05%)** | **42,803 (51.95%)** |

---

## 6. Dataset Citation
The underlying dataset is formally cited as:
```bibtex
@dataset{alam2024phishing,
  author    = {Naser Abdullah Alam},
  title     = {Phishing Email Dataset},
  year      = {2024},
  publisher = {Kaggle},
  doi       = {10.34740/KAGGLE/DS/5074342},
  url       = {https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset}
}
```

---

## 7. Features
The pipeline extracts **20 deterministic, measurable characteristics** from raw email texts:

1. `message_length`: Total character count of subject and body.
2. `subject_length`: Character count of the subject header.
3. `body_length`: Character count of the email body.
4. `word_count`: Total whitespace-separated token count.
5. `url_count`: Count of detected hyperlinks (`http://`, `https://`, `www.`, domains).
6. `has_url`: Binary flag (1 if `url_count > 0` else 0).
7. `email_count`: Count of embedded email addresses.
8. `suspicious_keyword_count`: Frequency of credential/banking keywords (`account`, `verify`, `password`, `bank`, `login`, `paypal`, `ssn`, `tax`, `wire`).
9. `urgency_keyword_count`: Frequency of psychological urgency cues (`urgent`, `immediately`, `act now`, `expires`, `warning`, `deadline`).
10. `exclamation_count`: Frequency of exclamation marks (`!`).
11. `question_mark_count`: Frequency of question marks (`?`).
12. `uppercase_count`: Total count of uppercase alphabetic characters.
13. `uppercase_ratio`: Ratio of uppercase characters to total alphabetic characters.
14. `digit_count`: Count of numeric digits (`0`–`9`).
15. `digit_ratio`: Ratio of numeric digits to total message length.
16. `special_char_count`: Count of non-alphanumeric punctuation marks.
17. `currency_symbol_count`: Frequency of currency signs (`$`, `€`, `£`, `¥`).
18. `html_tag_count`: Count of HTML formatting tags (`<a`, `<div`, `<html>`, etc.).
19. `html_indicator`: Binary flag (1 if `html_tag_count > 0` else 0).
20. `attachment_indicator`: Binary flag indicating attachment references or file extensions (`.exe`, `.zip`, `.pdf`, `.xls`).

---

## 8. Exploratory Data Analysis (EDA)
EDA figures are automatically generated and saved to `outputs/figures/`:
- `01_class_distribution.png`: Near 50/50 balance across 82,391 emails.
- `02_missing_values.png`: Quantified missing subject headers across source collections.
- `03_message_length_distribution.png`: Contrast in body length distributions across classes.
- `04_url_count_distribution.png`: URL density comparisons between phishing and routine correspondence.
- `05_feature_boxplots.png`: Dispersion of keywords, capitalization, and punctuation.
- `06_correlation_heatmap.png`: Inter-feature Pearson correlation matrix.
- `07_feature_target_relationship.png`: Ranked feature associations with the threat label.

---

## 9. Preprocessing & Leakage Prevention
- **Leakage Prevention**: Stratified train/test split (80% train, 20% test, `random_state=42`) is performed before fitting any transformer.
- **ColumnTransformer**: Imputes missing numeric values with median, scales continuous features using `StandardScaler` fitted strictly on `X_train`, and passes binary indicators cleanly.
- **Pipeline Encapsulation**: The preprocessor is chained with each model inside a Scikit-learn `Pipeline`, guaranteeing zero test contamination during training, evaluation, and Streamlit inference.

---

## 10. Machine Learning Models
Three foundational architectures were trained using reproducible hyperparameters:
1. **Logistic Regression**: Linear baseline with L-BFGS solver, $L_2$ regularization ($C = 1.0$), and `class_weight='balanced'`.
2. **K-Nearest Neighbors (KNN)**: Non-parametric instance learner ($k = 5$, uniform weights, Euclidean distance).
3. **Decision Tree Classifier**: Non-linear recursive partitioning (`max_depth=12`, `min_samples_leaf=10`, `class_weight='balanced'`).

---

## 11. Evaluation Metrics
Evaluations are conducted on the reserved holdout test partition of **16,479 emails**:
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Full confusion matrix counts: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)

---

## 12. Results (Empirical Experimental Findings)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 74.34% | 73.11% | 80.05% | 76.42% | 0.7912 | 6,853 | 5,398 | 2,520 | 1,708 |
| **K-Nearest Neighbors (KNN)** | 88.06% | 88.92% | 87.97% | 88.44% | 0.9406 | 7,531 | 6,980 | 938 | 1,030 |
| **Decision Tree (Final Model)** | **89.64%** | **92.00%** | **87.69%** | **89.79%** | **0.9553** | 7,507 | 7,265 | **653** | 1,054 |

### Final Model Selection
The **Decision Tree Pipeline** is selected as the primary capstone model (`models/final_model.joblib`) because:
1. It achieves the **highest Precision (92.00%)** and lowest False Positive count (653), reducing false alarms.
2. It yields the **highest ROC-AUC (0.9553)** and **highest F1-score (89.79%)**.
3. It executes in **<0.1 ms per sample**, avoiding the latency of KNN.
4. Its hierarchical decision paths provide complete white-box explainability for student viva examination.

---

## 13. Streamlit Application
The web demonstration application (`app/app.py`) provides:
- Quick-load presets (Urgent Account Suspension, Lottery Wire Scam, Corporate Sprint Planning, Shipping Receipt).
- Real-time feature extraction table displaying all 20 characteristics.
- Binary threat prediction banner (Routine vs. Threatening).
- Threat probability gauge (%) and Risk Tier classification (Low, Moderate, High, Critical).
- List of observable threat indicators detected in the message.
- Interactive tabs for Measurable Features Guide and Model Performance Comparison.

---

## 14. Project Structure
```
Email_Threat_Classification/
│
├── dataset/
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── SpamAssasin.csv
│   └── phishing_email.csv
│
├── notebooks/
│   └── email_threat_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── error_analysis.py
│   ├── prediction.py
│   ├── generate_eda_figures.py
│   └── generate_notebook.py
│
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── knn_pipeline.joblib
│   ├── decision_tree_pipeline.joblib
│   └── final_model.joblib
│
├── outputs/
│   ├── figures/
│   │   ├── 01_class_distribution.png
│   │   ├── 02_missing_values.png
│   │   ├── 03_message_length_distribution.png
│   │   ├── 04_url_count_distribution.png
│   │   ├── 05_feature_boxplots.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_feature_target_relationship.png
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_knn.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── feature_importance.png
│   │   └── roc_curve_comparison.png
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   └── feature_importance.csv
│   ├── predictions/
│   │   └── test_predictions.csv
│   └── reports/
│       ├── data_quality_report.csv
│       ├── engineered_features.csv
│       └── test_split.csv
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_app_logic.py
│
├── docs/
│   ├── DATASET_DOCUMENTATION.md
│   ├── METHODOLOGY.md
│   ├── MODEL_COMPARISON.md
│   ├── ERROR_ANALYSIS.md
│   └── VIVA_QUESTIONS.md
│
├── technical_paper/
│   └── technical_paper.md
│
├── README.md
├── requirements.txt
├── .gitignore
└── run_project.bat
```

---

## 15. Installation (Windows & Cross-Platform)

### Step 1: Clone or Navigate to Project
```bash
cd c:\Users\Dell\Downloads\Project
```

### Step 2: Create and Activate Virtual Environment (Python 3.10+)
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```cmd
pip install -r requirements.txt
```

---

## 16. Running the Project

### A. Launch Streamlit Web Application
```cmd
streamlit run app/app.py
```
*Or double-click `run_project.bat`.*

### B. Retrain All Machine Learning Models
```cmd
python src/train_models.py
```

### C. Run Comprehensive Model Evaluation
```cmd
python src/evaluate_models.py
```

### D. Run Feature Importance & Error Analysis
```cmd
python src/error_analysis.py
```

### E. Launch Jupyter Notebook
```cmd
jupyter notebook notebooks/email_threat_classification.ipynb
```

---

## 17. Testing
Execute the complete automated test suite using `pytest`:
```cmd
pytest tests/ -v
```
All **15 unit and integration tests** validate data loading, feature extraction, regular expressions, preprocessing, model pipelines, and empty/adversarial input resilience.

---

## 18. Limitations
1. **Header Authentication**: The system operates on email content and visible headers; it does not query live SPF, DKIM, or DMARC DNS records.
2. **Attachment Sandboxing**: Evaluates attachment indicator keywords and file extensions rather than executing file payloads in a secure sandbox.
3. **Evasion Susceptibility**: Short, plain-text spear-phishing emails containing zero URLs or generic conversational phrasing exhibit low signal density and represent the majority of false negatives.

---

## 19. Future Scope
1. **Domain Age & Threat Intel API Feeds**: Querying passive DNS and VirusTotal/PhishTank for real-time URL reputation.
2. **Dynamic Threshold Tuning**: Adjusting classification cutoff thresholds to maximize Recall for high-security enterprise environments.
3. **Lightweight Hybrid Ensembles**: Combining top-100 TF-IDF n-grams with our 20 structural features for enhanced vocabulary sensitivity.

---

## 20. Disclaimer
> **⚠️ ACADEMIC & RESEARCH DISCLAIMER**:  
> This software is an educational machine-learning capstone prototype developed strictly for academic demonstration, research, and viva examination. It is **NOT** a production cybersecurity system and must not be deployed as the sole perimeter defense against cyber threats or real-world email compromise.

## 👨‍💻 Authors

| Name                          | Role      |
| ----------------------------- | --------- |
| **Rashmi Ranjan Sahoo** | ML Developer |
