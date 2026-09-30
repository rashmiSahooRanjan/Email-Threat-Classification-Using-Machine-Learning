# Comparative Analysis of Foundational Machine Learning Models

## 1. Executive Summary
This document provides a rigorous comparative evaluation of three foundational machine learning architectures for **Email Threat Classification**:
1. **Logistic Regression** (Linear baseline with $L_2$ regularization)
2. **K-Nearest Neighbors (KNN)** (Instance-based non-parametric classifier, $k=5$)
3. **Decision Tree Classifier** (Non-linear recursive partitioning, $\text{max\_depth}=12$)

Evaluation was performed on a held-out, stratified test set of **16,479 emails** (20% partition of 82,391 unique emails). All preprocessing parameters were fitted strictly on the 80% training set.

---

## 2. Quantitative Performance Comparison

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 74.34% | 73.11% | 80.05% | 76.42% | 0.7912 | 6,853 | 5,398 | 2,520 | 1,708 | **< 0.1 ms** |
| **K-Nearest Neighbors (KNN)** | 88.06% | 88.92% | 87.97% | 88.44% | 0.9406 | 7,531 | 6,980 | 938 | 1,030 | ~1.3 ms |
| **Decision Tree (Selected)** | **89.64%** | **92.00%** | **87.69%** | **89.79%** | **0.9553** | 7,507 | 7,265 | **653** | 1,054 | **< 0.1 ms** |

---

## 3. In-Depth Multi-Metric Analysis

Rather than declaring the highest-accuracy model automatically superior, we evaluate the models across four operational dimensions:

### 3.1 Discrimination Ability (ROC-AUC)
- **Decision Tree** achieved the highest discrimination ability with an **ROC-AUC of 0.9553**, closely followed by **KNN (0.9406)**.
- **Logistic Regression (0.7912)** underperformed because threat indicators exhibit non-linear interactions (e.g. an email having multiple URLs is only dangerous if paired with credential keywords or high urgency, which linear combinations struggle to capture without manual interaction terms).

### 3.2 False Positive Burden (Precision)
- In enterprise security, excessive False Positives cause "alert fatigue" and disrupt business workflows when legitimate correspondence is quarantined.
- **Decision Tree achieved 92.00% Precision** with only **653 False Positives** (a 3.96% false alarm rate on the test set).
- By contrast, **Logistic Regression produced 2,520 False Positives** (nearly 4x higher), demonstrating that linear boundaries misclassify many complex legitimate emails.

### 3.3 Threat Catch Rate (Recall)
- Missing a malicious attack (False Negative) is the most hazardous outcome in email defense.
- **KNN caught 7,531 threats (87.97% Recall)**, **Decision Tree caught 7,507 threats (87.69% Recall)**, and **Logistic Regression caught 6,853 threats (80.05% Recall)**.
- Both Decision Tree and KNN demonstrated high sensitivity to phishing lures, missing fewer than 1,060 attacks out of 8,561 test threats.

### 3.4 Computational Efficiency & Latency
- **Decision Tree**: Training took **1.72 seconds**; inference takes **<0.1 ms per sample** via simple if-else branches.
- **KNN**: Training took **0.35 seconds** (memory storage only), but test evaluation took **~21 seconds** because each query requires distance computation across 65,912 training vectors. For large-scale email stream filtering, KNN's $\mathcal{O}(N \cdot d)$ inference latency is a significant operational bottleneck.
- **Logistic Regression**: Training took **1.64 seconds**; inference takes **<0.1 ms** via vector dot-product.

---

## 4. Confusion Matrix Breakdown

### A. Logistic Regression
- $\text{TN} = 5,398 \quad | \quad \text{FP} = 2,520$
- $\text{FN} = 1,708 \quad | \quad \text{TP} = 6,853$
- *Observation*: High False Positive rate due to linear hyperplane inability to model non-linear keyword/URL thresholds.

### B. K-Nearest Neighbors (KNN)
- $\text{TN} = 6,980 \quad | \quad \text{FP} = 938$
- $\text{FN} = 1,030 \quad | \quad \text{TP} = 7,531$
- *Observation*: Strong local neighborhood clustering captures phishing templates well, but computationally expensive.

### C. Decision Tree
- $\text{TN} = 7,265 \quad | \quad \text{FP} = 653$
- $\text{FN} = 1,054 \quad | \quad \text{TP} = 7,507$
- *Observation*: Optimal balance of lowest False Positives, high Recall, and superior overall F1-score (89.79%).

---

## 5. Justification for Final Model Selection

The **Decision Tree Pipeline** is formally designated as the project's **Final Production Model (`models/final_model.joblib`)** based on five concrete criteria:
1. **Best Balanced Metric Profile**: Leads the benchmark in Accuracy (89.64%), Precision (92.00%), F1-Score (89.79%), and ROC-AUC (0.9553).
2. **Minimal False Alarms**: Prunes False Positives down to 653 (compared to 938 for KNN and 2,520 for LR), preserving routine business communication.
3. **Sub-Millisecond Inference**: Extremely lightweight execution enables real-time scoring in the Streamlit application without user latency.
4. **Complete White-Box Explainability**: The decision logic consists of inspectable, hierarchical rules ideal for academic defense and stakeholder audits.
5. **No Data Leakage**: The entire pipeline incorporates the training-fitted ColumnTransformer, ensuring deterministic behavior in production.
