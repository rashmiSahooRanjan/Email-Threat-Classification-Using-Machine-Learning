# Methodology: Email Threat Classification Using Machine Learning

## 1. Research Problem and Mathematical Formulation

### 1.1 Problem Statement
Email remains the primary initial access vector for modern cyber attacks, responsible for credential harvesting, corporate account compromise, and ransomware deployment. Traditional signature-based detection mechanisms fail against novel, polymorphic, or zero-day phishing campaigns. 

This research investigates the following core scientific question:
> *"Which observable, measurable structural and stylistic email characteristics reliably contribute to threat classification under a zero-cost, foundational machine learning paradigm?"*

### 1.2 Binary Classification Formulation
Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ represent the training corpus of $N$ email instances, where:
- $\mathbf{x}_i \in \mathbb{R}^{d}$ denotes the $d$-dimensional feature vector ($d = 20$) extracted from the $i$-th email message.
- $y_i \in \{0, 1\}$ represents the ground-truth binary label:
  $$y_i = \begin{cases} 0, & \text{Routine / Legitimate Email} \\ 1, & \text{Threatening / Phishing Email} \end{cases}$$

The objective is to learn a decision hypothesis $h: \mathbb{R}^{d} \to [0, 1]$ estimating the posterior probability of threat:
$$\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$$
subject to a classification decision boundary threshold $\tau = 0.50$:
$$\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)$$

---

## 2. End-to-End System Architecture

```
   Raw Dataset Files (CEAS_08, Enron, Ling, Nazario, Nigerian, SpamAssassin)
                                     │
                                     ▼
                   [ Data Cleaning & Sanitization ]
                   - Impute missing text fields
                   - Prune exact duplicates (82,391 unique)
                                     │
                                     ▼
                   [ Feature Engineering Pipeline ]
                   - Extract 20 deterministic features
                   - Length, URLs, keywords, punctuation, HTML
                                     │
                                     ▼
                 [ Leakage-Safe Stratified Split ]
                 - 80% Training Set (N = 65,912)
                 - 20% Independent Test Holdout (N = 16,479)
                                     │
                                     ▼
             [ Preprocessing: ColumnTransformer Pipeline ]
             - Median Imputation (leakage-safe)
             - StandardScaler (fitted exclusively on Train)
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
            [ Logistic Reg ]     [ KNN (k=5) ]   [ Decision Tree ]
            (max_iter=1000)     (uniform/mink)    (max_depth=12)
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                       [ Comprehensive Evaluation ]
                       - Accuracy, Precision, Recall, F1
                       - Confusion Matrix (TP, TN, FP, FN)
                       - ROC Curves & AUC Analysis
                                     │
                                     ▼
                       [ Final Model Selection ]
                       - Export final_model.joblib
                                     │
                                     ▼
                      [ Streamlit Deployment App ]
                      - Real-time feature extraction
                      - Probability gauge & risk tier assessment
```

---

## 3. Data Sanitization and Leakage Prevention Protocols

### 3.1 Strict Data Leakage Safeguards
Data leakage occurs when information from outside the training partition influences model training, yielding unrealistically optimistic test performance. We implement four mandatory anti-leakage safeguards:
1. **Zero Global Scaling**: No transformation parameters (means, standard deviations, medians) are computed over the full dataset.
2. **Stratified Splitting First**: Train/test splitting ($80/20, \text{random\_state}=42, \text{stratify}=y$) occurs before any transformer is fitted.
3. **Encapsulated Scikit-Learn Pipelines**: The `ColumnTransformer` preprocessor is chained directly with the estimator within a single `Pipeline`. Calling `pipeline.fit(X_train, y_train)` fits the scaler strictly on the training partition.
4. **Independent Holdout Evaluation**: Test samples ($X_{\text{test}}$) are passed strictly through `pipeline.predict()` and `pipeline.predict_proba()`, applying only the training-derived scaling factors.

---

## 4. Measurable Feature Engineering Methodology

Rather than opaque deep NLP embeddings that require excessive computation and lack explainability, our methodology engineers 20 deterministic structural indicators across four thematic categories:

### A. Dimensional & Volumetric Characteristics
1. **`message_length`**: Total characters ($\text{len}(\text{subject} + \text{body})$).
2. **`subject_length`**: Subject header character count.
3. **`body_length`**: Body character count.
4. **`word_count`**: Total whitespace-separated token count.

### B. Hyperlink and Network Vector Characteristics
5. **`url_count`**: Number of regex-matched URLs (`https?://\S+|www\.\S+`).
6. **`has_url`**: Binary indicator ($\mathbb{I}(\text{url\_count} > 0)$).
7. **`email_count`**: Embedded email address count.

### C. Psychological & Pretexting Keyword Frequencies
8. **`suspicious_keyword_count`**: Matches for credential and account lures (`account`, `verify`, `password`, `banking`, `suspend`, `paypal`, `tax`, `wire`, etc.).
9. **`urgency_keyword_count`**: Matches for psychological urgency cues (`urgent`, `immediately`, `act now`, `expires`, `warning`, `risk`, `deadline`).
10. **`currency_symbol_count`**: Matches for currency symbols (`$`, `€`, `£`, `¥`).
11. **`attachment_indicator`**: Binary flag for attachment terminology or file extensions (`.exe`, `.zip`, `.pdf`, `.xls`, etc.).

### D. Stylistic & Structural Formatting Indicators
12. **`exclamation_count`**: Frequency of `!` punctuation.
13. **`question_mark_count`**: Frequency of `?` punctuation.
14. **`uppercase_count`**: Total count of uppercase alphabetic characters.
15. **`uppercase_ratio`**: $\frac{\text{uppercase\_count}}{\text{total\_alphabetic\_characters}}$.
16. **`digit_count`**: Total count of numeric digits (`0`–`9`).
17. **`digit_ratio`**: $\frac{\text{digit\_count}}{\text{message\_length}}$.
18. **`special_char_count`**: Count of non-alphanumeric punctuation marks.
19. **`html_tag_count`**: Count of regex-matched HTML formatting tags.
20. **`html_indicator`**: Binary flag ($\mathbb{I}(\text{html\_tag\_count} > 0)$).

---

## 5. Foundational Machine Learning Models

### 5.1 Logistic Regression
- **Mechanism**: Models log-odds as a linear combination of standardized features:
  $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Optimization**: L-BFGS solver with $L_2$ regularization penalty ($C = 1.0$) and `class_weight='balanced'`.
- **Explainability**: Weights $\mathbf{w}$ provide direct feature coefficients indicating direction and magnitude of risk.

### 5.2 K-Nearest Neighbors (KNN)
- **Mechanism**: Non-parametric instance-based learner classifying query samples by majority vote among the $k = 5$ closest Euclidean neighbors in standardized feature space:
  $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^{d} (u_j - v_j)^2}$$
- **Hyperparameters**: $k = 5$, uniform weights, Euclidean distance metric ($p = 2$).

### 5.3 Decision Tree Classifier
- **Mechanism**: Recursive binary partitioning maximizing Information Gain via the Gini Impurity metric:
  $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
- **Regularization & Anti-Overfitting**: `max_depth=12`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'`.
- **Explainability**: Highly transparent decision tree splits directly inspectable for capstone defense and viva examination.

---

## 6. Evaluation Protocol and Metrics

Every model is assessed on the holdout test set using six primary performance criteria:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

Where:
- **TP (True Positive)**: Threatening/phishing emails correctly caught.
- **TN (True Negative)**: Routine emails correctly permitted.
- **FP (False Positive)**: Routine emails erroneously blocked (False Alarm).
- **FN (False Negative)**: Threatening emails erroneously admitted (Undetected Threat — critical risk).
