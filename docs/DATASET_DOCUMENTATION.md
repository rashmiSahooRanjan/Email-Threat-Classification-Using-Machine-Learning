# Dataset Documentation: Phishing Email Dataset

## 1. Dataset Overview and Provenance
- **Dataset Title**: Phishing Email Dataset
- **Lead Compiler / Author**: Naser Abdullah Alam (2024)
- **Host Repository**: Kaggle Datasets
- **Digital Object Identifier (DOI)**: [10.34740/KAGGLE/DS/5074342](https://doi.org/10.34740/KAGGLE/DS/5074342)
- **Compilation Context**: The collection aggregates multiple well-known public cybersecurity email corpora into a single harmonized repository to benchmark phishing and email threat detection algorithms.
- **Total Unified Corpus Records**: 82,486 email instances.

---

## 2. Directory Inventory & Files Detected

The local `./dataset/` directory contains 7 CSV files totaling ~256 MB:

| File Name | File Size (MB) | Total Rows | Total Columns | Column List | Class 0 (Routine) | Class 1 (Threat) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| `CEAS_08.csv` | 64.76 | 39,154 | 7 | `['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']` | 17,312 (44.2%) | 21,842 (55.8%) |
| `Enron.csv` | 43.45 | 29,767 | 3 | `['subject', 'body', 'label']` | 15,791 (53.0%) | 13,976 (47.0%) |
| `Ling.csv` | 8.91 | 2,859 | 3 | `['subject', 'body', 'label']` | 2,401 (84.0%) | 458 (16.0%) |
| `Nazario.csv` | 7.45 | 1,565 | 7 | `['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']` | 0 (0.0%) | 1,565 (100.0%) |
| `Nigerian_Fraud.csv` | 8.77 | 3,332 | 7 | `['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']` | 0 (0.0%) | 3,332 (100.0%) |
| `SpamAssasin.csv` | 14.19 | 5,809 | 7 | `['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']` | 4,091 (70.4%) | 1,718 (29.6%) |
| `phishing_email.csv` | 101.67 | 82,486 | 2 | `['text_combined', 'label']` | 39,595 (48.0%) | 42,891 (52.0%) |

### Mathematical Alignment Verification
Sum of rows of source datasets:
$$\sum = 39,154 + 29,767 + 2,859 + 1,565 + 3,332 + 5,809 = 82,486 \text{ rows}$$
Sum of Class 1 (Threat) labels:
$$21,842 + 13,976 + 458 + 1,565 + 3,332 + 1,718 = 42,891$$
Sum of Class 0 (Routine) labels:
$$17,312 + 15,791 + 2,401 + 0 + 0 + 4,091 = 39,595$$

This confirms empirically that `phishing_email.csv` is the concatenated, lowercased aggregate of the six primary source corpora.

---

## 3. Detailed Column Descriptions

### A. Raw Source Dataset Schema (e.g. `CEAS_08.csv`, `Nazario.csv`, `SpamAssasin.csv`)
1. **`sender`** (`object` / string): The email `From:` header containing sender display name and address (e.g. `Young Esposito <Young@iworld.de>`).
2. **`receiver`** (`object` / string): The recipient address in `To:` or `Cc:` headers. Missing in some archived spam feeds where envelope BCC was used.
3. **`date`** (`object` / string): RFC-822 formatted transmission timestamp.
4. **`subject`** (`object` / string): Email subject line header. Contains capitalization, punctuation, and urgent phrasing.
5. **`body`** (`object` / string): Full multiline email payload containing text, ASCII art, HTML tags, and hyperlinks.
6. **`urls`** (`int64` / numeric): Count or binary indicator of hyperlinks identified during corpus harvesting.
7. **`label`** (`int64`): Ground truth binary classification indicator (`0` = Routine / Legitimate, `1` = Threatening / Phishing).

### B. Consolidated Corpus Schema
1. **`source_file`** (`string`): The originating benchmark collection from which the email was drawn.
2. **`subject`** (`string`): Cleaned subject string.
3. **`body`** (`string`): Cleaned body string.
4. **`text_combined`** (`string`): Unified full-text representation (`subject + " " + body`).
5. **`label`** (`int64`): Ground truth target variable.

---

## 4. Target Variable and Class Definitions

The classification task is formulated as a supervised binary classification problem:

- **Class `0`: Routine / Legitimate Email**
  - **Definition**: Normal business communications, interpersonal letters, technical newsletters, order confirmations, and non-malicious conversational messages.
  - **Sources**: Enron corporate email corpus, SpamAssassin ham archives, Ling-Spam legitimate emails, and CEAS 2008 legitimate challenge emails.
  - **Total instances**: 39,595 (48.0% of raw corpus).

- **Class `1`: Threatening / Phishing Email**
  - **Definition**: Malicious communications engineered to deceive recipients into surrendering credentials, wire transfers, revealing personal sensitive information (SSN, banking PINs), or opening malware attachments.
  - **Sources**: Nazario phishing corpus, Nigerian advance-fee fraud archives, CEAS 2008 phishing lures, and SpamAssassin spam/threat feeds.
  - **Total instances**: 42,891 (52.0% of raw corpus).

---

## 5. Data Quality, Missing Values, and Duplicates Analysis

### A. Missing Values
- **`subject`**: 347 records across all source files have missing or null subject lines (Enron: 198, Ling: 62, Nigerian Fraud: 39, CEAS_08: 28, SpamAssassin: 16, Nazario: 4).
  - *Handling Decision*: Imputed with empty string `""` so feature extractors yield `subject_length = 0` without data loss.
- **`body`**: Only 1 record in `SpamAssasin.csv` had a null body. Imputed with empty string `""`.
- **`receiver` / `date`**: Contain missing values in Nazario and Nigerian Fraud due to header stripping in archived spam repositories. Because our research emphasizes content-based measurable characteristics, these headers are not required for classification.

### B. Duplicates & Deduplication
- In the raw concatenated dataset, 95 records are exact duplicates of both subject and body.
- *Handling Decision*: Exact duplicates are pruned during sanitization, leaving **82,391 unique emails**. This prevents identical emails from leaking between the training and test sets.

### C. Class Balance Assessment
- Unique Cleaned Dataset:
  - Threatening (Class 1): 42,803 (51.95%)
  - Routine (Class 0): 39,588 (48.05%)
- **Imbalance Ratio**: 1.08 : 1.
- *Conclusion*: The dataset is remarkably balanced. Extreme resampling (such as SMOTE or aggressive undersampling) is unnecessary and would risk introducing synthetic artifacts. Standard class weighting (`class_weight="balanced"`) is applied to give equal penalty to both classes.

---

## 6. Preprocessing and Feature Extraction Decisions

1. **Raw Source Text Utilization**:
   Rather than relying exclusively on the pre-lowercased and punctuation-stripped `phishing_email.csv`, our pipeline consolidates the raw source CSVs to preserve uppercase capitalization, exclamation marks, question marks, and HTML tags.
2. **Measurable Characteristics vs. Deep NLP**:
   In strict compliance with the capstone scope, 20 deterministic, explainable numerical features are extracted per email.
3. **Leakage Prevention**:
   All imputers, standardizers, and robust scalers are fitted exclusively on the 80% training partition after stratified splitting (`stratify=y, random_state=42`).

---

## 7. Citation Information

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
Acknowledge original underlying corpora:
- CEAS 2008 Anti-Spam Challenge
- Enron Email Dataset (FERC / CALO Project)
- Ling-Spam Dataset (Ion Androutsopoulos et al.)
- Jose Nazario Phishing Corpus
- Nigerian 419 Fraud Email Archive
- Apache SpamAssassin Public Corpus
