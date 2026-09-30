# Empirical Error Analysis

## 1. Overview of Error Cohorts
The error analysis was conducted on the reserved test partition of **16,479 emails** (20% stratified holdout).
Evaluated model: **Decision Tree Pipeline**.

### Test Set Confusion Breakdown:
- **True Positives (TP)**: 7,507 (45.55%) — Threatening/phishing emails correctly identified.
- **True Negatives (TN)**: 7,265 (44.09%) — Routine/legitimate emails correctly identified.
- **False Positives (FP)**: 653 (3.96%) — Routine emails misclassified as threats (False Alarm).
- **False Negatives (FN)**: 1,054 (6.4%) — Threatening emails misclassified as routine (Undetected Threat).

- **False Positive Rate (FPR)**: 8.25% of all legitimate emails.
- **False Negative Rate (FNR)**: 12.31% of all threatening emails.

---

## 2. Average Feature Profiles Across Cohorts

| Feature | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) |
| :--- | :---: | :---: | :---: | :---: |
| `message_length` | 2237.46 | 1842.34 | 1592.98 | 1236.04 |
| `url_count` | 4.06 | 1.13 | 1.12 | 3.25 |
| `suspicious_keyword_count` | 1.0 | 2.01 | 1.75 | 2.44 |
| `urgency_keyword_count` | 0.48 | 0.4 | 0.35 | 0.51 |
| `uppercase_ratio` | 0.04 | 0.04 | 0.03 | 0.08 |
| `special_char_count` | 186.39 | 332.04 | 92.04 | 101.51 |
| `html_indicator` | 0.0 | 0.0 | 0.0 | 0.0 |
| `attachment_indicator` | 0.13 | 0.06 | 0.03 | 0.02 |

---

## 3. In-Depth Root Cause Investigation

### A. False Positives (Routine Emails Classified as Threats)
1. **Promotional and Transactional Emails**:
   - Marketing newsletters and automated receipts from legitimate corporate sources frequently contain multiple links (`url_count`), HTML formatting (`html_indicator=1`), and decorative punctuation.
   - Legitimate transactional notices (e.g. order confirmations, flight itineraries) frequently contain transaction numbers and keywords like 'confirm', 'account', or 'verify' which elevates the `suspicious_keyword_count`.
2. **Elevated Punctuation and Capitalization**:
   - Informal business correspondence or enthusiastic interpersonal announcements using uppercase letters or exclamation marks mimic phishing lure characteristics.

### B. False Negatives (Threats Classified as Routine)
1. **Low-Keyword, Stealthy Phishing**:
   - Sophisticated targeted phishing (spear phishing) often eschews generic spam keywords ('free', 'winner', 'urgent') in favor of subtle, natural phrasing (e.g., "Please review the updated agreement").
2. **Plain-Text URL-Free Pretexting**:
   - Attackers engaging in multi-stage social engineering (e.g., initial relationship building or phone-based vishing pretext) do not include external links or attachments in their introductory email, causing `url_count=0` and `has_url=0`.
3. **Short Lures with Low Signal Density**:
   - Very brief phishing messages with fewer than 10 words do not provide sufficient statistical signal across length, keyword, or character distribution metrics.

---

## 4. Mitigation Strategies & Recommendations
1. **Domain Reputation Integration**: Supplement feature extraction with sender domain age, SPF/DKIM verification status, and WHOIS lookup.
2. **Threshold Tuning**: Adjust decision thresholds based on organizational tolerance: lowering the classification cutoff prioritizes Recall (minimizing dangerous False Negatives) at the cost of higher False Positives.
3. **Ensemble Stacking**: Combine measurable characteristic models with domain-level threat intelligence feeds for layered perimeter defense.
