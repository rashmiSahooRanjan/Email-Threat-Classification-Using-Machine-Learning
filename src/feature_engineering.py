"""
src/feature_engineering.py
Feature extraction and engineering module for Email Threat Classification.

Extracts objectively measurable characteristics from raw email text, subjects,
and bodies. Features are fully reproducible and explainable for capstone defense.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np

# Pre-compiled regular expressions for high throughput and consistent matching
URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?",
    re.IGNORECASE
)
EMAIL_REGEX = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)
HTML_TAG_REGEX = re.compile(
    r"<[a-zA-Z][^>]*>|</[a-zA-Z]+>|<!--.*?-->",
    re.IGNORECASE | re.DOTALL
)
SUSPICIOUS_KEYWORD_REGEX = re.compile(
    r"\b(account|verify|verification|password|banking|bank|suspend|suspended|"
    r"security|secure|click|update|login|logon|free|winner|claim|paypal|"
    r"ssn|social security|tax|refund|wire|transfer|dollar|dollars|usd|"
    r"unauthorized|access|confirm|confirmation|wallet|billing|invoice|"
    r"inheritance|fund|funds|lottery|prize|credential|credentials)\b",
    re.IGNORECASE
)
URGENCY_KEYWORD_REGEX = re.compile(
    r"\b(urgent|urgently|immediately|immediate|act now|action required|"
    r"attention|expire|expires|expired|expiration|asap|warning|risk|"
    r"terminate|termination|limited time|restricted|critical|alert|"
    r"within 24 hours|within 48 hours|final notice|deadline)\b",
    re.IGNORECASE
)
ATTACHMENT_KEYWORD_REGEX = re.compile(
    r"\b(attachment|attached|attach|enclosed|hplno|\.exe|\.zip|\.scr|\.pdf|\.doc|\.docx|\.xls|\.xlsx|\.iso|\.vbs|\.bat)\b",
    re.IGNORECASE
)
CURRENCY_REGEX = re.compile(r"[\$€£¥₹]")
SPECIAL_CHARS = set("!@#$%^&*()_+-=[]{}|;:'\",.<>/?`~\\")

# Complete feature documentation dictionary for Section 6 requirement
FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "message_length": {
        "definition": "Total character count of the combined email (subject + body).",
        "calculation": "len(text)",
        "reason": "Phishing emails often exhibit distinct lengths—either extremely concise lures or long auto-generated disclaimers.",
        "data_source": "subject + body text",
        "limitations": "Legitimate newsletters can also be lengthy; length alone is not conclusive."
    },
    "subject_length": {
        "definition": "Character count of the email subject header.",
        "calculation": "len(subject)",
        "reason": "Threat actors frequently craft short, alarming subjects (e.g. 'URGENT') or spoofed forwarded trails.",
        "data_source": "subject column",
        "limitations": "May be 0 if the email lacks a subject line."
    },
    "body_length": {
        "definition": "Character count of the email body text.",
        "calculation": "len(body)",
        "reason": "Separates the primary body content payload size from the subject line.",
        "data_source": "body column",
        "limitations": "Raw HTML bodies have larger byte counts than plain text."
    },
    "word_count": {
        "definition": "Total count of whitespace-separated tokens.",
        "calculation": "len(text.split())",
        "reason": "Measures message verbosity; spam templates follow different vocabulary distributions than routine correspondence.",
        "data_source": "subject + body text",
        "limitations": "Does not account for multilingual tokenization differences."
    },
    "url_count": {
        "definition": "Count of web hyperlinks and domain references detected in text.",
        "calculation": "Count of regex matches for http(s), www, or domain structures.",
        "reason": "Credential harvesting and malware distribution depend on directing victims to malicious URLs.",
        "data_source": "subject + body text",
        "limitations": "Obfuscated or defanged URLs (e.g. hxxp) may evade basic regex."
    },
    "has_url": {
        "definition": "Binary indicator specifying if at least one URL is present.",
        "calculation": "1 if url_count > 0 else 0",
        "reason": "Provides a non-linear decision boundary threshold separating URL-bearing emails from purely conversational emails.",
        "data_source": "derived from url_count",
        "limitations": "Binary feature compresses the magnitude of URL density."
    },
    "email_count": {
        "definition": "Count of email addresses referenced within the message.",
        "calculation": "Count of regex matches for email address formats.",
        "reason": "Advance-fee fraud and phishing scams frequently list alternate reply-to or money transfer contacts.",
        "data_source": "subject + body text",
        "limitations": "Legitimate thread headers also contain multiple forwarded email addresses."
    },
    "suspicious_keyword_count": {
        "definition": "Frequency of security, credential, and financial lure keywords.",
        "calculation": "Count of word boundary matches for targeted lure terminology.",
        "reason": "Social engineering attacks heavily leverage security, credential reset, and financial pretexting.",
        "data_source": "subject + body text",
        "limitations": "Routine IT support or bank statements also contain legitimate security terminology."
    },
    "urgency_keyword_count": {
        "definition": "Frequency of psychological urgency and pressure terminology.",
        "calculation": "Count of word boundary matches for urgency-inducing vocabulary.",
        "reason": "Attackers induce panic (e.g. 'account suspension in 24 hours') to bypass rational scrutiny.",
        "data_source": "subject + body text",
        "limitations": "Legitimate business deadlines can contain urgent phrases."
    },
    "exclamation_count": {
        "definition": "Frequency of exclamation mark (!) characters.",
        "calculation": "text.count('!')",
        "reason": "Phishing lures and scam prize notifications rely heavily on dramatic punctuation for emphasis.",
        "data_source": "subject + body text",
        "limitations": "Excited informal legitimate personal emails may also use exclamation marks."
    },
    "question_mark_count": {
        "definition": "Frequency of question mark (?) characters.",
        "calculation": "text.count('?')",
        "reason": "Phishing pretexts frequently employ provocative inquiries ('Did you authorize this transfer?').",
        "data_source": "subject + body text",
        "limitations": "Normal inquiries and conversational emails inherently contain question marks."
    },
    "uppercase_count": {
        "definition": "Total count of uppercase alphabetic characters.",
        "calculation": "Sum of isupper() characters in text.",
        "reason": "SHOUTING in subjects and bodies is a hallmark of aggressive spam and urgent threat lures.",
        "data_source": "subject + body text",
        "limitations": "Acronyms and standard corporate capitalized headers contribute to counts."
    },
    "uppercase_ratio": {
        "definition": "Ratio of uppercase characters to total alphabetic characters.",
        "calculation": "uppercase_count / total_alpha_count (or 0.0 if no letters).",
        "reason": "Normalized capitalization metric independent of total document length.",
        "data_source": "subject + body text",
        "limitations": "Very short emails with acronyms (e.g. 'OK THANKS') may yield high ratios."
    },
    "digit_count": {
        "definition": "Total count of numeric digits (0-9).",
        "calculation": "Sum of isdigit() characters in text.",
        "reason": "Scam messages include transaction IDs, telephone numbers, dollar figures, and spoofed bank pins.",
        "data_source": "subject + body text",
        "limitations": "Routine financial statements and tabular data contain high digit counts."
    },
    "digit_ratio": {
        "definition": "Ratio of numeric digits to total character length.",
        "calculation": "digit_count / message_length (or 0.0 if empty).",
        "reason": "Measures numeric density normalized across email sizes.",
        "data_source": "subject + body text",
        "limitations": "Spreadsheet-heavy business communications can elevate digit ratio."
    },
    "special_char_count": {
        "definition": "Count of non-alphanumeric punctuation and symbol characters.",
        "calculation": "Sum of characters belonging to ASCII special characters.",
        "reason": "Obfuscation, decorative ASCII borders, and currency symbols appear frequently in spam/phishing.",
        "data_source": "subject + body text",
        "limitations": "Code snippets, technical logs, or formatted signatures contain high punctuation."
    },
    "currency_symbol_count": {
        "definition": "Count of financial currency symbols ($, €, £, ¥, ₹).",
        "calculation": "Regex count of currency symbols.",
        "reason": "Advance fee fraud, lottery, and payroll phishing messages frequently emphasize financial amounts.",
        "data_source": "subject + body text",
        "limitations": "Legitimate purchasing invoices and ecommerce receipts also feature currency symbols."
    },
    "html_tag_count": {
        "definition": "Count of HTML tags detected within the email body.",
        "calculation": "Count of regex matches for HTML tag structures.",
        "reason": "Brand impersonation phishing relies heavily on HTML rendering to replicate corporate visual identities.",
        "data_source": "subject + body text",
        "limitations": "Legitimate promotional marketing emails also use complex HTML formatting."
    },
    "html_indicator": {
        "definition": "Binary flag indicating whether HTML formatting tags are present.",
        "calculation": "1 if html_tag_count > 0 else 0",
        "reason": "Distinguishes plain-text interpersonal correspondence from rich-format HTML templates.",
        "data_source": "derived from html_tag_count",
        "limitations": "Does not differentiate between simple bolding and hidden redirect forms."
    },
    "attachment_indicator": {
        "definition": "Binary flag indicating presence of attachment terminology or dangerous file extensions.",
        "calculation": "1 if attachment keywords or file extensions are matched else 0.",
        "reason": "Phishing campaigns frequently deliver payloads via attached malicious documents or archives.",
        "data_source": "subject + body text",
        "limitations": "Routine business communications frequently attach legitimate documents."
    }
}

FEATURE_COLUMNS = list(FEATURE_METADATA.keys())


def extract_features_from_text(
    text: str,
    subject: str = "",
    body: Optional[str] = None
) -> Dict[str, Union[int, float]]:
    """
    Extracts all 20 measurable features from raw email text.

    Args:
        text: Full email text, or subject + body combined.
        subject: Optional dedicated subject string.
        body: Optional dedicated body string. If None, text is treated as body.

    Returns:
        Dictionary mapping feature names to numerical values.
    """
    if text is None:
        text = ""
    if subject is None:
        subject = ""
    if body is None:
        body = text

    full_text = f"{subject} {body}".strip() if subject and body else (text or body or subject)

    # Basic length and counts
    msg_len = len(full_text)
    subj_len = len(subject)
    body_len = len(body)
    words = full_text.split()
    word_cnt = len(words)

    # Character-level single-pass scan
    alpha_cnt = 0
    upper_cnt = 0
    digit_cnt = 0
    excl_cnt = 0
    quest_cnt = 0
    special_cnt = 0

    for ch in full_text:
        if ch.isalpha():
            alpha_cnt += 1
            if ch.isupper():
                upper_cnt += 1
        elif ch.isdigit():
            digit_cnt += 1
        elif ch == "!":
            excl_cnt += 1
        elif ch == "?":
            quest_cnt += 1

        if ch in SPECIAL_CHARS:
            special_cnt += 1

    upper_ratio = (upper_cnt / alpha_cnt) if alpha_cnt > 0 else 0.0
    digit_ratio = (digit_cnt / msg_len) if msg_len > 0 else 0.0

    # Pattern-based features
    url_cnt = len(URL_REGEX.findall(full_text))
    has_url = 1 if url_cnt > 0 else 0

    email_cnt = len(EMAIL_REGEX.findall(full_text))
    html_tags = len(HTML_TAG_REGEX.findall(full_text))
    html_ind = 1 if html_tags > 0 else 0

    sus_cnt = len(SUSPICIOUS_KEYWORD_REGEX.findall(full_text))
    urg_cnt = len(URGENCY_KEYWORD_REGEX.findall(full_text))
    currency_cnt = len(CURRENCY_REGEX.findall(full_text))
    att_ind = 1 if ATTACHMENT_KEYWORD_REGEX.search(full_text) else 0

    return {
        "message_length": msg_len,
        "subject_length": subj_len,
        "body_length": body_len,
        "word_count": word_cnt,
        "url_count": url_cnt,
        "has_url": has_url,
        "email_count": email_cnt,
        "suspicious_keyword_count": sus_cnt,
        "urgency_keyword_count": urg_cnt,
        "exclamation_count": excl_cnt,
        "question_mark_count": quest_cnt,
        "uppercase_count": upper_cnt,
        "uppercase_ratio": round(upper_ratio, 4),
        "digit_count": digit_cnt,
        "digit_ratio": round(digit_ratio, 4),
        "special_char_count": special_cnt,
        "currency_symbol_count": currency_cnt,
        "html_tag_count": html_tags,
        "html_indicator": html_ind,
        "attachment_indicator": att_ind,
    }


def extract_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts engineered features from a DataFrame.
    Expects 'subject' and 'body' columns, or a 'text_combined' column.

    Returns:
        DataFrame containing only the 20 engineered numerical features.
    """
    has_subj_body = "subject" in df.columns and "body" in df.columns

    records = []
    if has_subj_body:
        for subj, bdy in zip(df["subject"], df["body"]):
            rec = extract_features_from_text(text="", subject=str(subj), body=str(bdy))
            records.append(rec)
    elif "text_combined" in df.columns:
        for txt in df["text_combined"]:
            rec = extract_features_from_text(text=str(txt))
            records.append(rec)
    else:
        raise ValueError("DataFrame must contain ('subject', 'body') or 'text_combined' columns.")

    feat_df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    return feat_df


def extract_full_features_dataset(
    df_clean: pd.DataFrame,
    cache_path: Optional[str] = "outputs/reports/engineered_features.parquet"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds or loads cached engineered features for the cleaned dataset.
    Ensures reproducibility and instant reloading across training and evaluation runs.

    Args:
        df_clean: Sanitized input DataFrame with labels.
        cache_path: Optional file path to store/load parquet cache.

    Returns:
        Tuple of (X_features, y_labels).
    """
    # Check if parquet or csv cache exists
    csv_cache = "outputs/reports/engineered_features.csv"
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached engineered features from {cache_path}...")
        feat_df = pd.read_parquet(cache_path)
        y = feat_df["label"]
        X = feat_df[FEATURE_COLUMNS]
        return X, y
    elif os.path.exists(csv_cache):
        print(f"Loading cached engineered features from {csv_cache}...")
        feat_df = pd.read_csv(csv_cache)
        y = feat_df["label"]
        X = feat_df[FEATURE_COLUMNS]
        return X, y

    print(f"Extracting 20 engineered features across {len(df_clean)} emails...")
    X = extract_features_dataframe(df_clean)
    y = df_clean["label"].reset_index(drop=True)

    # Save cache for fast reuse
    os.makedirs("outputs/reports", exist_ok=True)
    combined_export = X.copy()
    combined_export["label"] = y
    try:
        combined_export.to_parquet(cache_path, index=False)
        print(f"Engineered features successfully cached to: {cache_path}")
    except Exception:
        # Fallback to CSV if parquet engine not installed
        combined_export.to_csv(csv_cache, index=False)
        print(f"Engineered features successfully cached to: {csv_cache}")

    return X, y


if __name__ == "__main__":
    print("Testing feature_engineering.py with sample inputs...")
    sample_email = (
        "Subject: URGENT: Your PayPal Account Has Been Suspended!\n\n"
        "Dear Customer,\nWe detected unauthorized login attempts on your bank account. "
        "Click the link below immediately to verify your password and credentials:\n"
        "http://paypal-security-update.com/login?id=88392\n"
        "Failure to act within 24 hours will result in permanent termination of your $5,000 balance!\n"
        "Please see attached invoice: invoice_98.pdf\n"
        "<html><body><a href='http://evil.com'>Verify Now</a></body></html>"
    )

    features = extract_features_from_text(sample_email)
    print("Extracted Features for Suspicious Email:")
    for k, v in features.items():
        print(f"  {k:26}: {v}")

    print("\nFeature Metadata count:", len(FEATURE_METADATA))
