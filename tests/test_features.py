"""
tests/test_features.py
Unit tests for feature engineering, regular expressions, and edge cases.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    extract_features_from_text,
    extract_features_dataframe,
)


def test_feature_columns_count():
    """Verify exactly 20 engineered features are defined and documented."""
    assert len(FEATURE_COLUMNS) == 20
    assert len(FEATURE_METADATA) == 20
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_METADATA


def test_extract_features_empty_string():
    """Verify empty text produces 0 for counts and 0.0 for ratios without crashing."""
    feats = extract_features_from_text("")
    assert isinstance(feats, dict)
    assert len(feats) == 20
    assert feats["message_length"] == 0
    assert feats["word_count"] == 0
    assert feats["uppercase_ratio"] == 0.0
    assert feats["digit_ratio"] == 0.0
    assert feats["url_count"] == 0


def test_extract_features_suspicious_email():
    """Verify detection of URLs, keywords, urgency, currency, and uppercase."""
    text = (
        "URGENT: Click http://bank-verify.com to confirm your password! "
        "Your account balance of $5,000 will be suspended immediately! "
        "<html><body>Download attachment invoice.pdf</body></html>"
    )
    feats = extract_features_from_text(text, subject="URGENT")
    assert feats["url_count"] >= 1
    assert feats["has_url"] == 1
    assert feats["suspicious_keyword_count"] >= 3
    assert feats["urgency_keyword_count"] >= 2
    assert feats["exclamation_count"] >= 2
    assert feats["currency_symbol_count"] >= 1
    assert feats["html_indicator"] == 1
    assert feats["attachment_indicator"] == 1
    assert feats["uppercase_ratio"] > 0.0


def test_extract_features_dataframe():
    """Verify DataFrame feature extraction outputs expected schema."""
    df_sample = pd.DataFrame({
        "subject": ["Hello", "Urgent Alert"],
        "body": ["Can we meet at 2pm?", "Click http://verify.com to win $1000"]
    })
    df_feats = extract_features_dataframe(df_sample)
    assert df_feats.shape == (2, 20)
    assert list(df_feats.columns) == FEATURE_COLUMNS
