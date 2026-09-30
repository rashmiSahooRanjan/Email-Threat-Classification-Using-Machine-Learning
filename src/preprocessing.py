"""
src/preprocessing.py
Data preprocessing and train/test splitting module for Email Threat Classification.

Implements strict data leakage prevention by ensuring all transformers
are fitted strictly on training data after an 80/20 stratified split.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.feature_engineering import FEATURE_COLUMNS

# Continuous / count features that benefit from standard scaling
CONTINUOUS_FEATURES = [
    "message_length",
    "subject_length",
    "body_length",
    "word_count",
    "url_count",
    "email_count",
    "suspicious_keyword_count",
    "urgency_keyword_count",
    "exclamation_count",
    "question_mark_count",
    "uppercase_count",
    "uppercase_ratio",
    "digit_count",
    "digit_ratio",
    "special_char_count",
    "currency_symbol_count",
    "html_tag_count",
]

# Discrete binary indicator features
BINARY_FEATURES = [
    "has_url",
    "html_indicator",
    "attachment_indicator",
]


def create_preprocessor(scaler_type: str = "standard") -> ColumnTransformer:
    """
    Creates a Scikit-Learn ColumnTransformer preprocessor.

    - Imputes missing numerical values with median (leakage-safe).
    - Scales continuous features with StandardScaler or RobustScaler.
    - Passes binary indicators through cleanly.

    Args:
        scaler_type: 'standard' for StandardScaler, 'robust' for RobustScaler.

    Returns:
        Configured ColumnTransformer object.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix X and target y into training and testing partitions
    using stratified sampling to maintain class proportions.

    Args:
        X: Feature DataFrame (20 engineered features).
        y: Binary target series (0 or 1).
        test_size: Proportion of dataset allocated to evaluation (default: 0.20).
        random_state: Deterministic random seed for full reproducibility (default: 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Data Split Summary:")
    print(f"  Training set size : {len(X_train):,} samples ({round((1 - test_size) * 100)}%)")
    print(f"  Test set size     : {len(X_test):,} samples ({round(test_size * 100)}%)")
    print(f"  Train class counts: {dict(y_train.value_counts())}")
    print(f"  Test class counts : {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Testing preprocessing pipeline creation...")
    preprocessor = create_preprocessor()
    print("ColumnTransformer successfully initialized with stages:")
    for name, pipe, cols in preprocessor.transformers:
        print(f"  Stage '{name}': {len(cols)} columns -> {cols[:3]}...")
