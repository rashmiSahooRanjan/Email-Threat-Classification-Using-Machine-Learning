"""
tests/test_model.py
Integration tests for model pipelines, serialization, and predictions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import create_preprocessor, split_data
from src.feature_engineering import FEATURE_COLUMNS
from src.prediction import EmailThreatPredictor


def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms feature matrices correctly."""
    preprocessor = create_preprocessor()
    dummy_data = pd.DataFrame(np.ones((10, 20)), columns=FEATURE_COLUMNS)
    transformed = preprocessor.fit_transform(dummy_data)
    assert transformed.shape == (10, 20)


def test_split_data_reproducibility():
    """Verify split_data is stratified and deterministic."""
    dummy_X = pd.DataFrame(np.random.randn(100, 20), columns=FEATURE_COLUMNS)
    dummy_y = pd.Series([0] * 60 + [1] * 40)
    X_train1, X_test1, y_train1, y_test1 = split_data(dummy_X, dummy_y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(dummy_X, dummy_y, test_size=0.2, random_state=42)

    assert len(X_train1) == 80
    assert len(X_test1) == 20
    assert y_test1.sum() == 8  # 40% of 20
    assert X_train1.equals(X_train2)


def test_model_loading_and_prediction():
    """Verify trained pipeline can be loaded and outputs valid prediction structure."""
    model_paths = [
        "models/final_model.joblib",
        "models/decision_tree_pipeline.joblib",
        "models/logistic_regression_pipeline.joblib",
    ]
    existing = [p for p in model_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Models not yet serialized; skipping live model test.")

    predictor = EmailThreatPredictor(model_path=existing[0])
    res = predictor.predict_email("Subject: Team Meeting\n\nLet's meet tomorrow at 10am.")
    assert res["status"] == "success"
    assert res["prediction"] in ["ROUTINE / LEGITIMATE", "THREATENING / PHISHING"]
    assert 0.0 <= res["threat_probability"] <= 100.0
    assert "features" in res
