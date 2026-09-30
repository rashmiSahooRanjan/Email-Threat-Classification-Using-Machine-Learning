"""
tests/test_app_logic.py
Unit tests for Streamlit application validation logic and edge-case resilience.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction import EmailThreatPredictor, classify_email


def test_empty_input_handling():
    """Verify empty email inputs return clean error status rather than crashing."""
    model_paths = [
        "models/final_model.joblib",
        "models/decision_tree_pipeline.joblib",
        "models/logistic_regression_pipeline.joblib",
    ]
    existing = [p for p in model_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Models not yet serialized; skipping test.")

    predictor = EmailThreatPredictor(model_path=existing[0])

    # Empty string
    res1 = predictor.predict_email("", "")
    assert res1["status"] == "error"
    assert "empty" in res1["message"].lower()

    # Whitespace only
    res2 = predictor.predict_email("   \n\t  ", "  ")
    assert res2["status"] == "error"


def test_extreme_input_handling():
    """Verify system handles very long texts and special characters gracefully."""
    model_paths = [
        "models/final_model.joblib",
        "models/decision_tree_pipeline.joblib",
        "models/logistic_regression_pipeline.joblib",
    ]
    existing = [p for p in model_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Models not yet serialized; skipping test.")

    predictor = EmailThreatPredictor(model_path=existing[0])

    # Extremely long text (e.g. 50,000 characters)
    huge_text = "This is a legitimate system notification repeated. " * 1000
    res_huge = predictor.predict_email(huge_text, subject="Big Test")
    assert res_huge["status"] == "success"
    assert res_huge["prediction"] in ["ROUTINE / LEGITIMATE", "THREATENING / PHISHING"]

    # Special characters and unicode
    unicode_text = "🔒 Alert: $500,000 won! 🎁 Click https://test.xyz/lottery 💰💰💰"
    res_unicode = predictor.predict_email(unicode_text, subject="🎉 Winner!")
    assert res_unicode["status"] == "success"
