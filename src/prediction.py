"""
src/prediction.py
Inference and prediction module for Email Threat Classification.

Provides unified inference pipeline for raw email text, extracting engineered
features and serving predictions via the serialized final model pipeline.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    extract_features_from_text,
)


class EmailThreatPredictor:
    """
    Production-grade inference wrapper for Email Threat Classification.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes predictor by loading the trained pipeline artifact.

        Args:
            model_path: Path to serialized pipeline (.joblib). If None, defaults
                        to 'models/final_model.joblib' or 'models/decision_tree_pipeline.joblib'.
        """
        if model_path is None:
            candidate_paths = [
                "models/final_model.joblib",
                "models/decision_tree_pipeline.joblib",
                "models/logistic_regression_pipeline.joblib",
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    model_path = cp
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please train models first using src/train_models.py."
            )

        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace(".joblib", "")
        self.pipeline = joblib.load(model_path)

    def predict_email(
        self,
        text: str,
        subject: str = ""
    ) -> Dict[str, Any]:
        """
        Classifies an email message given raw text and optional subject.

        Args:
            text: Body text or complete email string.
            subject: Optional email subject line.

        Returns:
            Dictionary containing prediction label, numeric class, threat probability,
            risk tier, extracted features, and key risk indicators.
        """
        # Input validation
        clean_text = "" if text is None else str(text).strip()
        clean_subj = "" if subject is None else str(subject).strip()

        if len(clean_text) == 0 and len(clean_subj) == 0:
            return {
                "status": "error",
                "message": "Input email content is completely empty. Please enter an email subject or body.",
                "prediction": "UNKNOWN",
                "threat_probability": 0.0,
                "risk_tier": "None",
                "features": {},
                "indicators": [],
            }

        # 1. Feature extraction
        features_dict = extract_features_from_text(
            text=clean_text,
            subject=clean_subj,
            body=clean_text
        )

        # 2. Prepare single-row DataFrame with exact columns
        df_row = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        # 3. Model inference
        pred_class = int(self.pipeline.predict(df_row)[0])

        # 4. Probability estimation
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_row)[0]
            threat_prob = float(probs[1])
        else:
            threat_prob = 1.0 if pred_class == 1 else 0.0

        label_str = "THREATENING / PHISHING" if pred_class == 1 else "ROUTINE / LEGITIMATE"

        # Risk tier categorisation
        if threat_prob >= 0.80:
            risk_tier = "Critical Risk"
            risk_color = "#d9534f"
        elif threat_prob >= 0.60:
            risk_tier = "High Risk"
            risk_color = "#f0ad4e"
        elif threat_prob >= 0.40:
            risk_tier = "Moderate Risk"
            risk_color = "#eea236"
        else:
            risk_tier = "Low / Routine"
            risk_color = "#5cb85c"

        # Key observed threat indicators
        indicators = []
        if features_dict["url_count"] > 0:
            indicators.append(f"Detected {features_dict['url_count']} embedded hyperlink(s)")
        if features_dict["suspicious_keyword_count"] > 0:
            indicators.append(f"Found {features_dict['suspicious_keyword_count']} security/credential keyword(s)")
        if features_dict["urgency_keyword_count"] > 0:
            indicators.append(f"Detected {features_dict['urgency_keyword_count']} psychological urgency trigger(s)")
        if features_dict["uppercase_ratio"] > 0.15:
            indicators.append(f"Elevated uppercase ratio ({features_dict['uppercase_ratio']*100:.1f}%)")
        if features_dict["html_indicator"] == 1:
            indicators.append(f"HTML rendering structure present ({features_dict['html_tag_count']} tags)")
        if features_dict["attachment_indicator"] == 1:
            indicators.append("Attachment reference or suspicious file extension detected")
        if features_dict["currency_symbol_count"] > 0:
            indicators.append(f"Found {features_dict['currency_symbol_count']} financial currency symbol(s)")

        if not indicators:
            indicators.append("No prominent threat indicators detected in text structure.")

        return {
            "status": "success",
            "prediction": label_str,
            "prediction_class": pred_class,
            "threat_probability": round(threat_prob * 100, 2),
            "threat_probability_fraction": round(threat_prob, 4),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "model_used": self.model_name,
            "features": features_dict,
            "indicators": indicators,
        }


def classify_email(
    text: str,
    subject: str = "",
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience functional interface for single email classification.
    """
    predictor = EmailThreatPredictor(model_path=model_path)
    return predictor.predict_email(text=text, subject=subject)


if __name__ == "__main__":
    print("Testing src/prediction.py...")
    # Note: Requires models to be trained first
    try:
        res = classify_email("Urgent: Click here to verify your banking password! http://phish.com")
        print(f"Prediction: {res['prediction']} ({res['threat_probability']}%)")
    except Exception as e:
        print(f"Prediction test notice: {e} (run train_models.py to create models)")
