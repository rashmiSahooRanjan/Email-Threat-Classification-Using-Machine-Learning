"""
app/app.py
Streamlit web application for Email Threat Classification.

Educational capstone demonstration platform featuring:
- Email input (Subject & Body)
- Instant sample loading (Phishing, Routine, Suspicious)
- Real-time measurable feature extraction
- Binary threat prediction (Routine vs. Threatening)
- Threat probability score & risk tier assessment
- Explainable feature breakdown & indicators
- Model metadata & educational disclaimer
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    extract_features_from_text,
)
from src.prediction import EmailThreatPredictor


# Page configuration
st.set_page_config(
    page_title="Email Threat Classification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cybersecurity-themed visual aesthetics
st.markdown("""
<style>
    /* Global styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .threat-banner-red {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 6px solid #EF4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .threat-banner-green {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border-left: 6px solid #22C55E;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #F1F5F9;
        border: 1px dashed #94A3B8;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 2rem;
    }
    .feature-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-radius: 6px;
        padding: 4px 8px;
        margin: 2px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Preset email templates for testing
PRESETS = {
    "Select an example...": {
        "subject": "",
        "body": ""
    },
    "Phishing: Urgent Account Suspension": {
        "subject": "URGENT: Your Bank Account Has Been Suspended!",
        "body": (
            "Dear Valued Customer,\n\n"
            "We have detected unauthorized transactions on your account. For your security, "
            "your online access has been restricted. You must verify your identity immediately "
            "to prevent permanent account closure.\n\n"
            "Please click the secure verification link below within 24 hours:\n"
            "http://security-update-verification.bank-online-portal.com/login?id=83920\n\n"
            "Failure to act immediately will result in forfeiture of funds and a $500 penalty fee.\n\n"
            "Customer Support Security Team"
        )
    },
    "Phishing: Advance-Fee Lottery Prize": {
        "subject": "FINAL NOTICE: CLAIM YOUR $2,500,000 LOTTERY WINNINGS!",
        "body": (
            "ATTENTION WINNER!\n\n"
            "You have been selected as the grand prize winner of the International Wealth Lottery! "
            "Your ticket number matches winning code #US-98234-LOT. A certified bank draft of $2,500,000 USD "
            "is currently waiting for wire transfer.\n\n"
            "To claim your inheritance and prize money, reply immediately with your full legal name, "
            "home address, social security number (SSN), and banking details to barrister.williams@yahoo.com.\n\n"
            "Do not disclose this notification to anyone. ACT NOW before the claim expires!"
        )
    },
    "Routine: Corporate Project Sync": {
        "subject": "Weekly Sprint Planning - Agenda and Logistics",
        "body": (
            "Hi Team,\n\n"
            "Please find below the agenda for our upcoming sprint retrospective and planning session "
            "scheduled for tomorrow at 10:00 AM in Conference Room B.\n\n"
            "Agenda:\n"
            "1. Review of completed sprint deliverables\n"
            "2. Backlog grooming and velocity estimates\n"
            "3. Q4 engineering milestones update\n\n"
            "Let me know if anyone has additional discussion topics to include before end of day.\n\n"
            "Best regards,\n"
            "Sarah Jenkins\n"
            "Project Lead"
        )
    },
    "Routine: Shipping & Delivery Update": {
        "subject": "Your package has been delivered - Order #49281",
        "body": (
            "Hello Alexander,\n\n"
            "Your recent bookstore order #49281 has been delivered to your front door on Wednesday, Sept 30.\n\n"
            "Items delivered:\n"
            "- 'Designing Data-Intensive Applications' by Martin Kleppmann (Paperback)\n\n"
            "Thank you for shopping with us! If you need any assistance, you can view your order summary "
            "in your account settings.\n\n"
            "Sincerely,\n"
            "Customer Fulfillment Services"
        )
    }
}


@st.cache_resource
def get_available_models():
    """
    Discovers available model artifacts in the models directory.
    """
    models_dir = "models"
    available = {}
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib"):
                label = f.replace("_pipeline.joblib", "").replace(".joblib", "").replace("_", " ").title()
                available[label] = os.path.join(models_dir, f)
    return available


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Email Threat Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Educational Machine Learning Prototype for Threat & Phishing Detection using Measurable Characteristics</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration & Info")
        available_models = get_available_models()

        if not available_models:
            st.warning("⚠️ No trained models found in `models/`. Please execute `src/train_models.py` first.")
            selected_model_path = None
        else:
            default_index = 0
            # Prefer final_model or decision_tree as default
            keys = list(available_models.keys())
            for idx, k in enumerate(keys):
                if "Final" in k or "Decision Tree" in k:
                    default_index = idx
                    break

            selected_model_label = st.selectbox(
                "Select ML Model Pipeline:",
                options=keys,
                index=default_index,
                help="Choose which foundational ML model evaluates the email features."
            )
            selected_model_path = available_models[selected_model_label]
            st.info(f"Loaded: `{os.path.basename(selected_model_path)}`")

        st.markdown("---")
        st.subheader("📚 Project Overview")
        st.markdown(
            "This capstone project investigates **observable, measurable email characteristics** "
            "(length, URL density, keyword frequencies, capitalization ratios, punctuation, HTML indicators) "
            "to classify emails into **Routine/Legitimate** or **Threatening/Phishing** using foundational "
            "machine learning (Logistic Regression, KNN, Decision Tree)."
        )

        st.markdown("---")
        st.subheader("📊 Dataset Reference")
        st.markdown(
            "**Source**: Naser Abdullah Alam (2024)\n"
            "*Phishing Email Dataset*, Kaggle.\n"
            "DOI: [10.34740/KAGGLE/DS/5074342](https://doi.org/10.34740/KAGGLE/DS/5074342)\n"
            "82,486 total records combining CEAS_08, Enron, Ling, Nazario, Nigerian Fraud, and SpamAssassin."
        )

    # Main Area Tabs
    tab_classify, tab_features, tab_model_info = st.tabs([
        "🔍 Email Classifier",
        "📐 Measurable Features Guide",
        "📈 Model Architecture & Comparison"
    ])

    with tab_classify:
        col_preset, col_clear = st.columns([4, 1])
        with col_preset:
            preset_choice = st.selectbox(
                "Quick-Load Sample Emails:",
                options=list(PRESETS.keys()),
                index=0
            )

        # Prepopulate from preset if chosen
        current_preset = PRESETS.get(preset_choice, {"subject": "", "body": ""})

        with st.form("classification_form"):
            subject_input = st.text_input(
                "Email Subject (Optional):",
                value=current_preset["subject"],
                placeholder="e.g. URGENT: Action required on your account"
            )

            body_input = st.text_area(
                "Email Body / Message Text:",
                value=current_preset["body"],
                height=220,
                placeholder="Paste the email content here for analysis..."
            )

            col_submit, col_space = st.columns([2, 5])
            with col_submit:
                submit_btn = st.form_submit_button("🔍 Classify Email")

        # Classification execution
        if submit_btn:
            if not body_input.strip() and not subject_input.strip():
                st.warning("⚠️ Please provide an email subject or message body to classify.")
            elif selected_model_path is None:
                st.error("❌ No model artifact loaded. Run training script first.")
            else:
                try:
                    predictor = EmailThreatPredictor(model_path=selected_model_path)
                    result = predictor.predict_email(text=body_input, subject=subject_input)

                    if result["status"] == "error":
                        st.error(result["message"])
                    else:
                        st.markdown("---")
                        st.subheader("📊 Classification Result")

                        pred_label = result["prediction"]
                        is_threat = result["prediction_class"] == 1
                        prob = result["threat_probability"]
                        risk_tier = result["risk_tier"]

                        # Display prominent outcome banner
                        if is_threat:
                            st.markdown(f"""
                            <div class="threat-banner-red">
                                <h2 style="color: #991B1B; margin: 0;">🚨 THREATENING / PHISHING EMAIL DETECTED</h2>
                                <p style="color: #7F1D1D; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    Calculated Threat Probability: <strong>{prob:.1f}%</strong> — <strong>{risk_tier}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="threat-banner-green">
                                <h2 style="color: #166534; margin: 0;">✅ ROUTINE / LEGITIMATE EMAIL</h2>
                                <p style="color: #14532D; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    Calculated Threat Probability: <strong>{prob:.1f}%</strong> — <strong>{risk_tier}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Top metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Prediction", "THREAT" if is_threat else "ROUTINE")
                        with m2:
                            st.metric("Threat Probability", f"{prob:.1f}%")
                        with m3:
                            st.metric("Risk Tier", risk_tier)
                        with m4:
                            st.metric("Evaluator Model", result["model_used"].replace("_", " ").title())

                        st.progress(prob / 100.0)

                        # Observable Threat Indicators
                        st.subheader("🚩 Observable Threat Indicators Detected")
                        for ind in result["indicators"]:
                            st.markdown(f"- {ind}")

                        # Extracted features table
                        st.subheader("🔬 Extracted Measurable Characteristics (20 Features)")
                        feats = result["features"]

                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            df_f1 = pd.DataFrame([
                                {"Feature": "Message Character Length", "Value": str(feats.get("message_length", 0))},
                                {"Feature": "Subject Character Length", "Value": str(feats.get("subject_length", 0))},
                                {"Feature": "Body Character Length", "Value": str(feats.get("body_length", 0))},
                                {"Feature": "Word Count", "Value": str(feats.get("word_count", 0))},
                                {"Feature": "Hyperlink / URL Count", "Value": str(feats.get("url_count", 0))},
                                {"Feature": "Has Embedded URL", "Value": "Yes (1)" if feats.get("has_url") else "No (0)"},
                                {"Feature": "Referenced Email Addresses", "Value": str(feats.get("email_count", 0))},
                                {"Feature": "Security / Phish Keywords", "Value": str(feats.get("suspicious_keyword_count", 0))},
                                {"Feature": "Urgency / Pressure Keywords", "Value": str(feats.get("urgency_keyword_count", 0))},
                                {"Feature": "Currency Symbols ($/€/£)", "Value": str(feats.get("currency_symbol_count", 0))},
                            ])
                            st.table(df_f1)

                        with col_f2:
                            df_f2 = pd.DataFrame([
                                {"Feature": "Exclamation Marks (!)", "Value": str(feats.get("exclamation_count", 0))},
                                {"Feature": "Question Marks (?)", "Value": str(feats.get("question_mark_count", 0))},
                                {"Feature": "Uppercase Characters", "Value": str(feats.get("uppercase_count", 0))},
                                {"Feature": "Uppercase Ratio", "Value": f"{feats.get('uppercase_ratio', 0)*100:.1f}%"},
                                {"Feature": "Numeric Digits", "Value": str(feats.get("digit_count", 0))},
                                {"Feature": "Numeric Digit Ratio", "Value": f"{feats.get('digit_ratio', 0)*100:.1f}%"},
                                {"Feature": "Special Characters", "Value": str(feats.get("special_char_count", 0))},
                                {"Feature": "HTML Formatting Tags", "Value": str(feats.get("html_tag_count", 0))},
                                {"Feature": "HTML Indicator Flag", "Value": "Yes (1)" if feats.get("html_indicator") else "No (0)"},
                                {"Feature": "Attachment Reference Flag", "Value": "Yes (1)" if feats.get("attachment_indicator") else "No (0)"},
                            ])
                            st.table(df_f2)

                except Exception as e:
                    st.error(f"Error executing classification: {e}")

    with tab_features:
        st.header("📐 Measurable Features Specification")
        st.markdown(
            "In strict accordance with the capstone research direction (*'Investigate which observable "
            "email characteristics contribute to threat classification'*), the model extracts 20 objectively "
            "measurable characteristics directly from the email text and headers:"
        )

        feat_rows = []
        for name, meta in FEATURE_METADATA.items():
            feat_rows.append({
                "Feature Name": f"`{name}`",
                "Definition": meta["definition"],
                "Calculation": meta["calculation"],
                "Cybersecurity Rationale": meta["reason"],
                "Limitations": meta["limitations"],
            })
        st.table(pd.DataFrame(feat_rows))

    with tab_model_info:
        st.header("📈 Model Architecture & Empirical Performance")
        st.markdown(
            "The system evaluates three foundational supervised machine learning models: "
            "**Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Tree**."
        )

        # Check if model_comparison.csv exists
        comp_csv = "outputs/metrics/model_comparison.csv"
        if os.path.exists(comp_csv):
            st.subheader("🎯 Experimental Evaluation Metrics (Test Set)")
            df_comp = pd.read_csv(comp_csv)
            st.table(df_comp)
        else:
            st.info("Evaluation metrics will appear here once `src/evaluate_models.py` has run.")

        # Check for figures
        roc_img = "outputs/figures/roc_curve_comparison.png"
        feat_img = "outputs/figures/feature_importance.png"
        if os.path.exists(roc_img) or os.path.exists(feat_img):
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(roc_img):
                    st.image(roc_img, caption="ROC Curve Comparison")
            with c2:
                if os.path.exists(feat_img):
                    st.image(feat_img, caption="Feature Relevance Comparison")

    # Mandatory Capstone Disclaimer Box
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ ACADEMIC & RESEARCH DISCLAIMER:</strong><br>
        This application is an educational machine-learning capstone prototype developed for demonstration,
        research, and student viva examination. It is <strong>NOT</strong> a commercial or production email security system.
        It must not be utilized as the sole perimeter defense against cyber threats or live email compromise.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
