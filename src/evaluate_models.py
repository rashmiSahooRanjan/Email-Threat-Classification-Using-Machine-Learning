"""
src/evaluate_models.py
Evaluation module for Email Threat Classification.

Computes comprehensive metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix parameters: TP, TN, FP, FN
Generates confusion matrix figures and combined ROC curves.
Outputs metrics table to outputs/metrics/model_comparison.csv.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.feature_engineering import FEATURE_COLUMNS


def load_test_data(test_cache_path: str = "outputs/reports/test_split.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the reserved independent test partition.
    """
    if not os.path.exists(test_cache_path):
        raise FileNotFoundError(
            f"Test partition not found at {test_cache_path}. Run src/train_models.py first."
        )

    df_test = pd.read_csv(test_cache_path)
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test["label"].astype(int)
    return X_test, y_test


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str
) -> None:
    """
    Plots a professionally styled confusion matrix with counts and percentages.
    """
    plt.figure(figsize=(6, 5))
    total = np.sum(cm)
    labels = [
        [f"TN\n{cm[0, 0]:,}\n({cm[0, 0]/total*100:.1f}%)", f"FP\n{cm[0, 1]:,}\n({cm[0, 1]/total*100:.1f}%)"],
        [f"FN\n{cm[1, 0]:,}\n({cm[1, 0]/total*100:.1f}%)", f"TP\n{cm[1, 1]:,}\n({cm[1, 1]/total*100:.1f}%)"]
    ]

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Routine (0)", "Threat (1)"],
        yticklabels=["Routine (0)", "Threat (1)"],
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}", fontsize=13, pad=12, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


def plot_roc_comparison(
    roc_data: Dict[str, dict],
    output_path: str = "outputs/figures/roc_curve_comparison.png"
) -> None:
    """
    Plots combined ROC curves for all evaluated models on a single comparison plot.
    """
    plt.figure(figsize=(8, 6))
    colors = {
        "logistic_regression": "#1f77b4",
        "knn": "#ff7f0e",
        "decision_tree": "#2ca02c"
    }

    for name, data in roc_data.items():
        color = colors.get(name, "#333333")
        label_text = f"{data['display_name']} (AUC = {data['auc']:.4f})"
        plt.plot(data["fpr"], data["tpr"], label=label_text, color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, labelpad=8)
    plt.title("ROC Curve Comparison - Email Threat Classification Models", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC comparison curve saved: {output_path}")


def evaluate_all_models(
    models_dir: str = "models",
    metrics_output_path: str = "outputs/metrics/model_comparison.csv"
) -> pd.DataFrame:
    """
    Evaluates all trained model pipelines on the reserved test dataset.
    Saves metrics table, individual confusion matrices, and ROC comparison curve.
    """
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    X_test, y_test = load_test_data()

    models_to_evaluate = [
        ("logistic_regression", "Logistic Regression"),
        ("knn", "K-Nearest Neighbors"),
        ("decision_tree", "Decision Tree"),
    ]

    metrics_records = []
    roc_data = {}

    for model_key, display_name in models_to_evaluate:
        pipeline_path = os.path.join(models_dir, f"{model_key}_pipeline.joblib")
        if not os.path.exists(pipeline_path):
            print(f"Warning: Pipeline {pipeline_path} not found. Skipping.")
            continue

        print(f"\nEvaluating {display_name}...")
        pipeline = joblib.load(pipeline_path)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Probabilities
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[model_key] = {
                "display_name": display_name,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr
            }
        else:
            auc = np.nan
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cm_fig_path = f"outputs/figures/confusion_matrix_{model_key}.png"
        plot_confusion_matrix(cm, display_name, cm_fig_path)

        record = {
            "Model": display_name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4) if not np.isnan(auc) else "N/A",
            "True_Positive": int(tp),
            "True_Negative": int(tn),
            "False_Positive": int(fp),
            "False_Negative": int(fn),
        }
        metrics_records.append(record)

        print(f"  Accuracy : {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc:.4f}")
        print(f"  TP: {tp:,} | TN: {tn:,} | FP: {fp:,} | FN: {fn:,}")

    # Plot combined ROC
    if roc_data:
        plot_roc_comparison(roc_data)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"\nModel comparison table saved to: {metrics_output_path}")
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(metrics_df.to_string(index=False))

    return metrics_df


if __name__ == "__main__":
    print("Running Model Evaluation Suite...")
    evaluate_all_models()
