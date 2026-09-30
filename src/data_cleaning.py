"""
src/data_cleaning.py
Data quality analysis, cleaning, and sanitization module.

Generates the required data quality report and cleans raw email inputs
to ensure stability during feature extraction and modeling.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data_loader import (
    get_dataset_directory,
    load_raw_sources,
    load_combined_phishing_dataset,
    list_dataset_files,
    load_dataset_file,
)


def analyze_data_quality(df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Performs comprehensive data-quality analysis on a DataFrame:
    1. Shape
    2. Column names
    3. Data types
    4. Missing values
    5. Missing-value percentage
    6. Duplicate rows
    7. Empty strings
    8. Invalid values
    9. Infinite values
    10. Unique values
    11. Target classes
    12. Class distribution

    Returns:
        DataFrame summarizing quality metrics per column and overall.
    """
    rows, cols = df.shape
    total_duplicates = int(df.duplicated().sum())
    duplicate_pct = round((total_duplicates / rows) * 100, 3) if rows > 0 else 0.0

    report_rows = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / rows) * 100, 3) if rows > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))

        # Check empty string
        empty_str_count = 0
        if series.dtype == "object":
            empty_str_count = int((series.astype(str).str.strip() == "").sum())

        # Check infinite values
        inf_count = 0
        if pd.api.types.is_numeric_dtype(series.dtype):
            inf_count = int(np.isinf(series).sum())

        dtype_str = str(series.dtype)

        # Target class distribution if target column
        class_dist_str = "N/A"
        if col == "label":
            counts = series.value_counts(dropna=False).to_dict()
            class_dist_str = "; ".join([f"Class {k}: {v} ({round(v/rows*100, 2)}%)" for k, v in counts.items()])

        report_rows.append({
            "dataset_name": dataset_name,
            "column_name": col,
            "data_type": dtype_str,
            "total_rows": rows,
            "missing_count": null_count,
            "missing_pct": null_pct,
            "empty_string_count": empty_str_count,
            "infinite_count": inf_count,
            "unique_values": n_unique,
            "total_dataset_duplicates": total_duplicates,
            "duplicate_pct": duplicate_pct,
            "class_distribution": class_dist_str,
        })

    return pd.DataFrame(report_rows)


def generate_comprehensive_quality_report(output_path: str = "outputs/reports/data_quality_report.csv") -> pd.DataFrame:
    """
    Analyzes all files in the dataset folder plus the consolidated dataset,
    and saves the combined data quality report to outputs/reports/data_quality_report.csv.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_files = list_dataset_files()

    all_reports = []
    for fpath in dataset_files:
        fname = os.path.basename(fpath)
        df = load_dataset_file(fpath)
        df_report = analyze_data_quality(df, dataset_name=fname)
        all_reports.append(df_report)

    # Also analyze consolidated raw sources
    df_raw_sources = load_raw_sources()
    df_report_raw = analyze_data_quality(df_raw_sources, dataset_name="Consolidated_Raw_Sources")
    all_reports.append(df_report_raw)

    combined_report = pd.concat(all_reports, ignore_index=True)
    combined_report.to_csv(output_path, index=False)
    print(f"Data quality report saved to: {output_path}")
    return combined_report


def clean_email_text(text: Any) -> str:
    """
    Cleans raw email text:
    - Replaces null/NaN with empty string
    - Strips leading/trailing whitespace
    - Removes null bytes
    - Normalizes Windows carriage returns
    """
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    # Remove null bytes which can cause regex/C-string crashes
    text_str = text_str.replace("\x00", "")
    text_str = text_str.replace("\r\n", "\n").replace("\r", "\n")
    return text_str.strip()


def prepare_cleaned_dataset(drop_exact_duplicates: bool = True) -> pd.DataFrame:
    """
    Loads raw sources, cleans text fields, handles missing values,
    and returns a sanitized DataFrame ready for feature engineering.

    Args:
        drop_exact_duplicates: Whether to drop exact duplicates of (subject + body).
            Default is True for rigorous evaluation without duplicate leakage.

    Returns:
        Cleaned DataFrame with columns ['subject', 'body', 'text_combined', 'label', 'source_file'].
    """
    df = load_raw_sources()

    # Clean text columns
    df["subject"] = df["subject"].apply(clean_email_text)
    df["body"] = df["body"].apply(clean_email_text)

    # Create full combined text
    df["text_combined"] = df.apply(
        lambda r: f"{r['subject']} {r['body']}".strip() if r["subject"] else r["body"],
        axis=1
    )

    # Ensure label is strictly integer binary (0 or 1)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    # Filter out records where both subject and body are completely empty
    non_empty_mask = df["text_combined"].str.len() > 0
    df = df[non_empty_mask].copy()

    initial_count = len(df)
    if drop_exact_duplicates:
        df = df.drop_duplicates(subset=["text_combined", "label"]).reset_index(drop=True)
        dropped = initial_count - len(df)
        print(f"Sanitization: Removed {dropped} exact duplicate records ({len(df)} remaining).")
    else:
        print(f"Sanitization: Retained all {len(df)} records without deduplication.")

    return df


if __name__ == "__main__":
    print("Running data quality analysis...")
    report = generate_comprehensive_quality_report()
    print("Report sample:")
    print(report[["dataset_name", "column_name", "missing_count", "unique_values"]].head(10))
