"""
tests/test_data.py
Unit and integration tests for dataset discovery and loading modules.
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    get_dataset_directory,
    list_dataset_files,
    load_dataset_file,
    load_raw_sources,
    load_combined_phishing_dataset,
    get_dataset_inventory,
)
from src.data_cleaning import clean_email_text, prepare_cleaned_dataset


def test_dataset_directory_exists():
    """Verify that dataset directory is detected and valid."""
    d_dir = get_dataset_directory()
    assert os.path.isdir(d_dir), f"Dataset directory {d_dir} does not exist."


def test_csv_files_discovered():
    """Verify all 7 expected CSV files are present."""
    files = list_dataset_files()
    assert len(files) >= 7, f"Expected at least 7 CSV files, found {len(files)}"
    fnames = [os.path.basename(f) for f in files]
    expected_files = [
        "CEAS_08.csv",
        "Enron.csv",
        "Ling.csv",
        "Nazario.csv",
        "Nigerian_Fraud.csv",
        "SpamAssasin.csv",
        "phishing_email.csv",
    ]
    for exp in expected_files:
        assert exp in fnames, f"Missing required file: {exp}"


def test_load_dataset_file():
    """Verify single file loading works and returns DataFrame."""
    d_dir = get_dataset_directory()
    sample_path = os.path.join(d_dir, "Ling.csv")
    df = load_dataset_file(sample_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "label" in df.columns


def test_load_raw_sources():
    """Verify consolidated raw sources loading contains standard columns."""
    df = load_raw_sources()
    assert isinstance(df, pd.DataFrame)
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "label" in df.columns
    assert "source_file" in df.columns
    assert len(df) == 82486


def test_clean_email_text_handles_edge_cases():
    """Verify text cleaning handles nulls, whitespace, and null bytes."""
    assert clean_email_text(None) == ""
    assert clean_email_text(float("nan")) == ""
    assert clean_email_text("  hello world  \r\n") == "hello world"
    assert clean_email_text("safe\x00text") == "safetext"


def test_prepare_cleaned_dataset():
    """Verify dataset sanitization removes empty strings and ensures binary labels."""
    df = prepare_cleaned_dataset(drop_exact_duplicates=True)
    assert not df.empty
    assert set(df["label"].unique()).issubset({0, 1})
    assert (df["text_combined"].str.len() == 0).sum() == 0
