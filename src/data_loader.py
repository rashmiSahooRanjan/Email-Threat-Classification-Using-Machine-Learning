"""
src/data_loader.py
Data loading module for Email Threat Classification.

Handles discovery, validation, and loading of raw dataset files
from the local dataset directory using relative paths.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd


def get_dataset_directory(base_dir: Optional[str] = None) -> str:
    """
    Locates the dataset directory using case-tolerant relative paths.

    Args:
        base_dir: Optional base directory to search from. Defaults to project root.

    Returns:
        Path to the dataset directory.

    Raises:
        FileNotFoundError: If neither 'dataset' nor 'Dataset' is found.
    """
    if base_dir is None:
        # Determine project root relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, ".."))

    candidate_paths = [
        os.path.join(base_dir, "dataset"),
        os.path.join(base_dir, "Dataset"),
        os.path.join(".", "dataset"),
        os.path.join(".", "Dataset"),
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Dataset directory not found in candidate paths: {candidate_paths}. "
        "Please ensure the 'dataset' or 'Dataset' directory is present in the project root."
    )


def list_dataset_files(dataset_dir: Optional[str] = None) -> List[str]:
    """
    Discovers all CSV files available in the dataset directory.

    Args:
        dataset_dir: Path to dataset folder. If None, auto-detected.

    Returns:
        Sorted list of absolute file paths to CSV files.
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_directory()

    pattern = os.path.join(dataset_dir, "*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {dataset_dir}")

    return sorted(csv_files)


def load_dataset_file(file_path: str) -> pd.DataFrame:
    """
    Safely loads a single CSV dataset file into a pandas DataFrame.

    Args:
        file_path: Relative or absolute path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}") from e


def load_raw_sources(dataset_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and standardizes the 6 individual source datasets:
    - CEAS_08.csv
    - Enron.csv
    - Ling.csv
    - Nazario.csv
    - Nigerian_Fraud.csv
    - SpamAssasin.csv

    Standardizes columns to ['source_file', 'subject', 'body', 'label'].

    Returns:
        Consolidated DataFrame with all 82,486 raw emails containing
        original casing, punctuation, and structural indicators.
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_directory()

    source_names = [
        "CEAS_08.csv",
        "Enron.csv",
        "Ling.csv",
        "Nazario.csv",
        "Nigerian_Fraud.csv",
        "SpamAssasin.csv",
    ]

    dfs = []
    for name in source_names:
        file_path = os.path.join(dataset_dir, name)
        if not os.path.exists(file_path):
            continue

        df_source = load_dataset_file(file_path)
        df_standard = pd.DataFrame()
        df_standard["source_file"] = [name] * len(df_source)

        # Standardize subject
        if "subject" in df_source.columns:
            df_standard["subject"] = df_source["subject"].fillna("").astype(str)
        else:
            df_standard["subject"] = ""

        # Standardize body
        if "body" in df_source.columns:
            df_standard["body"] = df_source["body"].fillna("").astype(str)
        else:
            df_standard["body"] = ""

        # Standardize label
        if "label" in df_source.columns:
            df_standard["label"] = pd.to_numeric(df_source["label"], errors="coerce").fillna(0).astype(int)
        else:
            raise KeyError(f"Target column 'label' missing from source file {name}")

        dfs.append(df_standard)

    if not dfs:
        raise FileNotFoundError(f"None of the expected source files {source_names} were found in {dataset_dir}")

    consolidated_df = pd.concat(dfs, ignore_index=True)
    return consolidated_df


def load_combined_phishing_dataset(dataset_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the aggregate 'phishing_email.csv' dataset file.

    Returns:
        DataFrame containing ['text_combined', 'label'].
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_directory()

    file_path = os.path.join(dataset_dir, "phishing_email.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'phishing_email.csv' not found in {dataset_dir}")

    df = load_dataset_file(file_path)
    if "text_combined" not in df.columns or "label" not in df.columns:
        raise KeyError(f"Expected columns ['text_combined', 'label'] missing from {file_path}")

    return df


def get_dataset_inventory(dataset_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Generates a full metadata inventory for all CSV files in the dataset folder.

    Returns:
        Dictionary mapping filename to metadata (rows, cols, nulls, distribution).
    """
    files = list_dataset_files(dataset_dir)
    inventory = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        df = load_dataset_file(fpath)
        label_dist = (
            df["label"].value_counts(dropna=False).to_dict()
            if "label" in df.columns
            else {}
        )
        inventory[fname] = {
            "file_name": fname,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
        }

    return inventory


if __name__ == "__main__":
    print("Testing data_loader.py...")
    d_dir = get_dataset_directory()
    print(f"Dataset directory located: {d_dir}")
    inv = get_dataset_inventory(d_dir)
    for k, v in inv.items():
        print(f"-> {k}: {v['rows']} rows, {v['columns']} cols, labels={v['label_distribution']}")
    print("Data loading verification complete.")
