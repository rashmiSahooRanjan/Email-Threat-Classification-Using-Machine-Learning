"""
src/train_models.py
Model training pipeline for Email Threat Classification.

Trains foundational machine learning models:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
Saves trained Scikit-learn pipelines with embedded preprocessors to models/.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from src.data_cleaning import prepare_cleaned_dataset
from src.feature_engineering import (
    FEATURE_COLUMNS,
    extract_features_dataframe,
)
from src.preprocessing import create_preprocessor, split_data


def load_or_extract_features(
    cache_path: str = "outputs/reports/engineered_features.csv",
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads pre-extracted engineered features from disk, or computes them
    from the cleaned raw dataset if cache does not exist.
    """
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading existing engineered features from: {cache_path}")
        df_feat = pd.read_csv(cache_path)
        y = df_feat["label"].astype(int)
        X = df_feat[FEATURE_COLUMNS]
        print(f"Loaded {len(X):,} feature rows with {len(FEATURE_COLUMNS)} columns.")
        return X, y

    print("Extracting engineered features from cleaned raw dataset...")
    df_clean = prepare_cleaned_dataset(drop_exact_duplicates=True)
    t0 = time.time()
    X = extract_features_dataframe(df_clean)
    y = df_clean["label"].astype(int).reset_index(drop=True)
    t1 = time.time()
    print(f"Feature extraction complete in {t1 - t0:.1f}s.")

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    export_df = X.copy()
    export_df["label"] = y
    export_df.to_csv(cache_path, index=False)
    print(f"Features saved to cache: {cache_path}")

    return X, y


def build_model_pipelines() -> Dict[str, Pipeline]:
    """
    Builds Scikit-learn pipelines for the foundational models.
    Each pipeline couples the ColumnTransformer preprocessor with an estimator.
    """
    preprocessor = create_preprocessor(scaler_type="standard")

    # Model 1: Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Model 2: K-Nearest Neighbors
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ))
    ])

    # Model 3: Decision Tree
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    return {
        "logistic_regression": lr_pipeline,
        "knn": knn_pipeline,
        "decision_tree": dt_pipeline,
    }


def train_and_save_all_models(
    models_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """
    Executes the complete training workflow:
    1. Loads features and target.
    2. Performs stratified train/test split.
    3. Saves test partition for independent evaluation.
    4. Fits all 3 foundational pipelines strictly on training data.
    5. Saves serialized pipeline artifacts (.joblib).

    Returns:
        Dictionary of trained pipelines.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    X, y = load_or_extract_features()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=random_state)

    # Cache split for reproducible evaluation
    split_cache_path = "outputs/reports/test_split.csv"
    test_export = X_test.copy()
    test_export["label"] = y_test
    test_export.to_csv(split_cache_path, index=False)
    print(f"Test partition ({len(X_test):,} rows) cached to: {split_cache_path}")

    pipelines = build_model_pipelines()
    trained_pipelines = {}

    for name, pipe in pipelines.items():
        print(f"\nTraining Model: {name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"Model {name} trained in {elapsed:.2f} seconds.")

        # Save pipeline
        save_path = os.path.join(models_dir, f"{name}_pipeline.joblib")
        joblib.dump(pipe, save_path)
        print(f"Saved: {save_path}")
        trained_pipelines[name] = pipe

    print("\nAll foundational models successfully trained and serialized.")
    return trained_pipelines


if __name__ == "__main__":
    print("Starting Model Training Pipeline...")
    train_and_save_all_models()
