#!/usr/bin/env python3
"""
Train a simple sentiment model (TF-IDF + Logistic Regression) from a CSV.

Expected input CSV columns:
- text: the raw text
- label: sentiment label (positive/negative OR 1/0)

Outputs saved to:
- models/sentiment_pipeline.joblib
- models/label_map.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Config:
    data_path: Path
    text_col: str
    label_col: str
    test_size: float
    random_state: int
    max_features: int
    ngram_max: int
    model_out: Path
    label_map_out: Path


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train a sentiment analysis model from CSV.")
    p.add_argument("--data", default="data/train.csv", help="Path to training CSV.")
    p.add_argument("--text-col", default="text", help="Name of the text column.")
    p.add_argument("--label-col", default="label", help="Name of the label column.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--max-features", type=int, default=50000, help="TF-IDF vocab size.")
    p.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size.")
    p.add_argument("--model-out", default="models/sentiment_pipeline.joblib", help="Output model path.")
    p.add_argument("--label-map-out", default="models/label_map.json", help="Output label map path.")
    a = p.parse_args()

    return Config(
        data_path=Path(a.data),
        text_col=a.text_col,
        label_col=a.label_col,
        test_size=a.test_size,
        random_state=a.seed,
        max_features=a.max_features,
        ngram_max=a.ngram_max,
        model_out=Path(a.model_out),
        label_map_out=Path(a.label_map_out),
    )


def normalize_labels(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Convert labels to {0,1}. Accepts:
    - 0/1 already
    - "negative"/"positive"
    - "neg"/"pos"
    - True/False
    """
    # If numeric 0/1 already
    if pd.api.types.is_numeric_dtype(y):
        y_int = y.astype(int)
        uniq = sorted(y_int.unique().tolist())
        if set(uniq).issubset({0, 1}):
            return y_int, {"negative": 0, "positive": 1}
        raise ValueError(f"Numeric labels must be 0/1; found values: {uniq}")

    y_str = y.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0, "1": 1,
        "false": 0, "true": 1,
        "negative": 0, "neg": 0, "bad": 0,
        "positive": 1, "pos": 1, "good": 1,
    }

    unknown = sorted(set(y_str.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(
            f"Unknown label values: {unknown}. "
            "Use 0/1 or positive/negative (or neg/pos)."
        )

    y_int = y_str.map(mapping).astype(int)
    return y_int, {"negative": 0, "positive": 1}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = parse_args()

    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {cfg.data_path}. "
            "Create data/train.csv with columns text,label."
        )

    df = pd.read_csv(cfg.data_path)
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{cfg.text_col}' and '{cfg.label_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Basic cleaning
    df = df[[cfg.text_col, cfg.label_col]].dropna()
    df[cfg.text_col] = df[cfg.text_col].astype(str)

    y, label_map = normalize_labels(df[cfg.label_col])
    X = df[cfg.text_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=(1, cfg.ngram_max),
                lowercase=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                n_jobs=1,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Accuracy: {acc:.4f}\n")
    print("📊 Classification report:")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    # Save artifacts
    ensure_parent_dir(cfg.model_out)
    ensure_parent_dir(cfg.label_map_out)

    joblib.dump(pipeline, cfg.model_out)
    cfg.label_map_out.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print(f"\n💾 Saved model to: {cfg.model_out}")
    print(f"💾 Saved label map to: {cfg.label_map_out}\n")


if __name__ == "__main__":
    # Make sklearn a bit more deterministic for some environments
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()

