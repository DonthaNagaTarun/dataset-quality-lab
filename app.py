"""AI-Powered Dataset Quality Assessment and Its Impact on Machine Learning Model Performance.

Run locally with: streamlit run app.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_classification


APP_TITLE = (
    "AI-Powered Dataset Quality Assessment and Its Impact on "
    "Machine Learning Model Performance"
)
RANDOM_STATE = 42

st.set_page_config(page_title="Dataset Quality Assessment", page_icon="📊", layout="wide")
px.defaults.template = "plotly_dark"

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #f4f7ff;
        --muted: #9aa8c4;
        --line: rgba(151, 169, 207, 0.14);
        --panel: rgba(16, 24, 43, 0.86);
        --blue: #5b8cff;
        --violet: #a78bfa;
    }
    .stApp {
        background: #080d19;
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(70rem 34rem at 8% -12%, rgba(37, 99, 235, 0.21), transparent 55%),
            radial-gradient(58rem 34rem at 104% -10%, rgba(124, 58, 237, 0.19), transparent 51%),
            linear-gradient(180deg, #0a1020 0%, #080d19 55%, #0a1020 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(8, 13, 25, 0.35);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-top: 1.65rem;
        padding-bottom: 3.25rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1427 0%, #0a1020 58%, #0b1020 100%);
        border-right: 1px solid rgba(151, 169, 207, 0.12);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }
    [data-testid="stSidebar"] h2 {
        color: #eef3ff;
        letter-spacing: -0.025em;
    }
    h1 {
        background: linear-gradient(100deg, #f6f8ff 5%, #c4d1ff 49%, #9b8cff 100%);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent !important;
        font-size: clamp(1.8rem, 3.25vw, 2.7rem) !important;
        font-weight: 740 !important;
        letter-spacing: -0.055em !important;
        line-height: 1.08 !important;
        max-width: 1100px;
        padding-bottom: 0.35rem;
    }
    h2, h3 {
        color: #edf2ff;
        letter-spacing: -0.035em;
    }
    p, label, [data-testid="stMarkdownContainer"] {
        color: #c4cde0;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    .hero-kicker, .sidebar-kicker {
        align-items: center;
        color: #91a1c2;
        display: flex;
        font-size: 0.69rem;
        font-weight: 750;
        gap: 0.55rem;
        letter-spacing: 0.15em;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
    }
    .hero-kicker::before {
        background: linear-gradient(135deg, var(--blue), var(--violet));
        border-radius: 999px;
        content: "";
        height: 0.48rem;
        width: 0.48rem;
        box-shadow: 0 0 18px rgba(124, 92, 255, 0.8);
    }
    .hero-kicker span {
        border: 1px solid rgba(144, 163, 205, 0.2);
        border-radius: 999px;
        color: #a9b7d4;
        font-size: 0.62rem;
        letter-spacing: 0.1em;
        margin-left: 0.2rem;
        padding: 0.24rem 0.55rem;
    }
    .dataset-strip {
        align-items: center;
        background: linear-gradient(110deg, rgba(20, 31, 54, 0.92), rgba(22, 23, 48, 0.84));
        border: 1px solid rgba(130, 151, 197, 0.16);
        border-radius: 15px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16);
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 1rem;
        margin: 1.35rem 0 1.15rem;
        padding: 0.82rem 1rem;
    }
    .dataset-strip__dot {
        background: #43d6a1;
        border-radius: 50%;
        box-shadow: 0 0 12px rgba(67, 214, 161, 0.65);
        flex: 0 0 0.48rem;
        height: 0.48rem;
    }
    .dataset-strip__label {
        color: #93a2c1;
        font-size: 0.62rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .dataset-strip__name {
        color: #f1f5ff;
        font-size: 0.9rem;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    .dataset-strip__meta {
        color: #a5b2cc;
        font-size: 0.78rem;
        margin-left: auto;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 31, 55, 0.96), rgba(13, 20, 37, 0.96));
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.16);
        min-height: 6.3rem;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #a6b3cd !important;
        font-size: 0.69rem !important;
        font-weight: 720 !important;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: #f5f7ff !important;
        font-size: clamp(1.25rem, 2vw, 1.8rem) !important;
        font-weight: 720 !important;
        letter-spacing: -0.04em;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13, 20, 37, 0.82);
        border: 1px solid var(--line);
        border-radius: 15px;
        gap: 0.3rem;
        padding: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #a0aec9;
        font-size: 0.83rem;
        font-weight: 650;
        height: 2.65rem;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(105deg, rgba(59, 130, 246, 0.26), rgba(139, 92, 246, 0.27));
        color: #f5f7ff;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-testid="stVerticalBlock"] {
        gap: 1rem;
    }
    .stButton button, .stDownloadButton button {
        border: 1px solid rgba(125, 148, 205, 0.28);
        border-radius: 11px;
        font-weight: 650;
        transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(110deg, #367cf6, #8055e8);
        border: 0;
        box-shadow: 0 7px 22px rgba(73, 93, 237, 0.25);
        color: white;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color: rgba(151, 174, 237, 0.58);
        box-shadow: 0 8px 24px rgba(49, 76, 156, 0.2);
        transform: translateY(-1px);
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: rgba(15, 23, 42, 0.92);
        border-color: rgba(135, 153, 193, 0.23);
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(15, 23, 42, 0.58);
        border: 1px dashed rgba(139, 156, 199, 0.4);
        border-radius: 14px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] {
        background: rgba(20, 31, 55, 0.8);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    .comparison-insight {
        background: linear-gradient(105deg, rgba(35, 74, 143, 0.24), rgba(96, 58, 161, 0.22));
        border: 1px solid rgba(132, 145, 223, 0.24);
        border-radius: 14px;
        color: #dfe7ff;
        margin: 0.25rem 0 0.6rem;
        padding: 0.85rem 1rem;
    }
    .comparison-insight strong {
        color: #fff;
        margin-right: 0.6rem;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        h1 {
            font-size: 1.75rem !important;
        }
        .dataset-strip__meta {
            margin-left: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.72rem;
            padding: 0 0.55rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric DataFrame columns using bounds learned from training data."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IQRClipper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper expects a pandas DataFrame.")
        self.columns_ = list(X.columns)
        self.bounds_: dict[str, tuple[float, float]] = {}
        for column in self.columns_:
            values = pd.to_numeric(X[column], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            if pd.notna(q1) and pd.notna(q3):
                spread = q3 - q1
                self.bounds_[column] = (
                    float(q1 - self.factor * spread),
                    float(q3 + self.factor * spread),
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "bounds_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper expects a pandas DataFrame.")
        result = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce").clip(
                    lower=lower, upper=upper
                )
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = self.columns_ if input_features is None else input_features
        return np.asarray(names, dtype=object)


def make_synthetic_data(
    rows: int,
    feature_count: int,
    class_count: int,
    imbalance: str,
    missing_percent: int,
    duplicate_percent: int,
    outlier_percent: int,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible classification dataset with controllable quality issues."""
    weights = None
    if imbalance == "Moderate":
        weights = [0.75, 0.25] if class_count == 2 else [0.60, 0.25, 0.15]
    elif imbalance == "High":
        weights = [0.90, 0.10] if class_count == 2 else [0.75, 0.18, 0.07]

    informative = min(feature_count, max(2, int(np.ceil(np.log2(class_count))) + 1))
    X, y = make_classification(
        n_samples=rows,
        n_features=feature_count,
        n_informative=informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=class_count,
        n_clusters_per_class=1,
        weights=weights,
        class_sep=1.0,
        random_state=seed,
    )
    frame = pd.DataFrame(
        X, columns=[f"feature_{index + 1}" for index in range(feature_count)]
    )
    frame["target"] = pd.Series(y).map(lambda label: f"Class {label}")
    rng = np.random.default_rng(seed)

    if outlier_percent:
        cell_count = max(1, int(rows * feature_count * outlier_percent / 100))
        row_indexes = rng.integers(0, rows, size=cell_count)
        column_indexes = rng.integers(0, feature_count, size=cell_count)
        for row_index, column_index in zip(row_indexes, column_indexes):
            column = frame.columns[column_index]
            frame.loc[row_index, column] += rng.choice([-1, 1]) * rng.uniform(7, 12)

    if missing_percent:
        cell_count = max(1, int(rows * feature_count * missing_percent / 100))
        row_indexes = rng.integers(0, rows, size=cell_count)
        column_indexes = rng.integers(0, feature_count, size=cell_count)
        for row_index, column_index in zip(row_indexes, column_indexes):
            frame.iat[row_index, column_index] = np.nan

    if duplicate_percent:
        duplicate_count = min(rows - 1, int(rows * duplicate_percent / 100))
        if duplicate_count > 0:
            duplicates = frame.sample(n=duplicate_count, random_state=seed)
            frame = pd.concat([frame, duplicates], ignore_index=True)

    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def get_outlier_profile(
    frame: pd.DataFrame, target: str | None = None
) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    feature_frame = frame.drop(columns=[target], errors="ignore")
    numeric_columns = feature_frame.select_dtypes(include="number").columns
    rows_with_outliers = pd.Series(False, index=frame.index)
    for column in numeric_columns:
        values = feature_frame[column]
        q1, q3 = values.quantile([0.25, 0.75])
        if pd.isna(q1) or pd.isna(q3):
            counts[column] = 0
            continue
        spread = q3 - q1
        mask = values.lt(q1 - 1.5 * spread) | values.gt(q3 + 1.5 * spread)
        counts[column] = int(mask.sum())
        rows_with_outliers |= mask.fillna(False)
    return counts, int(rows_with_outliers.sum())


def assess_quality(frame: pd.DataFrame, target: str | None) -> dict[str, Any]:
    rows, columns = frame.shape
    missing_cells = int(frame.isna().sum().sum())
    total_cells = rows * columns
    missing_fraction = missing_cells / total_cells if total_cells else 0.0
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_fraction = duplicate_rows / rows if rows else 0.0
    outlier_by_column, outlier_rows = get_outlier_profile(frame, target)
    outlier_fraction = outlier_rows / rows if rows else 0.0

    distribution = pd.Series(dtype="int64")
    imbalance_fraction = 0.0
    imbalance_ratio = 1.0
    if target and target in frame:
        distribution = frame[target].dropna().astype(str).value_counts()
        if len(distribution) > 1 and int(distribution.min()) > 0:
            imbalance_ratio = float(distribution.max() / distribution.min())
            imbalance_fraction = 1.0 - float(distribution.min() / distribution.max())

    score = 100.0 - (
        25.0 * missing_fraction
        + 15.0 * duplicate_fraction
        + 25.0 * outlier_fraction
        + 35.0 * imbalance_fraction
    )
    return {
        "rows": rows,
        "columns": columns,
        "missing_cells": missing_cells,
        "missing_fraction": missing_fraction,
        "duplicate_rows": duplicate_rows,
        "duplicate_fraction": duplicate_fraction,
        "outlier_by_column": outlier_by_column,
        "outlier_rows": outlier_rows,
        "outlier_fraction": outlier_fraction,
        "class_distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_fraction": imbalance_fraction,
        "quality_score": float(np.clip(score, 0, 100)),
    }


def clean_dataset(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create a cleaned export; this is separate from leakage-safe model evaluation."""
    cleaned = frame.copy()
    removed_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    missing_target_rows = int(cleaned[target].isna().sum())
    cleaned = cleaned.dropna(subset=[target]).copy()

    numeric_columns = [
        column
        for column in cleaned.select_dtypes(include="number").columns
        if column != target
    ]
    categorical_columns = [
        column
        for column in cleaned.columns
        if column not in numeric_columns and column != target
    ]
    for column in numeric_columns:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        median = values.median()
        cleaned[column] = values.fillna(0 if pd.isna(median) else median)
        q1, q3 = cleaned[column].quantile([0.25, 0.75])
        spread = q3 - q1
        if pd.notna(spread):
            cleaned[column] = cleaned[column].clip(
                lower=q1 - 1.5 * spread, upper=q3 + 1.5 * spread
            )

    for column in categorical_columns:
        mode = cleaned[column].mode(dropna=True)
        replacement = mode.iloc[0] if not mode.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(replacement)

    return cleaned.reset_index(drop=True), {
        "removed_duplicates": removed_duplicates,
        "removed_missing_target": missing_target_rows,
    }


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    robust: bool,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = []
    if robust:
        numeric_steps.append(("iqr_clip", IQRClipper()))
    numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(steps=categorical_steps),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_model_comparison(
    frame: pd.DataFrame,
    target: str,
    test_fraction: float,
) -> dict[str, Any]:
    """Compare pipelines on one shared holdout split; all learned transforms fit on train."""
    modeling_data = frame.drop_duplicates().dropna(subset=[target]).copy()
    y = modeling_data[target].astype(str)
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("The selected target has fewer than two classes.")
    if len(counts) > 30:
        raise ValueError(
            f"The selected target has {len(counts)} classes. Choose a classification "
            "target with 2–30 classes for this comparison."
        )
    if int(counts.min()) < 2:
        raise ValueError(
            "Every target class needs at least two records after duplicate removal "
            "for a stratified train/test split."
        )

    X = modeling_data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    empty_training_columns = list(X_train.columns[X_train.isna().all()])
    if empty_training_columns:
        X_train = X_train.drop(columns=empty_training_columns)
        X_test = X_test.drop(columns=empty_training_columns)
    if X_train.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after excluding columns that are "
            "entirely missing in the training split."
        )
    numeric_columns = list(X_train.select_dtypes(include="number").columns)
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]

    scenarios = [
        ("Before cleaning", False, False),
        ("After cleaning", True, True),
    ]
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    for scenario, robust, balanced in scenarios:
        for model_name, estimator in models:
            scale_numeric = model_name == "Logistic Regression"
            model = estimator
            if balanced:
                model = (
                    LogisticRegression(
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    )
                    if model_name == "Logistic Regression"
                    else RandomForestClassifier(
                        n_estimators=150,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight="balanced",
                    )
                )
            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        make_preprocessor(
                            numeric_columns,
                            categorical_columns,
                            robust=robust,
                            scale_numeric=scale_numeric,
                        ),
                    ),
                    ("classifier", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            results.append(
                {
                    "Condition": scenario,
                    "Model": model_name,
                    "Accuracy": float(accuracy_score(y_test, predictions)),
                    "Macro F1": float(
                        f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )

    return {
        "metrics": pd.DataFrame(results),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "evaluation_rows": len(modeling_data),
        "duplicate_rows_excluded": int(frame.duplicated().sum()),
        "missing_target_rows_excluded": int(frame[target].isna().sum()),
        "class_counts": counts.sort_index(),
        "test_fraction": test_fraction,
    }


def make_report(
    frame: pd.DataFrame,
    target: str,
    assessment: dict[str, Any],
    cleaned: pd.DataFrame,
    cleaning_log: dict[str, int],
    comparison: dict[str, Any] | None,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {APP_TITLE}",
        "",
        f"**Generated:** {generated_at}",
        "",
        "## Abstract",
        "",
        "This report summarizes an exploratory dataset quality assessment and, when "
        "available, a controlled comparison of classification performance before "
        "and after robust preprocessing. Results are specific to the uploaded or "
        "synthetic dataset and do not establish causal claims beyond this evaluation.",
        "",
        "## Dataset and quality assessment",
        "",
        f"- Source dimensions: {len(frame):,} rows × {len(frame.columns):,} columns",
        f"- Target: `{target}`",
        f"- Missing cells: {assessment['missing_cells']:,} "
        f"({assessment['missing_fraction']:.2%})",
        f"- Exact duplicate rows: {assessment['duplicate_rows']:,}",
        f"- Rows with at least one numeric IQR outlier: {assessment['outlier_rows']:,}",
        f"- Majority/minority class ratio: {assessment['imbalance_ratio']:.2f}:1",
        f"- Composite quality score: {assessment['quality_score']:.1f}/100",
        "",
        "The score subtracts weighted penalties for missing cells (25 points), "
        "duplicate rows (15), rows with numeric IQR outliers (25), and class "
        "imbalance (35). Each penalty is scaled by its measured fraction; the "
        "final score is clipped to 0–100.",
        "",
        "## Cleaning applied to the export",
        "",
        f"- Removed exact duplicate rows: {cleaning_log['removed_duplicates']:,}",
        f"- Removed rows with missing target: {cleaning_log['removed_missing_target']:,}",
        "- Imputed numeric feature values with medians and capped numeric outliers "
        "at 1.5×IQR bounds.",
        "- Imputed categorical feature values with their most frequent value "
        "(or `Unknown` when no value was available).",
        f"- Cleaned export dimensions: {len(cleaned):,} rows × {len(cleaned.columns):,} columns",
        "",
        "## Model evaluation methodology",
        "",
        "The evaluation removes exact duplicates and rows without a target before "
        "a single stratified train/test split. Both conditions use the same holdout "
        "records. Missing-value imputers, one-hot encoders, numeric scaling, and "
        "IQR clipping thresholds are learned inside scikit-learn pipelines from "
        "training records only. The test set is never used to fit preprocessing.",
        "",
        "The before-cleaning pipelines use median/mode imputation as required for "
        "model compatibility. The after-cleaning pipelines add training-only IQR "
        "capping and balanced class weights. F1 is macro-averaged so every class "
        "contributes equally.",
        "",
    ]
    if comparison is None:
        lines.extend(
            [
                "No model comparison has been run yet. Run it in the Model impact "
                "tab to include measured results here.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Evaluation records: {comparison['evaluation_rows']:,}",
                f"- Training records: {comparison['train_rows']:,}",
                f"- Holdout records: {comparison['test_rows']:,}",
                f"- Holdout fraction: {comparison['test_fraction']:.0%}",
                "",
                "### Results",
                "",
                "| Condition | Model | Accuracy | Macro F1 |",
                "|---|---|---:|---:|",
            ]
        )
        for row in comparison["metrics"].itertuples(index=False):
            lines.append(
                f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} |"
            )
        lines.extend(
            [
                "",
                "These are held-out metrics from one random seed and split. Small "
                "datasets can have high variance; repeat with multiple seeds or "
                "cross-validation before drawing strong conclusions.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "IQR rules can flag valid rare observations; median/mode imputation and "
            "class weighting are not universally optimal. The composite score is a "
            "transparent heuristic rather than a standardized data-quality measure. "
            "This application does not infer causal impact and does not replace "
            "domain review.",
            "",
        ]
    )
    return "\n".join(lines)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def style_plotly_figure(figure: go.Figure, *, height: int = 340) -> go.Figure:
    """Give each interactive chart the app's dark, high-contrast visual treatment."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dbe5fa", "size": 12},
        margin={"l": 14, "r": 14, "t": 58, "b": 20},
        height=height,
        hoverlabel={
            "bgcolor": "#121a2d",
            "bordercolor": "rgba(151, 169, 207, 0.26)",
            "font": {"color": "#f4f7ff"},
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#c3cde2"},
        },
    )
    figure.update_xaxes(
        gridcolor="rgba(151, 169, 207, 0.08)",
        linecolor="rgba(151, 169, 207, 0.17)",
        tickfont={"color": "#aab7d0"},
        title_font={"color": "#9daaca"},
        zerolinecolor="rgba(151, 169, 207, 0.12)",
    )
    figure.update_yaxes(
        gridcolor="rgba(151, 169, 207, 0.11)",
        linecolor="rgba(151, 169, 207, 0.12)",
        tickfont={"color": "#aab7d0"},
        title_font={"color": "#9daaca"},
        zerolinecolor="rgba(151, 169, 207, 0.12)",
    )
    return figure


st.markdown(
    '<div class="hero-kicker">DATA SCIENCE LAB <span>QUALITY INTELLIGENCE</span></div>',
    unsafe_allow_html=True,
)
st.title(APP_TITLE)
st.caption(
    "Inspect data quality, create a cleaned export, and measure how robust "
    "preprocessing changes held-out classification performance."
)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-kicker">WORKSPACE / DATASET</div>',
        unsafe_allow_html=True,
    )
    st.header("Dataset")
    source = st.radio(
        "Choose a data source", ["Upload CSV", "Generate synthetic"], index=1
    )
    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="The file is processed in this app session and is not sent to an external service.",
        )
        st.caption("Or switch to synthetic data to explore the workflow immediately.")
    else:
        st.caption("Generate a classification dataset with controlled quality issues.")
        synthetic_rows = st.slider("Rows", 150, 5000, 1000, step=50)
        synthetic_features = st.slider("Numeric features", 3, 20, 8)
        synthetic_classes = st.selectbox("Number of classes", [2, 3])
        synthetic_imbalance = st.selectbox(
            "Class balance", ["Balanced", "Moderate", "High"]
        )
        synthetic_missing = st.slider("Missing feature cells (%)", 0, 30, 8)
        synthetic_duplicates = st.slider("Duplicate rows (%)", 0, 20, 5)
        synthetic_outliers = st.slider("Outlier feature cells (%)", 0, 10, 2)
        synthetic_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

frame: pd.DataFrame | None = None
dataset_label = ""
dataset_fingerprint = ""
if source == "Generate synthetic":
    frame = make_synthetic_data(
        rows=synthetic_rows,
        feature_count=synthetic_features,
        class_count=synthetic_classes,
        imbalance=synthetic_imbalance,
        missing_percent=synthetic_missing,
        duplicate_percent=synthetic_duplicates,
        outlier_percent=synthetic_outliers,
        seed=int(synthetic_seed),
    )
    dataset_label = "Synthetic classification dataset"
    dataset_fingerprint = hashlib.sha256(
        (
            f"{synthetic_rows}-{synthetic_features}-{synthetic_classes}-"
            f"{synthetic_imbalance}-{synthetic_missing}-{synthetic_duplicates}-"
            f"{synthetic_outliers}-{synthetic_seed}"
        ).encode("utf-8")
    ).hexdigest()
elif uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.getvalue()
        frame = pd.read_csv(BytesIO(raw_bytes), low_memory=False)
        dataset_label = uploaded_file.name
        dataset_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read this CSV: {error}")
        st.stop()

if frame is None:
    st.info(
        "Upload a CSV or switch to synthetic data in the sidebar to begin. "
        "Synthetic data includes adjustable missing values, duplicates, outliers, "
        "and class imbalance."
    )
    st.stop()

if frame.empty or len(frame.columns) < 2:
    st.error("The dataset needs at least one row and two columns (features and target).")
    st.stop()
if frame.columns.duplicated().any():
    st.error("CSV column names must be unique. Rename duplicate columns and upload again.")
    st.stop()

target = st.selectbox(
    "Select the classification target",
    options=list(frame.columns),
    index=len(frame.columns) - 1,
    help="The selected column must contain at least two classes for model comparison.",
)
assessment = assess_quality(frame, target)
cleaned_frame, cleaning_log = clean_dataset(frame, target)
quality_after = assess_quality(cleaned_frame, target)
model_state_key = f"{dataset_fingerprint}:{target}"

stored_dataset_key = st.session_state.get("comparison_dataset_key")
if stored_dataset_key is not None and stored_dataset_key != model_state_key:
    st.session_state.pop("comparison_result", None)
    st.session_state.pop("comparison_dataset_key", None)
    st.session_state.pop("comparison_test_fraction", None)

st.markdown(
    f"""
    <div class="dataset-strip">
        <span class="dataset-strip__dot"></span>
        <div>
            <div class="dataset-strip__label">Active dataset</div>
            <div class="dataset-strip__name">{escape(dataset_label)}</div>
        </div>
        <div class="dataset-strip__meta">
            {len(frame):,} rows&nbsp;&nbsp;·&nbsp;&nbsp;{len(frame.columns):,} columns
            &nbsp;&nbsp;·&nbsp;&nbsp;Target: {escape(str(target))}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, cleaning_tab, model_tab, report_tab = st.tabs(
    ["Quality overview", "Data cleaning", "Model impact", "Research report"]
)

with overview_tab:
    st.subheader("Dataset health")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Quality score", f"{assessment['quality_score']:.1f}/100")
    metric_columns[1].metric(
        "Missing cells",
        f"{assessment['missing_cells']:,}",
        f"{assessment['missing_fraction']:.1%} of all cells",
    )
    metric_columns[2].metric(
        "Duplicate rows", f"{assessment['duplicate_rows']:,}"
    )
    metric_columns[3].metric(
        "Rows with outliers", f"{assessment['outlier_rows']:,}"
    )

    score = assessment["quality_score"]
    if score >= 85:
        score_label, score_color = "Excellent", "#35d39a"
    elif score >= 70:
        score_label, score_color = "Good", "#65a6ff"
    elif score >= 50:
        score_label, score_color = "Fair", "#f5b95e"
    else:
        score_label, score_color = "Needs attention", "#fb7185"

    score_chart = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "/100",
                "valueformat": ".1f",
                "font": {"color": "#f5f7ff", "size": 38},
            },
            title={
                "text": f"DATA QUALITY INDEX · {score_label.upper()}",
                "font": {"color": "#a9b7d4", "size": 12},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#73819e",
                    "tickfont": {"color": "#8f9db8", "size": 10},
                },
                "bar": {"color": score_color, "thickness": 0.27},
                "bgcolor": "rgba(15, 23, 42, 0.88)",
                "bordercolor": "rgba(151, 169, 207, 0.17)",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 50], "color": "rgba(251, 113, 133, 0.14)"},
                    {"range": [50, 75], "color": "rgba(245, 185, 94, 0.14)"},
                    {"range": [75, 100], "color": "rgba(53, 211, 154, 0.13)"},
                ],
            },
        )
    )
    style_plotly_figure(score_chart, height=285)

    left, right = st.columns([0.95, 1.45])
    with left:
        st.plotly_chart(score_chart, use_container_width=True)
        st.caption(
            "Score heuristic: 25% missingness, 15% duplicates, 25% outlier rows, "
            "and 35% class imbalance penalties."
        )
        ratio = assessment["imbalance_ratio"]
        st.metric(
            "Majority / minority ratio",
            f"{ratio:.2f}:1",
            "Imbalance detected" if ratio >= 1.5 else "No material imbalance",
        )
    with right:
        if not assessment["class_distribution"].empty:
            class_frame = assessment["class_distribution"].rename("Records").rename_axis(
                "Class"
            ).reset_index()
            class_chart = px.bar(
                class_frame,
                x="Class",
                y="Records",
                title="Target class distribution",
                color="Class",
                color_discrete_sequence=["#5b8cff", "#a78bfa", "#3dd6d0", "#f5b95e"],
            )
            class_chart.update_layout(showlegend=False)
            class_chart.update_traces(
                marker_line_color="rgba(229, 237, 255, 0.14)",
                marker_line_width=1,
                opacity=0.94,
            )
            style_plotly_figure(class_chart, height=350)
            st.plotly_chart(class_chart, use_container_width=True)
        else:
            st.info("No non-missing target values are available for class distribution.")

    st.subheader("Column-level quality checks")
    missing_profile = frame.isna().sum().rename("Missing values")
    missing_profile = missing_profile.to_frame()
    missing_profile["Missing (%)"] = (
        frame.isna().mean().mul(100).round(2)
    )
    outlier_profile = pd.Series(
        assessment["outlier_by_column"], name="IQR outlier values"
    )
    quality_table = missing_profile.join(outlier_profile, how="left").fillna(0)
    st.dataframe(quality_table, use_container_width=True)

    with st.expander("Show source data sample"):
        st.dataframe(frame.head(100), use_container_width=True)

with cleaning_tab:
    st.subheader("Cleaning summary")
    st.write(
        "The cleaned CSV removes exact duplicate rows and records without a target, "
        "fills missing features, and caps numeric feature outliers using the 1.5×IQR rule."
    )
    cleaning_columns = st.columns(4)
    cleaning_columns[0].metric(
        "Duplicate rows removed", f"{cleaning_log['removed_duplicates']:,}"
    )
    cleaning_columns[1].metric(
        "Missing-target rows removed",
        f"{cleaning_log['removed_missing_target']:,}",
    )
    cleaning_columns[2].metric(
        "Rows after cleaning", f"{len(cleaned_frame):,}"
    )
    cleaning_columns[3].metric(
        "Quality score after", f"{quality_after['quality_score']:.1f}/100"
    )
    st.download_button(
        "Download cleaned CSV",
        data=csv_bytes(cleaned_frame),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.subheader("Cleaned data preview")
    st.dataframe(cleaned_frame.head(100), use_container_width=True)
    with st.expander("Cleaning rules"):
        st.markdown(
            "- Exact duplicate rows are removed.\n"
            "- Rows with a missing target are removed because the outcome cannot be inferred.\n"
            "- Numeric feature gaps are filled with the median; numeric outliers are capped at 1.5×IQR.\n"
            "- Categorical feature gaps are filled with the mode, or `Unknown` if the column is entirely empty.\n"
            "- Target values are not imputed or modified."
        )

with model_tab:
    st.subheader("Before vs. after cleaning")
    st.write(
        "Both conditions use the same stratified holdout. The before condition uses "
        "standard model-ready imputation; the after condition adds training-only IQR "
        "capping and class weighting. Duplicate rows are removed before splitting "
        "in both conditions to prevent identical records crossing the split."
    )
    test_fraction = st.slider(
        "Holdout test data (%)", min_value=15, max_value=40, value=20, step=5
    )
    current_model_key = f"{model_state_key}:{test_fraction}"
    if st.button("Run model comparison", type="primary"):
        with st.spinner("Training four pipelines on the shared train/test split..."):
            try:
                st.session_state["comparison_result"] = run_model_comparison(
                    frame, target, test_fraction / 100
                )
                st.session_state["comparison_dataset_key"] = model_state_key
                st.session_state["comparison_test_fraction"] = test_fraction
            except (ValueError, TypeError) as error:
                st.session_state.pop("comparison_result", None)
                st.session_state.pop("comparison_dataset_key", None)
                st.session_state.pop("comparison_test_fraction", None)
                st.error(str(error))
            except Exception as error:
                st.session_state.pop("comparison_result", None)
                st.session_state.pop("comparison_dataset_key", None)
                st.session_state.pop("comparison_test_fraction", None)
                st.error(f"Model training failed: {error}")

    comparison = None
    if (
        st.session_state.get("comparison_dataset_key") == model_state_key
        and st.session_state.get("comparison_test_fraction") == test_fraction
        and st.session_state.get("comparison_result") is not None
    ):
        comparison = st.session_state["comparison_result"]
        metrics = comparison["metrics"]
        metric_left, metric_right = st.columns(2)
        with metric_left:
            accuracy_chart = px.bar(
                metrics,
                x="Model",
                y="Accuracy",
                color="Condition",
                barmode="group",
                title="Accuracy",
                range_y=[0, 1],
                color_discrete_map={
                    "Before cleaning": "#66748f",
                    "After cleaning": "#9b7cff",
                },
                text_auto=".1%",
            )
            accuracy_chart.update_traces(
                marker_line_color="rgba(235, 240, 255, 0.16)",
                marker_line_width=1,
                opacity=0.94,
                textposition="outside",
            )
            style_plotly_figure(accuracy_chart, height=360)
            accuracy_chart.update_layout(
                legend={"orientation": "h", "y": 1.12, "x": 0}
            )
            accuracy_chart.update_yaxes(tickformat=".0%", title=None, range=[0, 1])
            st.plotly_chart(accuracy_chart, use_container_width=True)
        with metric_right:
            f1_chart = px.bar(
                metrics,
                x="Model",
                y="Macro F1",
                color="Condition",
                barmode="group",
                title="Macro F1 score",
                range_y=[0, 1],
                color_discrete_map={
                    "Before cleaning": "#66748f",
                    "After cleaning": "#5b8cff",
                },
                text_auto=".1%",
            )
            f1_chart.update_traces(
                marker_line_color="rgba(235, 240, 255, 0.16)",
                marker_line_width=1,
                opacity=0.94,
                textposition="outside",
            )
            style_plotly_figure(f1_chart, height=360)
            f1_chart.update_layout(
                legend={"orientation": "h", "y": 1.12, "x": 0}
            )
            f1_chart.update_yaxes(tickformat=".0%", title=None, range=[0, 1])
            st.plotly_chart(f1_chart, use_container_width=True)
        before_mean = metrics.loc[
            metrics["Condition"] == "Before cleaning", ["Accuracy", "Macro F1"]
        ].mean()
        after_mean = metrics.loc[
            metrics["Condition"] == "After cleaning", ["Accuracy", "Macro F1"]
        ].mean()
        accuracy_delta = (after_mean["Accuracy"] - before_mean["Accuracy"]) * 100
        f1_delta = (after_mean["Macro F1"] - before_mean["Macro F1"]) * 100
        st.markdown(
            '<div class="comparison-insight">'
            "<strong>Average change after cleaning</strong>"
            f"Accuracy {accuracy_delta:+.1f} pp&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Macro F1 {f1_delta:+.1f} pp "
            "<span>averaged across both models on the same holdout.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
        st.caption(
            f"Train: {comparison['train_rows']:,} rows · "
            f"Holdout: {comparison['test_rows']:,} rows · "
            "Macro F1 gives equal weight to each class."
        )
    else:
        st.info("Run the comparison to train both models before and after cleaning.")

with report_tab:
    comparison_for_report = None
    if st.session_state.get("comparison_dataset_key") == model_state_key:
        comparison_for_report = st.session_state.get("comparison_result")
    report = make_report(
        frame,
        target,
        assessment,
        cleaned_frame,
        cleaning_log,
        comparison_for_report,
    )
    st.subheader("Research report")
    st.write(
        "Download a Markdown report containing the assessment, cleaning steps, "
        "evaluation methodology, results (if run), and limitations."
    )
    st.download_button(
        "Download research report (.md)",
        data=report.encode("utf-8"),
        file_name="dataset_quality_research_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview report", expanded=True):
        st.markdown(report)
