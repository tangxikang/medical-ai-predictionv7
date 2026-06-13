from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.target_encoding import TargetEncoder


@dataclass(frozen=True)
class FeatureSpec:
    numeric_cols: list[str]
    categorical_cols: list[str]


def infer_feature_spec(*, df: pd.DataFrame, feature_cols: list[str]) -> FeatureSpec:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue

        # Fallback for object columns: try converting to numeric.
        # If >= 70% of non-NA values parse as numeric, treat as numeric.
        if s.dtype == object:
            parsed = pd.to_numeric(s, errors="coerce")
            non_na = s.notna().sum()
            parsed_non_na = parsed.notna().sum()
            if non_na > 0 and parsed_non_na / non_na >= 0.70:
                numeric_cols.append(c)
                continue

        categorical_cols.append(c)
    return FeatureSpec(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _one_hot_encoder() -> OneHotEncoder:
    # sklearn 1.2+ uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(*, spec: FeatureSpec, categorical_encoding: str = "onehot") -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    if categorical_encoding not in {"onehot", "target"}:
        raise ValueError("categorical_encoding must be one of: onehot, target")

    if categorical_encoding == "onehot":
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", _one_hot_encoder()),
            ]
        )
    else:
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("target", TargetEncoder(smoothing=12.0)),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_cols),
            ("cat", cat_pipe, spec.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(*, preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: best-effort; keeps training running even if names are unavailable
        return [f"f{i}" for i in range(int(getattr(preprocessor, "n_features_in_", 0) or 0))]


def ensure_2d_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)
