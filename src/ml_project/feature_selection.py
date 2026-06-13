from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml_project.preprocess import infer_feature_spec, make_preprocessor
from ml_project.table1 import summarize_table1


@dataclass(frozen=True)
class FeatureSelectionResult:
    ranked_features: list[str]
    k_values: list[int]
    mean_auc: list[float]
    best_k: int
    best_features: list[str]


def rank_features_by_pvalue(*, df_train: pd.DataFrame, outcome_col: str, id_col: str | None) -> pd.DataFrame:
    table = summarize_table1(df=df_train, group_col=outcome_col, id_col=id_col)
    return table


def select_optimal_k_via_rf(
    *,
    df_train: pd.DataFrame,
    outcome_col: str,
    id_col: str | None,
    ranked_features: list[str],
    k_min: int = 5,
    k_max: int = 35,
    cv_folds: int = 5,
    seed: int = 42,
    categorical_encoding: str = "onehot",
) -> FeatureSelectionResult:
    y = df_train[outcome_col].astype(int).to_numpy()
    feature_cols = [c for c in ranked_features if c not in {outcome_col, id_col}]
    if len(feature_cols) < k_min:
        raise ValueError(f"Not enough features after filtering: {len(feature_cols)}")

    k_max2 = min(k_max, len(feature_cols))
    k_values = list(range(k_min, k_max2 + 1))
    mean_auc: list[float] = []

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for k in k_values:
        cols = feature_cols[:k]
        spec = infer_feature_spec(df=df_train, feature_cols=cols)
        pre = make_preprocessor(spec=spec, categorical_encoding=categorical_encoding)
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            max_features="sqrt",
        )
        pipe = Pipeline(steps=[("preprocess", pre), ("model", rf)])
        # Note: avoid joblib multiprocessing (may be restricted in some environments)
        scores = cross_val_score(pipe, df_train[cols], y, cv=cv, scoring="roc_auc", n_jobs=1)
        mean_auc.append(float(np.mean(scores)))

    best_idx = int(np.argmax(mean_auc))
    best_k = int(k_values[best_idx])
    best_features = feature_cols[:best_k]

    return FeatureSelectionResult(
        ranked_features=feature_cols,
        k_values=k_values,
        mean_auc=mean_auc,
        best_k=best_k,
        best_features=best_features,
    )
