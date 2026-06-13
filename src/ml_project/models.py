from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

from ml_project.preprocess import FeatureSpec, infer_feature_spec, make_preprocessor


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    oof_prob: np.ndarray
    test_prob: np.ndarray


def _try_make_xgb(*, seed: int, scale_pos_weight: float | None):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    return XGBClassifier(
        booster="gbtree",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )


def _try_make_lgbm(*, seed: int, scale_pos_weight: float | None):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None

    return LGBMClassifier(
        random_state=seed,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
        force_row_wise=True,
    )


def build_model_zoo(*, seed: int = 42, scale_pos_weight: float | None = None) -> dict[str, object]:
    zoo: dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=900,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1200,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=400,
            learning_rate=0.3,
            random_state=seed,
            algorithm="SAMME",
        ),
    }

    xgb = _try_make_xgb(seed=seed, scale_pos_weight=scale_pos_weight)
    if xgb is not None:
        zoo["XGBoost"] = xgb

    lgbm = _try_make_lgbm(seed=seed, scale_pos_weight=scale_pos_weight)
    if lgbm is not None:
        zoo["LightGBM"] = lgbm

    # Stacking (kept within 6-8 models by using it as the 8th when available)
    if "XGBoost" in zoo and "LightGBM" in zoo:
        zoo["Stacking"] = StackingClassifier(
            estimators=[
                ("lr", clone(zoo["LogisticRegression"])),
                ("xgb", clone(zoo["XGBoost"])),
                ("lgbm", clone(zoo["LightGBM"])),
            ],
            final_estimator=LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
            stack_method="predict_proba",
            passthrough=False,
            cv=5,
            n_jobs=1,
        )

    # Keep total models in 6-8 range
    # Prefer boosted + stacking if present.
    if len(zoo) > 8:
        # Drop the weakest defaults first if we ever exceed 8
        for drop in ["ExtraTrees", "RandomForest"]:
            if len(zoo) <= 8:
                break
            zoo.pop(drop, None)

    return zoo


def train_and_predict(
    *,
    model_name: str,
    model,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    outcome_col: str,
    feature_cols: list[str],
    seed: int = 42,
    cv_folds: int = 5,
    tune: bool = False,
    categorical_encoding: str = "onehot",
) -> ModelResult:
    y_train = df_train[outcome_col].astype(int).to_numpy()
    y_test = df_test[outcome_col].astype(int).to_numpy()

    spec = infer_feature_spec(df=df_train, feature_cols=feature_cols)
    pre = make_preprocessor(spec=spec, categorical_encoding=categorical_encoding)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    if tune and model_name in {"XGBoost", "LightGBM", "ExtraTrees", "RandomForest"}:
        pipe = _tune_pipeline(
            model_name=model_name,
            base_pipeline=pipe,
            x=df_train[feature_cols],
            y=y_train,
            seed=seed,
            cv_folds=cv_folds,
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    # Note: avoid joblib multiprocessing (may be restricted in some environments)
    oof = cross_val_predict(pipe, df_train[feature_cols], y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    pipe.fit(df_train[feature_cols], y_train)
    test_prob = pipe.predict_proba(df_test[feature_cols])[:, 1]

    if oof.shape[0] != y_train.shape[0] or test_prob.shape[0] != y_test.shape[0]:
        raise RuntimeError("Prediction shape mismatch.")

    return ModelResult(name=model_name, pipeline=pipe, oof_prob=oof, test_prob=test_prob)


def _tune_pipeline(
    *,
    model_name: str,
    base_pipeline: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    cv_folds: int,
) -> Pipeline:
    """
    Light-weight CV tuning for boosted models only (keeps runtime bounded).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if model_name == "XGBoost":
        param_dist = {
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
            "model__n_estimators": [300, 500, 800, 1200],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1.0, 3.0, 5.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
        n_iter = 25
    elif model_name == "LightGBM":
        param_dist = {
            "model__num_leaves": [15, 31, 63, 127],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
            "model__n_estimators": [400, 800, 1200, 2000],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_samples": [5, 10, 20, 40],
            "model__reg_lambda": [0.0, 0.5, 1.0, 2.0],
        }
        n_iter = 25
    elif model_name == "ExtraTrees":
        param_dist = {
            "model__n_estimators": [600, 900, 1200, 1600, 2200],
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8, 12],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__max_features": ["sqrt", 0.5, 0.7, 0.9],
            "model__bootstrap": [False, True],
        }
        n_iter = 25
    elif model_name == "RandomForest":
        param_dist = {
            "model__n_estimators": [600, 900, 1200, 1600, 2200],
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8, 12],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__max_features": ["sqrt", 0.5, 0.7, 0.9],
            "model__bootstrap": [False, True],
        }
        n_iter = 20
    else:
        return base_pipeline

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(x, y)
    return search.best_estimator_
