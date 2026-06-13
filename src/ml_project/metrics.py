from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def compute_binary_metrics(*, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob length mismatch.")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return BinaryMetrics(
        auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def metrics_table(*, y_train: np.ndarray, y_test: np.ndarray, model_results: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in model_results:
        name = r["name"]
        train = compute_binary_metrics(y_true=y_train, y_prob=r["oof_prob"])
        test = compute_binary_metrics(y_true=y_test, y_prob=r["test_prob"])
        rows.append(
            {
                "model": name,
                "train_auc_oof": train.auc,
                "test_auc": test.auc,
                "test_accuracy": test.accuracy,
                "test_precision": test.precision,
                "test_recall": test.recall,
                "test_f1": test.f1,
                "test_tn": test.tn,
                "test_fp": test.fp,
                "test_fn": test.fn,
                "test_tp": test.tp,
            }
        )
    return pd.DataFrame(rows).sort_values(by="test_auc", ascending=False).reset_index(drop=True)

