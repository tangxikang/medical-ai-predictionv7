from __future__ import annotations

import numpy as np


def dca_curve(*, y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decision Curve Analysis net benefit curve.

    NetBenefit(t) = TP/N - FP/N * (t/(1-t))
    thresholds must be within (0,1).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.asarray(thresholds)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D arrays.")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1D.")
    if np.any(thresholds <= 0) or np.any(thresholds >= 1):
        raise ValueError("thresholds must be strictly within (0, 1).")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing.")

    y_bin = (y_true > 0).astype(int)
    n = y_bin.shape[0]
    if n == 0:
        raise ValueError("Empty y_true.")

    net_benefit = np.empty_like(thresholds, dtype=float)
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        tp = float(((y_pred == 1) & (y_bin == 1)).sum())
        fp = float(((y_pred == 1) & (y_bin == 0)).sum())
        net_benefit[i] = (tp / n) - (fp / n) * (t / (1.0 - t))

    return thresholds, net_benefit


def treat_all_curve(*, y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Net benefit of treating all patients at each threshold.
    """
    y_true = np.asarray(y_true)
    thresholds = np.asarray(thresholds)
    y_bin = (y_true > 0).astype(int)
    prevalence = float(y_bin.mean())
    return prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))

