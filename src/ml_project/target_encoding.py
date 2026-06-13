from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class _EncStats:
    global_mean: float
    mapping: dict[str, float]


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-safe target encoder when used inside CV pipelines.

    - Each categorical column is replaced with a smoothed mean target by category.
    - Unknown categories map to the global mean of y in the fitted data.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = float(smoothing)
        self._stats: list[_EncStats] | None = None
        self._feature_names_in: list[str] | None = None

    def fit(self, X, y):  # noqa: N802
        if y is None:
            raise ValueError("TargetEncoder requires y for fit.")

        x_df = pd.DataFrame(X)
        y_arr = np.asarray(y).astype(float)
        if x_df.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch.")

        self._feature_names_in = list(getattr(X, "columns", [f"x{i}" for i in range(x_df.shape[1])]))
        global_mean = float(np.mean(y_arr))

        stats: list[_EncStats] = []
        smoothing = max(0.0, float(self.smoothing))
        for col_idx in range(x_df.shape[1]):
            s = x_df.iloc[:, col_idx].astype("string").fillna("NA")
            grp = pd.DataFrame({"cat": s, "y": y_arr}).groupby("cat")["y"].agg(["mean", "count"])

            # smoothed mean: (count*mean + smoothing*global_mean)/(count+smoothing)
            enc = (grp["count"] * grp["mean"] + smoothing * global_mean) / (grp["count"] + smoothing)
            mapping = {str(k): float(v) for k, v in enc.to_dict().items()}
            stats.append(_EncStats(global_mean=global_mean, mapping=mapping))

        self._stats = stats
        return self

    def transform(self, X):  # noqa: N802
        if self._stats is None:
            raise ValueError("TargetEncoder is not fitted.")

        x_df = pd.DataFrame(X)
        if x_df.shape[1] != len(self._stats):
            raise ValueError("Unexpected number of columns.")

        out = np.empty((x_df.shape[0], x_df.shape[1]), dtype=float)
        for col_idx, st in enumerate(self._stats):
            s = x_df.iloc[:, col_idx].astype("string").fillna("NA")
            mapped = s.map(lambda v: st.mapping.get(str(v), st.global_mean))
            out[:, col_idx] = mapped.to_numpy(dtype=float)

        return out

    def get_feature_names_out(self, input_features: Any = None):  # noqa: ANN401
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        if self._feature_names_in is None:
            return np.asarray([], dtype=object)
        return np.asarray(self._feature_names_in, dtype=object)

