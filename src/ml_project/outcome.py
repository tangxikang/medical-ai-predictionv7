from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_TRUE_SET = {"1", "true", "t", "yes", "y", "是", "阳性", "positive"}
_FALSE_SET = {"0", "false", "f", "no", "n", "否", "阴性", "negative"}


def coerce_binary_outcome(values: pd.Series | Iterable[object]) -> pd.Series:
    """
    Coerce common binary labels into {0,1}.

    Accepts: 0/1, bool, "0"/"1", yes/no, 是/否, etc.
    Raises ValueError if an unknown label exists.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))

    if series.dtype == bool:
        return series.astype(int)

    # Keep NA as NA; validate non-NA values
    s = series.astype("string")
    normalized = s.str.strip().str.lower()

    out = pd.Series(index=series.index, dtype="Int64")
    is_na = normalized.isna()
    out.loc[is_na] = pd.NA

    truthy = normalized.isin(_TRUE_SET)
    falsy = normalized.isin(_FALSE_SET)
    out.loc[truthy] = 1
    out.loc[falsy] = 0

    unknown_mask = ~(is_na | truthy | falsy)
    if unknown_mask.any():
        unknown = sorted(set(normalized.loc[unknown_mask].dropna().tolist()))
        raise ValueError(f"Unknown outcome labels: {unknown[:20]}")

    if out.dropna().nunique() != 2:
        # allow degenerate in tiny unit tests? but the pipeline assumes binary
        uniques = out.dropna().unique().tolist()
        raise ValueError(f"Outcome must have both classes 0 and 1; got {uniques}")

    return out.astype(int)


def find_outcome_column(columns: list[str]) -> str:
    candidates = ["Outcome", "结局", "outcome", "Y", "y", "label", "target"]
    for c in candidates:
        if c in columns:
            return c
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError("Cannot find outcome column (expected one of Outcome/结局/label/target).")

