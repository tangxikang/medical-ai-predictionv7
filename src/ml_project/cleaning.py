from __future__ import annotations

import pandas as pd


import re


# PSS-10 scale mapping (0-4) — for forward-scored items (Q2, Q3, Q6)
_PSS_MAP = {
    "never": 0,
    "almost never": 1,
    "sometimes": 2,
    "fairly often": 3,
    "very often": 4,
}

# PSS-10 reverse mapping — for reverse-scored items (Q8, Q9, Q10)
_PSS_REVERSE_MAP = {
    "never": 4,
    "almost never": 3,
    "sometimes": 2,
    "fairly often": 1,
    "very often": 0,
}

# GAD-7 / PHQ-9 scale mapping (0-3)
_GAD_PHQ_MAP = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}

# Columns that are always numeric regardless of dtype
_NUMERIC_COL_PATTERNS = (
    "PSS_Q",
    "GAD_Q",
    "PHQ_Q",
    "PSS_Total",
    "GAD_Total",
    "PHQ_Total",
    "Children_Count",
)

# Known outcome positive labels (already handled by outcome.py, but keep for safety)
_KNOWN_OUTCOME_COLS = ("Outcome", "结局", "outcome", "Y", "y", "label", "target")


def _is_likely_label_row(series: pd.Series) -> bool:
    """Heuristic: if a row contains many long descriptive strings, it's likely a label row."""
    s = series.astype("string")
    non_na = s.dropna()
    if len(non_na) == 0:
        return False
    # Count how many cells look like descriptive labels (long text or repeated column names)
    long_text_count = sum(1 for v in non_na if len(str(v)) > 30)
    # If more than 30% are very long descriptions, likely label row
    return long_text_count / len(non_na) > 0.30


def skip_label_row(df: pd.DataFrame) -> pd.DataFrame:
    """Skip the first row if it appears to be a label/description row."""
    if len(df) <= 1:
        return df
    first_row = df.iloc[0]
    if _is_likely_label_row(first_row):
        return df.iloc[1:].reset_index(drop=True)
    return df


def _map_scale_value(val: object, mapping: dict[str, int]) -> int | object:
    """Map a single scale value using the provided mapping."""
    if pd.isna(val):
        return val
    # Strip regular whitespace + common unicode spaces (em-space, nbsp, etc.)
    s = re.sub(r"[\s\u2003\u00a0]+", " ", str(val)).strip().lower()
    if s in mapping:
        return mapping[s]
    # Try parsing as numeric directly
    try:
        return int(float(s))
    except ValueError:
        return val


def map_scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Map English scale descriptions to numeric scores for PSS/GAD/PHQ columns."""
    df = df.copy()

    # PSS columns — forward-scored vs reverse-scored
    pss_forward = {f"PSS_Q{i}" for i in (1, 2, 3, 4, 5, 6, 7)}
    pss_reverse = {f"PSS_Q{i}" for i in (8, 9, 10)}
    for col in [c for c in df.columns if c.startswith("PSS_Q")]:
        mapping = _PSS_REVERSE_MAP if col in pss_reverse else _PSS_MAP
        df[col] = df[col].apply(lambda v: _map_scale_value(v, mapping))

    # GAD columns
    gad_cols = [c for c in df.columns if c.startswith("GAD_Q")]
    for col in gad_cols:
        df[col] = df[col].apply(lambda v: _map_scale_value(v, _GAD_PHQ_MAP))

    # PHQ columns
    phq_cols = [c for c in df.columns if c.startswith("PHQ_Q")]
    for col in phq_cols:
        df[col] = df[col].apply(lambda v: _map_scale_value(v, _GAD_PHQ_MAP))

    return df


def clean_children_count(df: pd.DataFrame) -> pd.DataFrame:
    """Handle special string values in Children_Count."""
    if "Children_Count" not in df.columns:
        return df
    df = df.copy()

    def _conv(v):
        if pd.isna(v):
            return v
        s = str(v).strip().lower()
        if s in ("3 or more", "3+", ">=3"):
            return 3
        try:
            return int(float(s))
        except ValueError:
            return v

    df["Children_Count"] = df["Children_Count"].apply(_conv)
    return df


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns (including unicode spaces)."""
    df = df.copy()
    # Regex catches regular whitespace + em-space (\u2003) + non-breaking space (\u00a0) + thin space etc.
    _unicode_space_re = re.compile(r"[\s\u2003\u00a0\u2009\u200a]+")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: _unicode_space_re.sub(" ", str(x)).strip() if pd.notna(x) else x
            )
    return df


def coerce_numeric_columns(df: pd.DataFrame, numeric_patterns: tuple[str, ...] = _NUMERIC_COL_PATTERNS) -> pd.DataFrame:
    """Force columns matching known numeric patterns to numeric dtype."""
    df = df.copy()
    for col in df.columns:
        if any(col.startswith(p) or col == p for p in numeric_patterns):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the new Excel format.

    Steps:
      1. Skip label row if present
      2. Strip whitespace from strings
      3. Map scale scores (PSS/GAD/PHQ)
      4. Clean Children_Count
      5. Coerce known numeric columns
    """
    df = skip_label_row(df)
    df = strip_strings(df)
    df = map_scale_scores(df)
    df = clean_children_count(df)
    df = coerce_numeric_columns(df)
    return df
