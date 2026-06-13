from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_project.cleaning import clean_dataset
from ml_project.outcome import coerce_binary_outcome, find_outcome_column


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    id_col: str | None
    outcome_col: str


def load_dataset(*, xlsx_path: Path, id_col_candidates: list[str] | None = None) -> Dataset:
    if not xlsx_path.exists():
        raise FileNotFoundError(str(xlsx_path))

    df = pd.read_excel(xlsx_path)
    if df.shape[0] == 0:
        raise ValueError("Empty dataset.")

    # Clean dataset for new Excel format (skip label row, map scales, etc.)
    df = clean_dataset(df)

    outcome_col = find_outcome_column(list(df.columns))
    df2 = df.copy()
    df2[outcome_col] = coerce_binary_outcome(df2[outcome_col])

    id_col: str | None = None
    for c in (id_col_candidates or ["原来", "编号", "ID", "Id", "id", "patient_id", "subject_id"]):
        if c in df2.columns:
            id_col = c
            break

    # Sort by ID if available so train_test_split is deterministic regardless of row order in Excel
    if id_col is not None:
        df2 = df2.sort_values(by=id_col, key=lambda col: pd.to_numeric(col, errors="coerce")).reset_index(drop=True)

    return Dataset(df=df2, id_col=id_col, outcome_col=outcome_col)

