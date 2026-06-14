from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelArtifacts:
    model_path: Path
    metadata_path: Path
    selected_features: list[str]
    best_threshold: float | None
    outcome_col: str | None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    display_name: str
    kind: str
    default_value: Any
    min_value: float | None
    max_value: float | None
    choices: list[Any] | None
    choice_values: list[Any] | None


_PSS_CHOICES = ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"]
_PSS_FORWARD_VALUES = [0, 1, 2, 3, 4]
_PSS_REVERSE_VALUES = [4, 3, 2, 1, 0]
_GAD_PHQ_CHOICES = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
_GAD_PHQ_VALUES = [0, 1, 2, 3]


def _scale_choice_spec(column_name: str) -> tuple[list[str], list[int]] | None:
    if column_name.startswith("PSS_Q"):
        if column_name in {f"PSS_Q{i}" for i in (8, 9, 10)}:
            return _PSS_CHOICES, _PSS_REVERSE_VALUES
        return _PSS_CHOICES, _PSS_FORWARD_VALUES
    if column_name.startswith(("GAD_Q", "PHQ_Q")):
        return _GAD_PHQ_CHOICES, _GAD_PHQ_VALUES
    return None


def resolve_data_path(
    *,
    base_dir: Path,
    extra_base_dirs: list[Path] | None = None,
    preferred_filenames: tuple[str, ...] = ("新版本数据.xlsx", "data.xlsx"),
) -> Path:
    search_dirs = [base_dir, *(extra_base_dirs or [])]
    candidate_dirs = []
    seen_dirs: set[Path] = set()
    for search_dir in search_dirs:
        for candidate_dir in (search_dir, search_dir / "data"):
            resolved = candidate_dir.resolve()
            if resolved not in seen_dirs:
                candidate_dirs.append(candidate_dir)
                seen_dirs.add(resolved)

    candidates = [candidate_dir / filename for filename in preferred_filenames for candidate_dir in candidate_dirs]
    return next((path for path in candidates if path.exists()), candidates[0])


def read_feature_display_names(path: Path) -> dict[str, str]:
    raw = pd.read_excel(path, header=None, nrows=2)
    if raw.shape[0] < 2 or raw.shape[1] == 0:
        return {}

    first_cell = str(raw.iloc[0, 0]).strip()
    second_cell = str(raw.iloc[1, 0]).strip()
    if first_cell != "原来" or second_cell != "现在":
        return {}

    labels: dict[str, str] = {}
    for old_name, display_name in zip(raw.iloc[0].tolist(), raw.iloc[1].tolist()):
        if pd.isna(old_name) or pd.isna(display_name):
            continue
        old = str(old_name).strip()
        display = str(display_name).strip()
        if old and display:
            labels[old] = display
    return labels


def _read_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid metadata JSON object: {path}")
    return data


def resolve_latest_model_artifacts(*, base_dir: Path, extra_base_dirs: list[Path] | None = None) -> ModelArtifacts:
    search_dirs = [base_dir, *(extra_base_dirs or [])]
    candidates = sorted(
        {root.resolve() for search_dir in search_dirs for root in search_dir.glob("outputs_*")},
        key=lambda p: p.name,
        reverse=True,
    )
    for root in candidates:
        model_path = root / "models" / "best_model.joblib"
        metadata_path = root / "models" / "best_model_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            continue
        md = _read_metadata(metadata_path)
        selected = [str(x) for x in md.get("selected_features", [])]
        if not selected:
            raise ValueError(f"selected_features is empty in {metadata_path}")
        threshold_raw = md.get("best_threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        outcome = md.get("outcome_col")
        return ModelArtifacts(
            model_path=model_path,
            metadata_path=metadata_path,
            selected_features=selected,
            best_threshold=threshold,
            outcome_col=(str(outcome) if outcome is not None else None),
        )
    raise FileNotFoundError(
        f"Cannot find model artifacts under {search_dirs}. Expected outputs_*/models/best_model.joblib and best_model_metadata.json."
    )


def _is_categorical_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").dropna().unique()
        if len(vals) == 0:
            return False
        return set(np.unique(vals)).issubset({0, 1})
    return True


def infer_feature_specs(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_display_names: dict[str, str] | None = None,
) -> list[FeatureSpec]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    specs: list[FeatureSpec] = []
    display_names = feature_display_names or {}
    for col in feature_cols:
        s = df[col]
        display_name = display_names.get(col, col)
        scale_choices = _scale_choice_spec(col)
        if scale_choices is not None:
            choices, values = scale_choices
            specs.append(
                FeatureSpec(
                    name=col,
                    display_name=display_name,
                    kind="categorical",
                    default_value=choices[0],
                    min_value=None,
                    max_value=None,
                    choices=list(choices),
                    choice_values=list(values),
                )
            )
        elif _is_categorical_series(s):
            raw_choices = s.dropna().unique().tolist()
            choices = sorted(raw_choices)
            default = choices[0] if choices else ""
            specs.append(
                FeatureSpec(
                    name=col,
                    display_name=display_name,
                    kind="categorical",
                    default_value=default,
                    min_value=None,
                    max_value=None,
                    choices=choices,
                    choice_values=None,
                )
            )
        else:
            num = pd.to_numeric(s, errors="coerce").dropna()
            if len(num) == 0:
                default = 0.0
                min_v = 0.0
                max_v = 1.0
            else:
                default = float(num.median())
                min_v = float(num.min())
                max_v = float(num.max())
                if np.isclose(min_v, max_v):
                    min_v -= 1.0
                    max_v += 1.0
            specs.append(
                FeatureSpec(
                    name=col,
                    display_name=display_name,
                    kind="numeric",
                    default_value=default,
                    min_value=min_v,
                    max_value=max_v,
                    choices=None,
                    choice_values=None,
                )
            )
    return specs
