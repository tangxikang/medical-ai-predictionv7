from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Table1Row:
    variable: str
    var_type: str
    group0: str
    group1: str
    p_value: float
    effect_size: float | None


def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        # Treat numeric as continuous by default; only obvious binary 0/1 as categorical.
        vals = pd.to_numeric(series, errors="coerce").dropna().unique()
        if vals.size == 0:
            return False
        if set(np.unique(vals)).issubset({0, 1}):
            return True
        return False
    return True


def _format_mean_sd(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def _format_count_pct(series: pd.Series) -> str:
    n = series.notna().sum()
    if n == 0:
        return "NA"
    vc = series.astype("string").value_counts(dropna=True)
    if len(vc) == 0:
        return "NA"
    # For Table 1 we keep the most frequent level in the summary cell; details are expanded later if needed.
    top_level = vc.index[0]
    top_n = int(vc.iloc[0])
    return f"{top_level}: {top_n} ({top_n / n:.1%})"


def _cohens_d(x0: np.ndarray, x1: np.ndarray) -> float | None:
    if x0.size < 2 or x1.size < 2:
        return None
    s0 = x0.std(ddof=1)
    s1 = x1.std(ddof=1)
    s_pooled = np.sqrt(((x0.size - 1) * s0**2 + (x1.size - 1) * s1**2) / (x0.size + x1.size - 2))
    if s_pooled == 0:
        return 0.0
    return float((x1.mean() - x0.mean()) / s_pooled)


def _cramers_v(table: np.ndarray) -> float | None:
    if table.size == 0:
        return None
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    if n == 0:
        return None
    r, k = table.shape
    denom = n * (min(k - 1, r - 1))
    if denom == 0:
        return None
    return float(np.sqrt(chi2 / denom))


def summarize_table1(*, df: pd.DataFrame, group_col: str, id_col: str | None) -> pd.DataFrame:
    """
    Create a Table 1 style summary comparing two groups defined by group_col.

    Returns columns:
      - variable
      - type (continuous/categorical)
      - group0 (summary string)
      - group1 (summary string)
      - p_value
      - effect_size (Cohen's d or Cramer's V)
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")

    work = df.copy(deep=False)
    if id_col is not None and id_col in work.columns:
        work = work.drop(columns=[id_col])

    group_vals = work[group_col].dropna().unique().tolist()
    if len(group_vals) != 2:
        raise ValueError(f"group_col must have 2 groups; got {group_vals}")
    group_vals_sorted = sorted(group_vals)
    g0, g1 = group_vals_sorted[0], group_vals_sorted[1]

    rows: list[Table1Row] = []
    for col in [c for c in work.columns if c != group_col]:
        s = work[col]
        s0 = work.loc[work[group_col] == g0, col]
        s1 = work.loc[work[group_col] == g1, col]

        if _is_categorical(s):
            # contingency table on observed levels
            x0 = s0.astype("string")
            x1 = s1.astype("string")
            levels = sorted(set(x0.dropna().tolist()) | set(x1.dropna().tolist()))
            if len(levels) == 0:
                continue

            ct = np.array(
                [
                    [(x0 == lv).sum() for lv in levels],
                    [(x1 == lv).sum() for lv in levels],
                ],
                dtype=int,
            )

            # Chi-square; fallback to Fisher for 2x2 with small expected counts
            p_val: float
            try:
                chi2, p_val, _, expected = stats.chi2_contingency(ct)
                if ct.shape == (2, 2) and (expected < 5).any():
                    _, p_val = stats.fisher_exact(ct)
            except ValueError:
                p_val = float("nan")

            rows.append(
                Table1Row(
                    variable=col,
                    var_type="categorical",
                    group0=_format_count_pct(s0),
                    group1=_format_count_pct(s1),
                    p_value=float(p_val),
                    effect_size=_cramers_v(ct),
                )
            )
        else:
            x0 = pd.to_numeric(s0, errors="coerce").dropna().to_numpy(dtype=float)
            x1 = pd.to_numeric(s1, errors="coerce").dropna().to_numpy(dtype=float)
            if x0.size == 0 or x1.size == 0:
                p_val = float("nan")
            else:
                _, p_val = stats.ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
            rows.append(
                Table1Row(
                    variable=col,
                    var_type="continuous",
                    group0=_format_mean_sd(s0),
                    group1=_format_mean_sd(s1),
                    p_value=float(p_val),
                    effect_size=_cohens_d(x0, x1),
                )
            )

    out = pd.DataFrame([r.__dict__ for r in rows])
    out = out.sort_values(by="p_value", ascending=True, na_position="last").reset_index(drop=True)
    return out


def table1_full(*, df: pd.DataFrame, group_col: str, id_col: str | None) -> pd.DataFrame:
    """
    Full Table 1 layout:
    - Continuous: one row per variable (mean ± sd in each group)
    - Categorical: one row per level (n (%)), p-value shown on the first level row
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")

    work = df.copy(deep=False)
    if id_col is not None and id_col in work.columns:
        work = work.drop(columns=[id_col])

    group_vals = sorted(work[group_col].dropna().unique().tolist())
    if len(group_vals) != 2:
        raise ValueError("group_col must have exactly 2 groups")
    g0, g1 = group_vals[0], group_vals[1]

    rows: list[dict[str, object]] = []
    summary = summarize_table1(df=work, group_col=group_col, id_col=None)
    p_map = {r["variable"]: float(r["p_value"]) for _, r in summary.iterrows()}

    for var in [c for c in work.columns if c != group_col]:
        s = work[var]
        s0 = work.loc[work[group_col] == g0, var]
        s1 = work.loc[work[group_col] == g1, var]

        if _is_categorical(s):
            s0s = s0.astype("string")
            s1s = s1.astype("string")
            levels = sorted(set(s0s.dropna().tolist()) | set(s1s.dropna().tolist()))
            n0 = int(s0s.notna().sum())
            n1 = int(s1s.notna().sum())
            first = True
            for lv in levels:
                c0 = int((s0s == lv).sum())
                c1 = int((s1s == lv).sum())
                rows.append(
                    {
                        "variable": var,
                        "level": lv,
                        f"{g0}": f"{c0} ({(c0 / n0 if n0 else float('nan')):.1%})",
                        f"{g1}": f"{c1} ({(c1 / n1 if n1 else float('nan')):.1%})",
                        "p_value": p_map.get(var) if first else np.nan,
                    }
                )
                first = False
        else:
            rows.append(
                {
                    "variable": var,
                    "level": "",
                    f"{g0}": _format_mean_sd(s0),
                    f"{g1}": _format_mean_sd(s1),
                    "p_value": p_map.get(var),
                }
            )

    return pd.DataFrame(rows)


def expand_categorical_details(*, df: pd.DataFrame, group_col: str, variable: str) -> pd.DataFrame:
    """
    Expand a categorical variable into per-level counts & percentages for both groups.
    """
    if variable not in df.columns:
        raise ValueError(f"variable not found: {variable}")
    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")

    group_vals = sorted(df[group_col].dropna().unique().tolist())
    if len(group_vals) != 2:
        raise ValueError("group_col must have 2 groups")
    g0, g1 = group_vals[0], group_vals[1]

    s0 = df.loc[df[group_col] == g0, variable].astype("string")
    s1 = df.loc[df[group_col] == g1, variable].astype("string")
    levels = sorted(set(s0.dropna().tolist()) | set(s1.dropna().tolist()))

    def _counts(series: pd.Series) -> dict[str, tuple[int, float]]:
        n = int(series.notna().sum())
        vc = series.value_counts(dropna=True)
        return {lv: (int(vc.get(lv, 0)), (int(vc.get(lv, 0)) / n if n else float("nan"))) for lv in levels}

    c0 = _counts(s0)
    c1 = _counts(s1)

    rows: list[dict[str, Any]] = []
    for lv in levels:
        n0, p0 = c0[lv]
        n1, p1 = c1[lv]
        rows.append({"variable": variable, "level": lv, f"{g0}_n": n0, f"{g0}_pct": p0, f"{g1}_n": n1, f"{g1}_pct": p1})
    return pd.DataFrame(rows)
