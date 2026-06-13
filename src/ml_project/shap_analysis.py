from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_project.preprocess import ensure_2d_dense, get_feature_names

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ShapArtifacts:
    best_model_name: str
    summary_beeswarm_png: Path
    summary_bar_png: Path
    individual_pngs: list[Path]


def _extract_binary_shap_values(exp: shap.Explanation) -> shap.Explanation:
    # For binary classification, shap can return an "output" dimension of size 2.
    # Prefer SHAP's native slicing to keep internal metadata consistent.
    if getattr(exp, "values", None) is None:
        return exp

    values = exp.values
    base_values = getattr(exp, "base_values", None)

    if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[-1] >= 2:
        return exp[..., 1]

    return exp


def run_shap(
    *,
    model_name: str,
    pipeline,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    seed: int = 42,
) -> ShapArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    background = df_train[feature_cols].sample(n=min(120, len(df_train)), random_state=seed)
    x_bg = ensure_2d_dense(pre.transform(background))

    x_test = df_test[feature_cols]
    x_test_trans = ensure_2d_dense(pre.transform(x_test))

    feature_names = get_feature_names(preprocessor=pre)
    explainer = shap.Explainer(model, x_bg, feature_names=feature_names)

    # keep runtime bounded
    max_rows = min(250, x_test_trans.shape[0])
    x_eval = x_test_trans[:max_rows]
    shap_exp = explainer(x_eval)
    shap_exp = _extract_binary_shap_values(shap_exp)

    beeswarm_png = out_dir / f"shap_summary_beeswarm_{model_name}.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_exp.values, features=shap_exp.data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_png, dpi=170, bbox_inches="tight")
    plt.close()

    bar_png = out_dir / f"shap_summary_bar_{model_name}.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_exp.values, features=shap_exp.data, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=170, bbox_inches="tight")
    plt.close()

    # Individual explanations: first 4 patients in test set
    individual_pngs: list[Path] = []
    n_ind = min(4, x_test_trans.shape[0])
    if n_ind > 0:
        shap_ind = explainer(x_test_trans[:n_ind])
        shap_ind = _extract_binary_shap_values(shap_ind)
        for i in range(n_ind):
            p = out_dir / f"shap_patient_{i+1}_{model_name}.png"
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(shap_ind[i], max_display=18, show=False)
            plt.tight_layout()
            plt.savefig(p, dpi=170, bbox_inches="tight")
            plt.close()
            individual_pngs.append(p)

    return ShapArtifacts(
        best_model_name=model_name,
        summary_beeswarm_png=beeswarm_png,
        summary_bar_png=bar_png,
        individual_pngs=individual_pngs,
    )
