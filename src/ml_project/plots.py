from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from ml_project.dca import dca_curve, treat_all_curve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from scipy.ndimage import gaussian_filter1d  # type: ignore
except Exception:  # pragma: no cover
    gaussian_filter1d = None

def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def _smooth_roc(fpr: np.ndarray, tpr: np.ndarray, *, n: int = 300, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    xs = np.linspace(0.0, 1.0, n)
    ys = np.interp(xs, fpr, tpr)
    if gaussian_filter1d is not None and sigma > 0:
        ys = gaussian_filter1d(ys, sigma=sigma)
    ys = np.clip(ys, 0.0, 1.0)
    return xs, ys


def plot_feature_k_curve(*, k_values: list[int], mean_auc: list[float], out_png: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, mean_auc, marker="o")
    plt.xlabel("Top-k features")
    plt.ylabel("CV ROC-AUC (RF)")
    plt.title("RF: choose best number of features")
    _savefig(out_png)


def plot_roc_multi(
    *,
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    out_png: Path,
    smooth: bool = True,
) -> None:
    plt.figure(figsize=(7, 6))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        if smooth:
            xs, ys = _smooth_roc(fpr, tpr)
            auc = np.trapz(ys, xs)
            plt.plot(xs, ys, lw=2, label=f"{name} (AUC={auc:.3f})")
        else:
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8, loc="lower right")
    _savefig(out_png)


def plot_dca_multi(*, y_true: np.ndarray, curves: dict[str, np.ndarray], out_png: Path) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, np.zeros_like(thresholds), label="Treat None", c="grey", lw=1, ls=":")
    plt.plot(thresholds, treat_all_curve(y_true=y_true, thresholds=thresholds), label="Treat All", c="black", lw=1, ls="--")

    for name, prob in curves.items():
        t, nb = dca_curve(y_true=y_true, y_prob=prob, thresholds=thresholds)
        plt.plot(t, nb, lw=2, label=name)

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis (Test set)")
    # Match common DCA presentation: avoid Treat-All extreme values compressing the y-axis.
    plt.ylim(-0.08, 0.20)
    plt.legend(fontsize=8, loc="best", ncol=2)
    _savefig(out_png)


def plot_calibration_multi(*, y_true: np.ndarray, curves: dict[str, np.ndarray], out_png: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="Perfectly calibrated")
    for name, prob in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curves (Test set)")
    plt.legend(fontsize=8, loc="best")
    _savefig(out_png)


def plot_confusion_matrices(
    *,
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    out_png: Path,
    threshold: float = 0.5,
) -> None:
    names = list(curves.keys())
    n = len(names)
    cols = 2
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, name in enumerate(names):
        prob = curves[name]
        y_pred = (prob >= threshold).astype(int)
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[i], colorbar=False, values_format="d")
        disp.ax_.set_title(name)
    for j in range(i + 1, axes.size):
        axes[j].axis("off")

    fig.suptitle("Confusion matrices (Test set, threshold=0.5)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def pdf_append_images(*, pdf: PdfPages, image_paths: list[Path], title: str | None = None) -> None:
    if title:
        plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.5, 0.5, title, ha="center", va="center", fontsize=18)
        pdf.savefig()
        plt.close()

    for p in image_paths:
        img = plt.imread(p)
        plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.imshow(img)
        pdf.savefig()
        plt.close()
