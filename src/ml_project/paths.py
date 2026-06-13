from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    models: Path

    report_pdf: Path
    results_xlsx: Path
    metrics_csv: Path


def make_output_paths(*, base_dir: Path) -> OutputPaths:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / f"outputs_{ts}"
    figures = root / "figures"
    tables = root / "tables"
    models = root / "models"

    return OutputPaths(
        root=root,
        figures=figures,
        tables=tables,
        models=models,
        report_pdf=root / "report.pdf",
        results_xlsx=root / "results.xlsx",
        metrics_csv=root / "metrics.csv",
    )

