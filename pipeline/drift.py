"""Detección de drift con PSI, KS y Evidently."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from app.config import BASE_DIR, settings

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    dataset_drift: bool
    drift_share: float
    psi_scores: dict[str, float]
    ks_pvalues: dict[str, float]
    per_feature_drift: dict[str, bool]


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-4
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + eps
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + eps

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _compute_ks(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov test p-value. Low p-value means distributions differ."""
    if len(reference) < 2 or len(current) < 2:
        return 1.0
    _, p_value = stats.ks_2samp(reference, current)
    return float(p_value)


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftReport:
    """Compute PSI + KS for every shared numeric column.

    - PSI > 0.25 → significant drift (per reference doc threshold)
    - KS p-value < 0.05 → distributions differ significantly
    A feature is flagged as drifted if EITHER metric crosses its threshold.
    """
    common_numeric = [
        c
        for c in reference_df.select_dtypes(include="number").columns
        if c in current_df.columns
    ]

    psi_scores: dict[str, float] = {}
    ks_pvalues: dict[str, float] = {}
    per_feature_drift: dict[str, bool] = {}

    for col in common_numeric:
        ref = reference_df[col].dropna().to_numpy(dtype=float)
        cur = current_df[col].dropna().to_numpy(dtype=float)

        if ref.size < 2 or cur.size < 2:
            psi_scores[col] = 0.0
            ks_pvalues[col] = 1.0
            per_feature_drift[col] = False
            continue

        psi = _compute_psi(ref, cur)
        ks_p = _compute_ks(ref, cur)

        psi_scores[col] = round(psi, 6)
        ks_pvalues[col] = round(ks_p, 6)
        drifted = psi > 0.25 or ks_p < 0.05
        per_feature_drift[col] = drifted

        if drifted:
            logger.warning(
                "Drift detected in feature '%s': PSI=%.4f, KS_p=%.4f",
                col,
                psi,
                ks_p,
            )

    drifted_features = [f for f, d in per_feature_drift.items() if d]
    drift_share = len(drifted_features) / max(len(common_numeric), 1)
    dataset_drift = drift_share >= settings.DRIFT_THRESHOLD

    if dataset_drift:
        logger.error(
            "DATASET DRIFT detected (%.1f%% features drifted). Affected: %s",
            drift_share * 100,
            drifted_features,
        )

    # Best-effort Evidently enrichment (does NOT gate the report)
    try:
        from evidently import Report
        from evidently.metric_preset import DataDriftPreset

        evidently_report = Report(metrics=[DataDriftPreset()])
        evidently_report.run(reference_data=reference_df, current_data=current_df)
        logger.info("Evidently enrichment report generated successfully")
    except Exception:
        pass

    return DriftReport(
        dataset_drift=dataset_drift,
        drift_share=round(drift_share, 4),
        psi_scores=psi_scores,
        ks_pvalues=ks_pvalues,
        per_feature_drift=per_feature_drift,
    )


def save_drift_report(report: DriftReport, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or (BASE_DIR / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{ts}.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info("Drift report saved to %s", report_path)
    return report_path
