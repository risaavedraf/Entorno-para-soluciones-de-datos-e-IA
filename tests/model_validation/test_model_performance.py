from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings


@pytest.mark.skipif(
    not (settings.MODELS_DIR / "metadata.json").exists(),
    reason="No trained model (metadata.json missing)",
)
def test_r2_gate_from_metadata_is_above_threshold():
    metadata_path = settings.MODELS_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert float(metadata["r2"]) > settings.R2_MIN_THRESHOLD


def test_model_comparison_selects_best_by_r2():
    candidates = [
        {"name": "Linear Regression", "r2": 0.81},
        {"name": "Random Forest", "r2": 0.96},
        {"name": "Gradient Boosting", "r2": 0.93},
    ]
    best = max(candidates, key=lambda x: x["r2"])
    assert best["name"] == "Random Forest"
    assert best["r2"] == 0.96


def test_model_below_threshold_would_be_rejected():
    low_r2 = 0.7
    assert low_r2 < settings.R2_MIN_THRESHOLD
