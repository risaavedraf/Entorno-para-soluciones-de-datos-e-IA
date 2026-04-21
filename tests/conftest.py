from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_housing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "overall_qual": [7, 8, 6],
            "gr_liv_area": [1500.0, 1800.0, 1200.0],
            "total_bsmt_sf": [700.0, 0.0, 500.0],
            "full_bath": [2, 2, 1],
            "bedroom_abvgr": [3, 4, 2],
            "garage_cars": [2.0, 2.0, 1.0],
            "garage_area": [450.0, 500.0, 300.0],
            "lot_frontage": [60.0, 70.0, 50.0],
            "neighborhood": ["NAmes", "CollgCr", "OldTown"],
            "saleprice": [200000.0, 250000.0, 150000.0],
        }
    )


@pytest.fixture
def trained_mock_pipeline(sample_housing_df: pd.DataFrame) -> Pipeline:
    X = sample_housing_df.drop(columns=["saleprice", "neighborhood"])
    y = sample_housing_df["saleprice"]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model


@pytest.fixture
def mock_onnx_session() -> Any:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        pytest.skip("onnxruntime no disponible")

    session = MagicMock()
    input_meta = MagicMock()
    input_meta.name = "float_input"
    session.get_inputs.return_value = [input_meta]
    session.run.return_value = [np.array([[123456.0]], dtype=np.float32)]
    return session


@pytest.fixture
def test_settings_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "metadata.json").write_text(
        json.dumps({"version": "test-v1", "modelo": "Random Forest", "r2": 0.91}),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.config.settings.MODELS_DIR", models_dir)
    monkeypatch.setattr("app.config.settings.ONNX_MODEL_PATH", models_dir / "model.onnx")
    return models_dir


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)
