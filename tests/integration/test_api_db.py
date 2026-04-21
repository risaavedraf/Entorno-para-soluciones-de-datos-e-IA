from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.limpieza import create_features, encode_categoricals
from pipeline.validacion import validate_dataframe


def _train_basic_model() -> Pipeline:
    df = pd.DataFrame(
        {
            "overall_qual": [7, 8, 6, 5],
            "gr_liv_area": [1500.0, 1800.0, 1200.0, 1100.0],
            "total_bsmt_sf": [700.0, 0.0, 500.0, 400.0],
            "full_bath": [2, 2, 1, 1],
            "bedroom_abvgr": [3, 4, 2, 2],
            "garage_cars": [2.0, 2.0, 1.0, 1.0],
            "garage_area": [450.0, 500.0, 300.0, 250.0],
            "lot_frontage": [60.0, 70.0, 50.0, 45.0],
            "saleprice": [200000.0, 250000.0, 150000.0, 130000.0],
        }
    )
    X = df.drop(columns=["saleprice"])
    y = df["saleprice"]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model


def _payload() -> dict:
    return {
        "overall_qual": 7,
        "gr_liv_area": 1500,
        "total_bsmt_sf": 800,
        "full_bath": 2,
        "bedroom_abvgr": 3,
        "garage_cars": 2,
        "garage_area": 500,
        "lot_frontage": 60,
        "neighborhood": "NAmes",
    }


def test_predict_returns_valid_response_with_sklearn_fallback(tmp_path, monkeypatch):
    from app import main
    from app.dependencies import ModelManager

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(_train_basic_model(), models_dir / "model.joblib")
    (models_dir / "metadata.json").write_text(json.dumps({"version": "v-test", "r2": 0.9}), encoding="utf-8")

    monkeypatch.setattr("app.config.settings.MODELS_DIR", models_dir)
    monkeypatch.setattr("app.config.settings.ONNX_MODEL_PATH", models_dir / "model.onnx")

    manager = ModelManager()
    assert manager.runtime_type == "sklearn"

    main.app.state.model_manager = manager
    with TestClient(main.app) as client:
        response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["runtime_type"] == "sklearn"
    assert body["model_version"] == "v-test"


def test_predict_returns_valid_response_with_onnx_runtime(tmp_path, monkeypatch):
    from app.dependencies import ModelManager

    class FakeInput:
        name = "float_input"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, *_args, **_kwargs):
            return [np.array([[111111.0]], dtype=np.float32)]

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(_train_basic_model(), models_dir / "model.joblib")
    (models_dir / "model.onnx").write_bytes(b"fake-onnx")
    (models_dir / "metadata.json").write_text(json.dumps({"version": "v-onnx", "r2": 0.95}), encoding="utf-8")

    monkeypatch.setattr("app.config.settings.MODELS_DIR", models_dir)
    monkeypatch.setattr("app.config.settings.ONNX_MODEL_PATH", models_dir / "model.onnx")
    monkeypatch.setattr("app.dependencies._load_onnx_session", lambda _path: FakeSession())

    manager = ModelManager()
    result = manager.predict(pd.DataFrame([{"overall_qual": 7.0, "gr_liv_area": 1500.0}]))
    assert manager.runtime_type == "onnx"
    assert float(result[0]) == 111111.0


def test_health_returns_model_version_and_runtime(monkeypatch):
    from app import main

    class StubManager:
        runtime_type = "sklearn"
        model_version = "health-v1"

        def is_loaded(self):
            return True

    main.app.state.model_manager = StubManager()
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "health-v1"
    assert body["runtime_type"] == "sklearn"
    assert body["model_loaded"] is True


def test_end_to_end_ingest_validate_predict_flow(monkeypatch):
    from app import main

    raw = pd.DataFrame(
        {
            "overall_qual": [7],
            "gr_liv_area": [1500.0],
            "total_bsmt_sf": [700.0],
            "full_bath": [2],
            "bedroom_abvgr": [3],
            "garage_cars": [2.0],
            "garage_area": [450.0],
            "lot_frontage": [60.0],
            "neighborhood": ["NAmes"],
            "saleprice": [200000.0],
        }
    )
    result = validate_dataframe(raw)
    assert result.is_valid

    transformed = encode_categoricals(create_features(raw.drop(columns=["saleprice"])))

    class StubManager:
        runtime_type = "sklearn"
        model_version = "e2e-v1"

        def predict(self, df: pd.DataFrame):
            assert "ratio_area_banos" in df.columns
            assert "tiene_garage" in df.columns
            return np.array([222222.0])

        def is_loaded(self):
            return True

    main.app.state.model_manager = StubManager()
    with TestClient(main.app) as client:
        response = client.post("/predict", json=_payload())

    assert transformed.shape[0] == 1
    assert response.status_code == 200
    assert response.json()["precio_predicho"] == 222222.0
