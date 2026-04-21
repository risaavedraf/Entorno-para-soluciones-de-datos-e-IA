from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.conversion import build_conversion_metadata, convert_model_to_onnx


def test_conversion_metadata_is_generated(tmp_path):
    metadata = build_conversion_metadata(tmp_path / "model.onnx")
    assert "converted_at" in metadata
    assert "sklearn_version" in metadata
    assert metadata["onnx_model_path"].endswith("model.onnx")


def test_sklearn_pipeline_converts_to_valid_onnx(tmp_path, trained_mock_pipeline):
    pytest.importorskip("skl2onnx")
    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    import joblib

    joblib.dump(trained_mock_pipeline, model_path)
    converted = convert_model_to_onnx(model_path=model_path, onnx_path=onnx_path)
    assert converted.exists()
    assert converted.suffix == ".onnx"
    meta = json.loads((tmp_path / "model_onnx_metadata.json").read_text(encoding="utf-8"))
    assert meta["onnx_model_path"].endswith("model.onnx")


def test_onnx_predictions_match_sklearn_within_tolerance(tmp_path, trained_mock_pipeline):
    pytest.importorskip("skl2onnx")
    try:
        import onnxruntime as ort
    except Exception:
        pytest.skip("onnxruntime no disponible")

    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    import joblib

    joblib.dump(trained_mock_pipeline, model_path)
    convert_model_to_onnx(model_path=model_path, onnx_path=onnx_path)

    X = pd.DataFrame(
        {
            "overall_qual": [7.0],
            "gr_liv_area": [1500.0],
            "total_bsmt_sf": [700.0],
            "full_bath": [2.0],
            "bedroom_abvgr": [3.0],
            "garage_cars": [2.0],
            "garage_area": [450.0],
            "lot_frontage": [60.0],
        }
    )
    sk_pred = trained_mock_pipeline.predict(X)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: X.to_numpy(dtype=np.float32)})[0].ravel()

    assert np.allclose(sk_pred, onnx_pred, rtol=1e-3)


def test_conversion_raises_runtime_error_for_invalid_model(tmp_path):
    if pytest.importorskip("skl2onnx") is None:
        return
    invalid_model_path = tmp_path / "invalid.joblib"
    invalid_model_path.write_text("not a model", encoding="utf-8")

    with pytest.raises(RuntimeError):
        convert_model_to_onnx(model_path=invalid_model_path, onnx_path=tmp_path / "model.onnx")


def test_conversion_raises_runtime_error_if_dependency_missing(tmp_path, monkeypatch):
    import builtins

    model_path = tmp_path / "model.joblib"
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    y = pd.Series([10.0, 12.0, 14.0])
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=2))]
    )
    model.fit(X, y)
    import joblib

    joblib.dump(model, model_path)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("skl2onnx"):
            raise ModuleNotFoundError("No module named 'skl2onnx'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        convert_model_to_onnx(model_path=model_path, onnx_path=tmp_path / "model.onnx")
