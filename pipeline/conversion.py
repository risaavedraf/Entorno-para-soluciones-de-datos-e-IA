"""Conversión de pipeline sklearn a ONNX."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from app.config import settings


def build_conversion_metadata(onnx_path: Path) -> dict[str, str]:
    return {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "onnx_model_path": str(onnx_path),
    }


def convert_model_to_onnx(model_path: Path | None = None, onnx_path: Path | None = None) -> Path:
    model_path = model_path or (settings.MODELS_DIR / "model.joblib")
    onnx_path = onnx_path or settings.ONNX_MODEL_PATH
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        model = joblib.load(model_path)
        n_features = len(getattr(model, "feature_names_in_", [])) or 1
        onnx_model = convert_sklearn(
            model, initial_types=[("float_input", FloatTensorType([None, n_features]))]
        )
        onnx_path.write_bytes(onnx_model.SerializeToString())

        metadata = build_conversion_metadata(onnx_path)
        (onnx_path.parent / "model_onnx_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return onnx_path
    except Exception as exc:
        raise RuntimeError(f"Error converting sklearn model to ONNX: {exc}") from exc


if __name__ == "__main__":
    convert_model_to_onnx()
