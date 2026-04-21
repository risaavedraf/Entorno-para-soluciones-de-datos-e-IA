from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from pipeline.limpieza import align_to_feature_names

logger = logging.getLogger(__name__)


def _load_onnx_session(model_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path))


class ModelManager:
    def __init__(self):
        self.onnx_session = None
        self.sklearn_model = None
        self.runtime_type: Literal["onnx", "sklearn"] = "sklearn"
        self.model_version = "unknown"
        self.feature_names: list[str] = []
        self.last_latency_ms = 0.0
        self._load()

    def _load(self):
        self._load_metadata()
        onnx_path = settings.ONNX_MODEL_PATH
        sklearn_path = settings.MODELS_DIR / "model.joblib"

        if onnx_path.exists():
            try:
                self.onnx_session = _load_onnx_session(onnx_path)
                self.runtime_type = "onnx"
                # Also load sklearn for feature names (lightweight, just metadata)
                if sklearn_path.exists() and not self.sklearn_model:
                    self.sklearn_model = joblib.load(sklearn_path)
            except Exception:
                logger.exception(
                    "Failed to load ONNX model from %s, falling back to sklearn", onnx_path
                )
                self.onnx_session = None

        if self.onnx_session is None and sklearn_path.exists():
            self.sklearn_model = joblib.load(sklearn_path)
            self.runtime_type = "sklearn"

    def _load_metadata(self):
        metadata_path = settings.MODELS_DIR / "metadata.json"
        if not metadata_path.exists():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_version = str(metadata.get("version") or metadata.get("modelo") or "unknown")
        self.feature_names = metadata.get("features") or []

    def is_loaded(self) -> bool:
        return self.onnx_session is not None or self.sklearn_model is not None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        start = time.perf_counter()
        try:
            # Determine feature names for alignment
            feature_names = self.feature_names
            if (
                not feature_names
                and self.sklearn_model is not None
                and hasattr(self.sklearn_model, "feature_names_in_")
            ):
                feature_names = list(self.sklearn_model.feature_names_in_)

            # Align features if we have known feature names
            if feature_names:
                df = align_to_feature_names(df, feature_names)

            if self.onnx_session is not None:
                input_name = self.onnx_session.get_inputs()[0].name
                input_data = df.to_numpy(dtype=np.float32)
                pred = self.onnx_session.run(None, {input_name: input_data})[0]
                return np.asarray(pred).reshape(-1)

            if self.sklearn_model is None:
                raise RuntimeError("No hay modelo cargado")

            return np.asarray(self.sklearn_model.predict(df))
        finally:
            self.last_latency_ms = (time.perf_counter() - start) * 1000
