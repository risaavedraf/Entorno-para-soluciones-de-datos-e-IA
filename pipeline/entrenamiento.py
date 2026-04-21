"""Entrenamiento con MLflow y gate de calidad (R²)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from app.config import settings
from pipeline.config import PROCESSED_DATA_PATH
from pipeline.limpieza import create_features, encode_categoricals, split_xy


def enforce_r2_gate(r2_value: float) -> None:
    if r2_value < settings.R2_MIN_THRESHOLD:
        raise ValueError(
            f"R²={r2_value:.4f} por debajo del mínimo requerido {settings.R2_MIN_THRESHOLD:.2f}"
        )


def _build_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    x_path = PROCESSED_DATA_PATH / "X.csv"
    y_path = PROCESSED_DATA_PATH / "y.csv"
    if x_path.exists() and y_path.exists():
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path).squeeze("columns")
        return X, y

    engine = create_engine(settings.DATABASE_URL)
    df = pd.read_sql("SELECT * FROM vw_properties_clean", engine)
    df = create_features(df)
    df = encode_categoricals(df)
    return split_xy(df)


def train_and_select_best(X: pd.DataFrame, y: pd.Series) -> tuple[str, dict[str, float], Pipeline]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results: dict[str, dict[str, float | Pipeline]] = {}

    for name, model in _build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "pipeline": model,
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    best_name = max(results, key=lambda k: float(results[k]["r2"]))
    best = results[best_name]
    metrics = {"mae": float(best["mae"]), "rmse": float(best["rmse"]), "r2": float(best["r2"])}
    return best_name, metrics, best["pipeline"]  # type: ignore[return-value]


def persist_model(
    best_name: str,
    metrics: dict[str, float],
    pipeline: Pipeline,
    feature_names: list[str] | None = None,
) -> Path:
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = settings.MODELS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Feature names from training data (for ONNX alignment)
    if feature_names is None:
        feature_names = (
            list(pipeline.named_steps["model"].feature_names_in_)
            if hasattr(pipeline.named_steps.get("model"), "feature_names_in_")
            else []
        )

    metadata = {
        "modelo": best_name,
        **metrics,
        "features": feature_names,
    }
    (settings.MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return model_path


def run() -> Path:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    X, y = load_training_data()

    with mlflow.start_run(run_name="entrenamiento"):
        best_name, metrics, best_pipeline = train_and_select_best(X, y)
        enforce_r2_gate(metrics["r2"])

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("r2_min_threshold", settings.R2_MIN_THRESHOLD)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        try:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name="housing_prices_model",
            )
        except Exception as exc:
            mlflow.set_tag("promotion_status", "failed")
            mlflow.set_tag("promotion_error", str(exc)[:250])
            raise RuntimeError(
                f"Model registration failed — run marked as not promotable: {exc}"
            ) from exc

        return persist_model(best_name, metrics, best_pipeline, feature_names=list(X.columns))


if __name__ == "__main__":
    run()
