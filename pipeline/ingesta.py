"""Ingesta de datos crudos hacia PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import BASE_DIR, settings

CSV_PATH = BASE_DIR / "database" / "AmesHousing.csv"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

COLUMNAS_SELECCIONADAS = [
    "order",
    "pid",
    "ms_subclass",
    "ms_zoning",
    "lot_frontage",
    "lot_area",
    "street",
    "lot_shape",
    "neighborhood",
    "overall_qual",
    "overall_cond",
    "year_built",
    "year_remod_add",
    "house_style",
    "bldg_type",
    "gr_liv_area",
    "total_bsmt_sf",
    "1st_flr_sf",
    "2nd_flr_sf",
    "full_bath",
    "half_bath",
    "bedroom_abvgr",
    "kitchen_abvgr",
    "totrms_abvgrd",
    "garage_type",
    "garage_yr_blt",
    "garage_cars",
    "garage_area",
    "wood_deck_sf",
    "open_porch_sf",
    "pool_area",
    "mo_sold",
    "yr_sold",
    "sale_type",
    "sale_condition",
    "saleprice",
]

RENOMBRAR = {
    "order": "order_id",
    "1st_flr_sf": "first_flr_sf",
    "2nd_flr_sf": "second_flr_sf",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas para SQL-friendly naming."""
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return normalized


def build_raw_data_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Genera estadísticas mínimas para auditar ingesta."""
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "null_counts": {col: int(value) for col, value in df.isna().sum().to_dict().items()},
    }


def cargar_csv(ruta: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    return normalize_columns(df)


def filtrar_columnas(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols_existentes = [c for c in COLUMNAS_SELECCIONADAS if c in df_raw.columns]
    return df_raw[cols_existentes].rename(columns=RENOMBRAR)


def ejecutar_schema(engine, schema_path: Path = SCHEMA_PATH) -> None:
    with engine.connect() as conn:
        schema_sql = schema_path.read_text(encoding="utf-8")
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("--"):
                conn.execute(text(stmt))
        conn.commit()


def cargar_a_postgres(df: pd.DataFrame, engine) -> None:
    ejecutar_schema(engine)
    df.to_sql(name="properties_raw", con=engine, if_exists="append", index=False)


def log_raw_stats_to_mlflow(stats: dict[str, Any]) -> None:
    """Loguea stats como artifact JSON. No rompe si MLflow no está disponible."""
    try:
        import mlflow
    except Exception:
        return

    artifact_file = BASE_DIR / "reports" / "raw_data_stats.json"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ingesta"):
        mlflow.log_metrics(
            {
                "raw_rows": stats["row_count"],
                "raw_columns": stats["column_count"],
                "raw_total_nulls": int(sum(stats["null_counts"].values())),
            }
        )
        mlflow.log_artifact(str(artifact_file))


def run() -> pd.DataFrame:
    """Ejecuta pipeline de ingesta completo."""
    df_raw = cargar_csv(CSV_PATH)
    df_filtrado = filtrar_columnas(df_raw)
    engine = create_engine(settings.DATABASE_URL)
    cargar_a_postgres(df_filtrado, engine)

    stats = build_raw_data_stats(df_raw)
    log_raw_stats_to_mlflow(stats)
    return df_filtrado


if __name__ == "__main__":
    run()
