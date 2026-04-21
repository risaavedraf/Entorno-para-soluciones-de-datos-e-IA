"""Limpieza y preparación de features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from app.config import settings
from pipeline.config import PROCESSED_DATA_PATH
from pipeline.validacion import validate_dataframe


def read_clean_view(engine) -> pd.DataFrame:
    query = "SELECT * FROM vw_properties_clean"
    return pd.read_sql(query, engine)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ratio_area_banos"] = enriched["gr_liv_area"] / (enriched["full_bath"] + 1)
    enriched["area_por_habitacion"] = enriched["gr_liv_area"] / (enriched["bedroom_abvgr"] + 1)
    enriched["tiene_sotano"] = (enriched["total_bsmt_sf"] > 0).astype(int)
    enriched["tiene_garage"] = (enriched["garage_cars"] > 0).astype(int)
    return enriched


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["neighborhood"], prefix="nbh", drop_first=True)


def align_to_feature_names(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[feature_names]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols_excluir = ["order_id", "saleprice"]
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    y = df["saleprice"]
    return X, y


def run(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    engine = create_engine(settings.DATABASE_URL)
    df = read_clean_view(engine)

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise ValueError(f"Dataset no válido: {validation.errors}")

    df = create_features(df)
    df = encode_categoricals(df)
    X, y = split_xy(df)

    destination = output_dir or PROCESSED_DATA_PATH
    destination.mkdir(parents=True, exist_ok=True)
    X.to_csv(destination / "X.csv", index=False)
    y.to_csv(destination / "y.csv", index=False)
    return X, y


if __name__ == "__main__":
    run()
