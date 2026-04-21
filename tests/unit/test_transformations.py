from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.limpieza import align_to_feature_names, create_features, encode_categoricals


def test_ratio_area_banos_and_area_por_habitacion_are_calculated():
    df = pd.DataFrame(
        {
            "gr_liv_area": [1500.0],
            "full_bath": [2],
            "bedroom_abvgr": [3],
            "total_bsmt_sf": [700.0],
            "garage_cars": [2.0],
        }
    )

    out = create_features(df)
    assert out.loc[0, "ratio_area_banos"] == 500.0
    assert out.loc[0, "area_por_habitacion"] == 375.0


def test_tiene_sotano_and_tiene_garage_are_binary_flags():
    df = pd.DataFrame(
        {
            "gr_liv_area": [1000.0, 1000.0],
            "full_bath": [1, 1],
            "bedroom_abvgr": [2, 2],
            "total_bsmt_sf": [0.0, 100.0],
            "garage_cars": [0.0, 1.0],
        }
    )

    out = create_features(df)
    assert out["tiene_sotano"].tolist() == [0, 1]
    assert out["tiene_garage"].tolist() == [0, 1]


def test_feature_engineering_handles_zero_and_null_without_infinite_values():
    df = pd.DataFrame(
        {
            "gr_liv_area": [1000.0, None],
            "full_bath": [0, None],
            "bedroom_abvgr": [0, None],
            "total_bsmt_sf": [0.0, None],
            "garage_cars": [0.0, None],
            "neighborhood": ["NAmes", "CollgCr"],
        }
    )

    out = create_features(df)
    assert np.isfinite(out["ratio_area_banos"].fillna(0)).all()
    assert np.isfinite(out["area_por_habitacion"].fillna(0)).all()
    assert set(out["tiene_sotano"].dropna().astype(int).tolist()).issubset({0, 1})
    assert set(out["tiene_garage"].dropna().astype(int).tolist()).issubset({0, 1})


def test_one_hot_encoding_alignment_adds_missing_columns_and_orders_them():
    df = pd.DataFrame(
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
        }
    )
    encoded = encode_categoricals(create_features(df))
    expected_order = [
        "overall_qual",
        "gr_liv_area",
        "total_bsmt_sf",
        "full_bath",
        "bedroom_abvgr",
        "garage_cars",
        "garage_area",
        "lot_frontage",
        "ratio_area_banos",
        "area_por_habitacion",
        "tiene_sotano",
        "tiene_garage",
        "nbh_CollgCr",
        "nbh_NAmes",
    ]

    aligned = align_to_feature_names(encoded, expected_order)
    assert list(aligned.columns) == expected_order
    assert aligned.loc[0, "nbh_CollgCr"] == 0
