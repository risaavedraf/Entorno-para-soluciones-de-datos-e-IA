from __future__ import annotations

import pandas as pd

from pipeline.validacion import validate_dataframe


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "overall_qual": [5, 6],
            "gr_liv_area": [1000.0, 1200.0],
            "total_bsmt_sf": [0.0, 200.0],
            "full_bath": [1, 2],
            "bedroom_abvgr": [2, 3],
            "garage_cars": [1.0, 2.0],
            "garage_area": [200.0, 300.0],
            "lot_frontage": [50.0, 60.0],
            "neighborhood": ["NAmes", "CollgCr"],
            "saleprice": [100000.0, 150000.0],
        }
    )


def test_pandera_schema_contract_accepts_valid_dataframe():
    result = validate_dataframe(_valid_df())
    assert result.is_valid
    assert result.validated_rows == 2


def test_invalid_dataframe_rejected_for_wrong_schema_and_nulls():
    invalid = _valid_df().copy()
    invalid = invalid.drop(columns=["garage_area"])
    invalid.loc[0, "overall_qual"] = None

    result = validate_dataframe(invalid)
    assert result.is_valid is False


def test_column_range_checks_saleprice_positive():
    invalid = _valid_df().copy()
    invalid.loc[0, "saleprice"] = -10.0

    result = validate_dataframe(invalid)
    assert result.is_valid is False
