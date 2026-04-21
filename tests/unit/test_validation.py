from __future__ import annotations

from dataclasses import is_dataclass

import pandas as pd

from pipeline.validacion import (
    DuplicateReport,
    OutlierReport,
    ValidationResult,
    detect_duplicates,
    detect_outliers,
    validate_dataframe,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "overall_qual": [5, 6, 7],
            "gr_liv_area": [1000.0, 1100.0, 1200.0],
            "total_bsmt_sf": [0.0, 100.0, 200.0],
            "full_bath": [1, 2, 2],
            "bedroom_abvgr": [2, 3, 3],
            "garage_cars": [1.0, 2.0, 2.0],
            "garage_area": [200.0, 300.0, 350.0],
            "lot_frontage": [50.0, 60.0, 65.0],
            "neighborhood": ["NAmes", "CollgCr", "OldTown"],
            "saleprice": [100000.0, 150000.0, 200000.0],
        }
    )


def test_valid_dataframe_passes_schema_validation():
    result = validate_dataframe(_valid_df())
    assert result.is_valid is True
    assert result.validated_rows == 3
    assert result.errors == []


def test_invalid_dataframe_fails_on_nulls_or_missing_columns():
    invalid = _valid_df().drop(columns=["garage_area"]).copy()
    invalid.loc[0, "full_bath"] = None
    result = validate_dataframe(invalid)
    assert result.is_valid is False
    assert result.validated_rows == 0


def test_zscore_outlier_detection_detects_known_outlier():
    df = _valid_df()
    df.loc[3] = [10, 10000.0, 900.0, 6, 8, 5.0, 1500.0, 250.0, "NAmes", 900000.0]
    report = detect_outliers(df, ["gr_liv_area"])
    assert report.zscore_outliers["gr_liv_area"] >= 1


def test_iqr_outlier_detection_handles_edge_case_zero_iqr():
    df = _valid_df()
    df["gr_liv_area"] = 1000.0
    report = detect_outliers(df, ["gr_liv_area"])
    assert report.iqr_outliers["gr_liv_area"] == 0


def test_validation_result_and_outlier_report_are_dataclasses():
    assert is_dataclass(ValidationResult)
    assert is_dataclass(OutlierReport)
    assert is_dataclass(DuplicateReport)


def test_detect_duplicates_finds_exact_duplicate_rows():
    df = _valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate first row
    report = detect_duplicates(df)
    assert report.duplicate_rows == 1
    assert report.total_rows == 4
    assert report.duplicate_pct == 25.0


def test_detect_duplicates_no_duplicates():
    df = _valid_df()
    report = detect_duplicates(df)
    assert report.duplicate_rows == 0
    assert report.duplicate_pct == 0.0
