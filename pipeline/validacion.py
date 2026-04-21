"""Validación de datos con Pandera y detección de outliers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    validated_rows: int


@dataclass
class OutlierReport:
    zscore_outliers: dict[str, int]
    iqr_outliers: dict[str, int]
    total_outliers: int


@dataclass
class DuplicateReport:
    total_rows: int
    duplicate_rows: int
    duplicate_pct: float


REQUIRED_COLUMNS = {
    "overall_qual": int,
    "gr_liv_area": float,
    "total_bsmt_sf": float,
    "full_bath": int,
    "bedroom_abvgr": int,
    "garage_cars": float,
    "garage_area": float,
    "lot_frontage": float,
    "neighborhood": str,
    "saleprice": float,
}


def build_schema():
    try:
        import pandera.pandas as pa
    except Exception:
        return None

    return pa.DataFrameSchema(
        {
            "overall_qual": pa.Column(int, nullable=False),
            "gr_liv_area": pa.Column(float, nullable=False, coerce=True),
            "total_bsmt_sf": pa.Column(float, nullable=False, coerce=True),
            "full_bath": pa.Column(int, nullable=False),
            "bedroom_abvgr": pa.Column(int, nullable=False),
            "garage_cars": pa.Column(float, nullable=False, coerce=True),
            "garage_area": pa.Column(float, nullable=False, coerce=True),
            "lot_frontage": pa.Column(float, nullable=False, coerce=True),
            "neighborhood": pa.Column(str, nullable=False),
            "saleprice": pa.Column(float, nullable=False, coerce=True, checks=pa.Check.gt(0)),
        },
        strict=False,
    )


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    schema = build_schema()
    if schema is not None:
        try:
            validated = schema.validate(df)
        except Exception as exc:
            return ValidationResult(is_valid=False, errors=[str(exc)], validated_rows=0)
        return ValidationResult(is_valid=True, errors=[], validated_rows=int(validated.shape[0]))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(
            is_valid=False,
            errors=[f"Missing required columns: {', '.join(missing)}"],
            validated_rows=0,
        )

    if "saleprice" in df.columns:
        saleprice = pd.to_numeric(df["saleprice"], errors="coerce")
        if (saleprice <= 0).any():
            return ValidationResult(
                is_valid=False,
                errors=["saleprice must be greater than 0"],
                validated_rows=0,
            )

    return ValidationResult(is_valid=True, errors=[], validated_rows=int(df.shape[0]))


def _zscore_outliers(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0

    if clean.shape[0] < 10:
        median = float(clean.median())
        mad = float((clean - median).abs().median())
        if mad == 0:
            return 0
        modified_z = 0.6745 * (clean - median) / mad
        return int((modified_z.abs() > 3.5).sum())

    std = float(clean.std(ddof=0))
    if std == 0:
        return 0
    zscores = (clean - clean.mean()) / std
    return int((zscores.abs() > 3).sum())


def _iqr_outliers(series: pd.Series) -> int:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    return int(((series < lower) | (series > upper)).sum())


def detect_outliers(df: pd.DataFrame, numeric_columns: list[str]) -> OutlierReport:
    zscore_outliers: dict[str, int] = {}
    iqr_outliers: dict[str, int] = {}

    for column in numeric_columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            zscore_outliers[column] = 0
            iqr_outliers[column] = 0
            continue
        zscore_outliers[column] = _zscore_outliers(numeric)
        iqr_outliers[column] = _iqr_outliers(numeric)

    total = int(
        sum(max(zscore_outliers.get(k, 0), iqr_outliers.get(k, 0)) for k in zscore_outliers)
    )
    return OutlierReport(
        zscore_outliers=zscore_outliers,
        iqr_outliers=iqr_outliers,
        total_outliers=total,
    )


def detect_duplicates(df: pd.DataFrame) -> DuplicateReport:
    """Detect exact duplicate rows in the DataFrame."""
    total = len(df)
    dupes = int(df.duplicated().sum())
    pct = round(dupes / total * 100, 2) if total > 0 else 0.0
    return DuplicateReport(total_rows=total, duplicate_rows=dupes, duplicate_pct=pct)
