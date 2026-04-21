from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_package_version_and_config_paths():
    from pipeline import __version__
    from pipeline.config import DATA_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, REFERENCE_DATA_PATH

    assert __version__ == "1.0.0"
    assert DATA_DIR.name == "data"
    assert RAW_DATA_PATH.name == "raw"
    assert PROCESSED_DATA_PATH.name == "processed"
    assert REFERENCE_DATA_PATH.name == "reference"


def test_ingesta_normalizes_columns_and_builds_stats():
    from pipeline.ingesta import build_raw_data_stats, normalize_columns

    df = pd.DataFrame({"Sale Price": [1, 2], "Lot/Area": [3, None]})
    normalized = normalize_columns(df)

    assert list(normalized.columns) == ["sale_price", "lot_area"]

    stats = build_raw_data_stats(normalized)
    assert stats["row_count"] == 2
    assert stats["column_count"] == 2
    assert stats["null_counts"]["lot_area"] == 1


def test_validacion_reports_schema_and_outliers():
    from pipeline.validacion import detect_outliers, validate_dataframe

    df = pd.DataFrame(
        {
            "overall_qual": [5, 6, 7, 100],
            "gr_liv_area": [1000, 1100, 1050, 5000],
            "total_bsmt_sf": [0.0, 100.0, 95.0, 600.0],
            "full_bath": [1, 2, 2, 10],
            "bedroom_abvgr": [2, 3, 3, 10],
            "garage_cars": [1.0, 1.0, 2.0, 5.0],
            "garage_area": [200.0, 250.0, 240.0, 1500.0],
            "lot_frontage": [50.0, 60.0, 55.0, 300.0],
            "neighborhood": ["NAmes", "NAmes", "CollgCr", "OldTown"],
            "saleprice": [100000, 120000, 125000, 500000],
        }
    )

    result = validate_dataframe(df)
    assert result.is_valid

    report = detect_outliers(df, ["gr_liv_area", "saleprice"])
    assert report.total_outliers >= 1
    assert "gr_liv_area" in report.zscore_outliers


def test_limpieza_creates_expected_features_and_xy():
    from pipeline.limpieza import create_features, encode_categoricals, split_xy

    df = pd.DataFrame(
        {
            "order_id": [1, 2],
            "neighborhood": ["NAmes", "CollgCr"],
            "overall_qual": [7, 8],
            "year_built": [2000, 2005],
            "gr_liv_area": [1500, 1700],
            "total_bsmt_sf": [700.0, 800.0],
            "full_bath": [2, 2],
            "bedroom_abvgr": [3, 4],
            "garage_cars": [2.0, 2.0],
            "garage_area": [450.0, 500.0],
            "lot_frontage": [60.0, 70.0],
            "saleprice": [200000, 250000],
        }
    )

    featured = create_features(df)
    assert "ratio_area_banos" in featured.columns
    assert "tiene_sotano" in featured.columns

    encoded = encode_categoricals(featured)
    assert any(col.startswith("nbh_") for col in encoded.columns)

    X, y = split_xy(encoded)
    assert "saleprice" not in X.columns
    assert y.name == "saleprice"


def test_entrenamiento_gate_blocks_low_r2():
    from pipeline.entrenamiento import enforce_r2_gate

    import pytest

    with pytest.raises(ValueError, match="R²"):
        enforce_r2_gate(0.2)


def test_conversion_metadata_structure(tmp_path):
    from pipeline.conversion import build_conversion_metadata

    metadata = build_conversion_metadata(tmp_path / "model.onnx")
    assert "converted_at" in metadata
    assert "sklearn_version" in metadata
    assert metadata["onnx_model_path"].endswith("model.onnx")


def test_drift_report_contains_feature_scores(tmp_path):
    from pipeline.drift import DriftReport, save_drift_report

    report = DriftReport(
        dataset_drift=False,
        drift_share=0.0,
        psi_scores={"gr_liv_area": 0.01},
        ks_pvalues={"gr_liv_area": 0.9},
        per_feature_drift={"gr_liv_area": False},
    )
    path = save_drift_report(report, output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"


def test_detect_drift_computes_real_psi_ks():
    import numpy as np
    from pipeline.drift import detect_drift

    rng = np.random.RandomState(42)
    reference = pd.DataFrame({"x": rng.normal(100, 10, 500), "y": rng.normal(50, 5, 500)})
    current = pd.DataFrame({"x": rng.normal(100, 10, 500), "y": rng.normal(50, 5, 500)})

    report = detect_drift(reference, current)
    assert "x" in report.psi_scores
    assert "y" in report.psi_scores
    assert report.psi_scores["x"] < 0.25  # similar distributions, low PSI
    assert report.dataset_drift is False


def test_detect_drift_catches_significant_shift():
    import numpy as np
    from pipeline.drift import detect_drift

    rng = np.random.RandomState(42)
    reference = pd.DataFrame({"x": rng.normal(100, 10, 500)})
    current = pd.DataFrame({"x": rng.normal(200, 10, 500)})  # big shift

    report = detect_drift(reference, current)
    assert report.psi_scores["x"] > 0.25  # significant PSI
    assert report.per_feature_drift["x"] is True


def test_run_cli_supports_full_and_step_commands():
    from pipeline.run import build_parser

    parser = build_parser()
    assert parser.parse_args(["ingest"]).step == "ingest"
    assert parser.parse_args(["full"]).step == "full"


def test_scripts_are_thin_wrappers():
    ingest_script = (PROJECT_ROOT / "scripts" / "ingesta.py").read_text(encoding="utf-8")
    clean_script = (PROJECT_ROOT / "scripts" / "limpieza.py").read_text(encoding="utf-8")
    train_script = (PROJECT_ROOT / "scripts" / "entrenamiento.py").read_text(encoding="utf-8")

    assert "pipeline.ingesta" in ingest_script
    assert "pipeline.limpieza" in clean_script
    assert "pipeline.entrenamiento" in train_script
