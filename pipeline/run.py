"""CLI para ejecutar pasos del pipeline MLOps."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_tabular(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Formato no soportado: {path}. Usá .csv o .parquet")


def _run_drift_step(reference_path: Path, current_path: Path, report_dir: Path | None) -> None:
    from pipeline import drift

    reference_df = _read_tabular(reference_path)
    current_df = _read_tabular(current_path)

    report = drift.detect_drift(reference_df=reference_df, current_df=current_df)
    output_path = drift.save_drift_report(report, output_dir=report_dir)

    print(
        f"[drift] dataset_drift={report.dataset_drift} "
        f"drift_share={report.drift_share:.4f} report={output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLOps pipeline runner")
    parser.add_argument(
        "step",
        choices=["ingest", "validate", "clean", "train", "convert", "drift", "full"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Ruta al dataset de referencia para drift (.csv/.parquet)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        help="Ruta al dataset actual para drift (.csv/.parquet)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directorio de salida para el reporte de drift (default: reports/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.step in {"ingest", "full"}:
        from pipeline import ingesta

        ingesta.run()
    if args.step in {"clean", "full"}:
        from pipeline import limpieza

        limpieza.run()
    if args.step in {"train", "full"}:
        from pipeline import entrenamiento

        entrenamiento.run()
    if args.step in {"convert", "full"}:
        from pipeline import conversion

        conversion.convert_model_to_onnx()
    if args.step in {"drift", "full"}:
        if args.reference is not None and args.current is not None:
            _run_drift_step(
                reference_path=args.reference,
                current_path=args.current,
                report_dir=args.report_dir,
            )
        elif args.step == "drift":
            raise SystemExit(
                "Para ejecutar drift debés pasar --reference y --current (csv/parquet)."
            )
        else:
            print(
                "[drift] omitido en full: pasá --reference y --current para activarlo."
            )
    if args.step == "validate":
        # Validation is integrated in limpieza.run via Pandera
        limpieza.run()


if __name__ == "__main__":
    main()
