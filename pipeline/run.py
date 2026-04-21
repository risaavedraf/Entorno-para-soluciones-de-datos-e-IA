"""CLI para ejecutar pasos del pipeline MLOps."""

from __future__ import annotations

import argparse

from pipeline import conversion, drift, entrenamiento, ingesta, limpieza


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLOps pipeline runner")
    parser.add_argument(
        "step",
        choices=["ingest", "validate", "clean", "train", "convert", "drift", "full"],
        help="Pipeline step to run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.step in {"ingest", "full"}:
        ingesta.run()
    if args.step in {"clean", "full"}:
        limpieza.run()
    if args.step in {"train", "full"}:
        entrenamiento.run()
    if args.step in {"convert", "full"}:
        conversion.convert_model_to_onnx()
    if args.step in {"drift", "full"}:
        # Placeholder: caller should provide real datasets in production
        pass
    if args.step == "validate":
        # Validation is integrated in limpieza.run via Pandera
        limpieza.run()


if __name__ == "__main__":
    main()
