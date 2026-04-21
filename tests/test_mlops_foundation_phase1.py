from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_include_mlops_dependencies():
    content = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    expected = {
        "mlflow",
        "onnxruntime",
        "skl2onnx",
        "pandera",
        "evidently",
        "structlog",
        "pydantic-settings",
    }

    lines = {line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")}
    for dep in expected:
        assert dep in lines


def test_config_settings_defaults_and_paths():
    from app.config import BASE_DIR, Settings

    s = Settings()
    assert BASE_DIR.exists()
    assert s.MLFLOW_EXPERIMENT == "housing_prices"
    assert s.DRIFT_THRESHOLD == 0.1
    assert s.R2_MIN_THRESHOLD == 0.85
    assert s.ONNX_MODEL_PATH.name == "model.onnx"
    assert s.MODELS_DIR.name == "models"


def test_logging_setup_uses_json_renderer_and_config_level():
    from app.logging_config import setup_logging

    setup_logging()
    assert setup_logging() is None


def test_phase1_directories_have_gitkeep():
    assert (PROJECT_ROOT / "mlruns" / ".gitkeep").exists()
    assert (PROJECT_ROOT / "reports" / ".gitkeep").exists()
    assert (PROJECT_ROOT / "logs" / ".gitkeep").exists()


def test_dockerfile_includes_pipeline_copy_and_onnxruntime_install():
    content = (PROJECT_ROOT / "dockerfile").read_text(encoding="utf-8")
    assert "onnxruntime" in content
    assert "COPY pipeline/ pipeline/" in content
