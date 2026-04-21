from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_ci_keeps_existing_core_steps_and_adds_phase5_gates():
    ci = _read(".github/workflows/ci.yml")

    # Existing core steps must remain
    assert "Run tests" in ci
    assert "python -m pytest tests/ -v --tb=short" in ci
    assert "Run ruff check" in ci
    assert "Verify Docker build" in ci

    # New compliance gates
    assert "Data Validation" in ci
    assert "Model Performance Gate" in ci
    assert "ONNX Conversion Test" in ci


def test_ci_model_gate_supports_current_and_future_metadata_keys():
    ci = _read(".github/workflows/ci.yml")

    assert "meta.get('r2_score'" in ci
    assert "meta.get('r2'" in ci
    assert "assert r2 >= 0.85" in ci


def test_cd_workflow_exists_with_test_and_render_deploy_jobs():
    cd_path = PROJECT_ROOT / ".github/workflows/cd.yml"
    assert cd_path.exists()

    cd = cd_path.read_text(encoding="utf-8")
    assert "name: CD - Deploy to Render" in cd
    assert "branches: [main]" in cd
    assert "workflow_dispatch" in cd
    assert "needs: test" in cd
    assert "Model Performance Gate" in cd
    assert "${{ secrets.RENDER_DEPLOY_HOOK }}" in cd


def test_dockerfile_remains_production_ready_with_pipeline_and_onnx_support():
    dockerfile = _read("dockerfile")

    assert "FROM python:3.12-slim" in dockerfile
    assert "COPY pipeline/ pipeline/" in dockerfile
    assert "COPY models/ models/" in dockerfile
    assert "model.onnx" in dockerfile
    assert "onnxruntime" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
