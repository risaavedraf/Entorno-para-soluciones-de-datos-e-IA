from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/dbname"
    MLFLOW_TRACKING_URI: str = f"sqlite:///{BASE_DIR}/mlflow.db"
    MLFLOW_EXPERIMENT: str = "housing_prices"
    DRIFT_THRESHOLD: float = 0.1
    ONNX_MODEL_PATH: Path = BASE_DIR / "models" / "model.onnx"
    MODELS_DIR: Path = BASE_DIR / "models"
    R2_MIN_THRESHOLD: float = 0.85
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
