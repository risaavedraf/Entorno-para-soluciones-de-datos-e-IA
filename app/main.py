"""
app/main.py
-----------
API FastAPI para predicción de precios inmobiliarios.
Carga el modelo entrenado (Random Forest) y expone endpoint /predict.

Uso:
    uvicorn app.main:app --reload
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.dependencies import ModelManager
from app.logging_config import setup_logging
from pipeline.limpieza import align_to_feature_names, create_features, encode_categoricals

# ────────────────────────────────────────────────────
# 1. RUTAS Y CONFIGURACIÓN
# ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None


def get_logger():
    if structlog is None:
        return None
    return structlog.get_logger("app.main")


def _get_model_manager() -> ModelManager:
    manager = getattr(app.state, "model_manager", None)
    if manager is None:
        manager = ModelManager()
        app.state.model_manager = manager
    return manager


# ────────────────────────────────────────────────────
# 2. LIFESPAN: Cargar modelo al iniciar la app
# ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API."""
    setup_logging()
    logger = get_logger()
    if not hasattr(app.state, "model_manager"):
        app.state.model_manager = ModelManager()
    if logger:
        logger.info(
            "startup",
            model_loaded=app.state.model_manager.is_loaded(),
            runtime_type=app.state.model_manager.runtime_type,
            model_version=app.state.model_manager.model_version,
        )

    yield  # La app está corriendo
    if logger:
        logger.info("shutdown")


# ────────────────────────────────────────────────────
# 3. APP FASTAPI
# ────────────────────────────────────────────────────
app = FastAPI(
    title="API Predicción Precios Inmobiliarios",
    summary="Servicio REST para estimar precios de propiedades en Ames, Iowa",
    description=(
        "API para inferencia de precios inmobiliarios con FastAPI. "
        "Incluye predicción, health check y metadata del modelo activo."
    ),
    version="1.0.0",
    contact={
        "name": "Ricardo Saavedra",
        "url": "https://github.com/risaavedraf/Entorno-para-soluciones-de-datos-e-IA",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=[
        {"name": "Meta", "description": "Información general del servicio"},
        {"name": "Monitoring", "description": "Estado y salud operativa de la API"},
        {"name": "Model", "description": "Información del modelo desplegado"},
        {"name": "Predictions", "description": "Predicción de precios inmobiliarios"},
    ],
    servers=[
        {"url": "http://localhost:8000", "description": "Local"},
        {
            "url": "https://entorno-para-soluciones-de-datos-e-ia.onrender.com",
            "description": "Producción",
        },
    ],
    lifespan=lifespan,
)

# Montar archivos estáticos (frontend)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ────────────────────────────────────────────────────
# 4. MODELOS PYDANTIC (Validación de datos)
# ────────────────────────────────────────────────────
class PropiedadInput(BaseModel):
    """Input para predicción de precio de propiedad."""

    overall_qual: int = Field(..., ge=1, le=10, description="Calidad general (1-10)")
    gr_liv_area: int = Field(..., gt=0, description="Superficie habitable (pies²)")
    total_bsmt_sf: float = Field(0, ge=0, description="Superficie sótano (pies²)")
    full_bath: int = Field(0, ge=0, description="Baños completos")
    bedroom_abvgr: int = Field(0, ge=0, description="Habitaciones sobre tierra")
    garage_cars: float = Field(0, ge=0, description="Capacidad garage (autos)")
    garage_area: float = Field(0, ge=0, description="Superficie garage (pies²)")
    lot_frontage: float = Field(0, ge=0, description="Frente del lote (pies)")
    neighborhood: str = Field(..., description="Barrio (ej: 'NAmes', 'CollgCr')")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "overall_qual": 7,
                    "gr_liv_area": 1500,
                    "total_bsmt_sf": 800,
                    "full_bath": 2,
                    "bedroom_abvgr": 3,
                    "garage_cars": 2,
                    "garage_area": 500,
                    "lot_frontage": 60,
                    "neighborhood": "NAmes",
                }
            ]
        }
    }


class PrediccionOutput(BaseModel):
    """Output de la predicción."""

    precio_predicho: float = Field(..., description="Precio estimado en USD")
    precio_formateado: str = Field(..., description="Precio formateado")
    modelo_usado: str = Field(..., description="Nombre del modelo")
    confianza_r2: float = Field(..., description="R² del modelo (0-1)")
    model_version: str = Field(..., description="Versión de modelo activa")
    runtime_type: str = Field(..., description="Runtime activo (onnx/sklearn)")


class ErrorResponse(BaseModel):
    """Contrato estándar de error para respuestas no exitosas."""

    detail: str = Field(..., description="Descripción legible del error")


class ValidationErrorResponse(BaseModel):
    """Respuesta estandarizada para errores de validación de entrada."""

    detail: str = Field(..., description="Mensaje principal del error")
    errors: list[dict[str, Any]] = Field(
        ..., description="Lista detallada de errores de validación"
    )


class ApiEndpoints(BaseModel):
    """Catálogo de rutas públicas de la API."""

    predict: str = Field(..., description="Ruta de predicción")
    health: str = Field(..., description="Ruta de health check")
    info: str = Field(..., description="Ruta de metadata del modelo")
    docs: str = Field(..., description="Ruta de documentación Swagger")


class ApiInfoResponse(BaseModel):
    """Información general de la API."""

    mensaje: str = Field(..., description="Nombre o descripción del servicio")
    estado: str = Field(..., description="Estado general reportado por la API")
    modelo: str = Field(..., description="Versión o nombre del modelo activo")
    endpoints: ApiEndpoints = Field(..., description="Listado de endpoints disponibles")


class HealthResponse(BaseModel):
    """Payload de salud del servicio."""

    status: str = Field(..., description="Estado del servicio: ok o degraded")
    modelo_cargado: bool = Field(..., description="Indica si hay modelo listo para inferencia")
    model_loaded: bool = Field(..., description="Alias de compatibilidad para clients existentes")
    model_version: str = Field(..., description="Versión del modelo cargado")
    runtime_type: str = Field(..., description="Runtime activo (onnx/sklearn)")


class ModelInfoResponse(BaseModel):
    """Metadata del modelo. Permite campos extra para compatibilidad."""

    model_config = ConfigDict(extra="allow")

    runtime_type: str = Field(..., description="Runtime activo (onnx/sklearn)")
    model_version: str = Field(..., description="Versión del modelo cargado")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger = get_logger()
    if logger:
        logger.warning(
            "request_validation_failed",
            path=str(request.url.path),
            method=request.method,
            errors=exc.errors(),
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Error de validación en el request",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger = get_logger()
    if logger:
        logger.warning(
            "http_exception",
            path=str(request.url.path),
            method=request.method,
            status_code=exc.status_code,
            detail=str(exc.detail),
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger = get_logger()
    if logger:
        logger.exception(
            "unhandled_exception",
            path=str(request.url.path),
            method=request.method,
            error=str(exc),
        )
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})


# ────────────────────────────────────────────────────
# 5. ENDPOINTS
# ────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def read_root():
    """Sirve el frontend HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    # Fallback si no hay HTML
    return {"mensaje": "API de Predicción Inmobiliaria", "estado": "operativo"}


@app.get(
    "/api",
    tags=["Meta"],
    summary="Información general de la API",
    description="Devuelve estado, versión del modelo y catálogo de endpoints disponibles.",
    response_model=ApiInfoResponse,
    response_description="Información general del servicio",
)
def api_info():
    """Devuelve info de la API en JSON."""
    manager = _get_model_manager()
    return {
        "mensaje": "🏠 API de Predicción Inmobiliaria",
        "estado": "operativo",
        "modelo": manager.model_version,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "info": "GET /model/info",
            "docs": "GET /docs",
        },
    }


@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Health check del servicio",
    description="Indica disponibilidad de la API y estado de carga del modelo.",
    response_model=HealthResponse,
    response_description="Estado de salud del servicio",
)
def health_check():
    """Health check para Render y monitoreo."""
    manager = _get_model_manager()
    model_loaded = manager.is_loaded()
    return {
        "status": "ok" if model_loaded else "degraded",
        "modelo_cargado": model_loaded,
        "model_loaded": model_loaded,
        "model_version": manager.model_version,
        "runtime_type": manager.runtime_type,
    }


@app.get(
    "/model/info",
    tags=["Model"],
    summary="Metadata del modelo activo",
    description="Retorna métricas y metadatos del modelo actualmente desplegado.",
    response_model=ModelInfoResponse,
    response_description="Metadata del modelo",
    responses={
        404: {
            "description": "Modelo o metadata no encontrado",
            "model": ErrorResponse,
        }
    },
)
def model_info():
    """Devuelve información del modelo entrenado."""
    manager = _get_model_manager()
    metadata_path = settings.MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(
            status_code=404, detail="Modelo no encontrado. Ejecuta: python scripts/entrenamiento.py"
        )
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["runtime_type"] = manager.runtime_type
    metadata["model_version"] = manager.model_version
    return metadata


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Predecir precio de una propiedad",
    description=(
        "Recibe atributos de una propiedad residencial y devuelve un precio estimado en USD, "
        "junto a metadatos del modelo en uso."
    ),
    response_model=PrediccionOutput,
    response_description="Predicción generada correctamente",
    responses={
        400: {"description": "Error de datos o features inválidas", "model": ErrorResponse},
        422: {"description": "Error de validación de entrada", "model": ValidationErrorResponse},
        503: {"description": "Modelo no cargado", "model": ErrorResponse},
        500: {"description": "Error interno de predicción", "model": ErrorResponse},
    },
)
def predict(
    propiedad: PropiedadInput = Body(
        ...,
        examples={
            "caso_estandar": {
                "summary": "Propiedad residencial estándar",
                "value": {
                    "overall_qual": 7,
                    "gr_liv_area": 1500,
                    "total_bsmt_sf": 800,
                    "full_bath": 2,
                    "bedroom_abvgr": 3,
                    "garage_cars": 2,
                    "garage_area": 500,
                    "lot_frontage": 60,
                    "neighborhood": "NAmes",
                },
            }
        },
    ),
):
    """
    Predice el precio de una propiedad.

    Recibe las características de la propiedad y devuelve el precio estimado.
    """
    manager = _get_model_manager()
    if not manager.is_loaded():
        raise HTTPException(
            status_code=503, detail="Modelo no cargado. Ejecuta: python scripts/entrenamiento.py"
        )

    try:
        request_id = str(uuid4())
        start = time.perf_counter()
        # Convertir input a DataFrame
        datos = {
            "overall_qual": [propiedad.overall_qual],
            "gr_liv_area": [propiedad.gr_liv_area],
            "total_bsmt_sf": [propiedad.total_bsmt_sf],
            "full_bath": [propiedad.full_bath],
            "bedroom_abvgr": [propiedad.bedroom_abvgr],
            "garage_cars": [propiedad.garage_cars],
            "garage_area": [propiedad.garage_area],
            "lot_frontage": [propiedad.lot_frontage],
        }

        df = pd.DataFrame(datos)

        # Feature engineering (reuse training pipeline logic)
        df["neighborhood"] = propiedad.neighborhood
        df = create_features(df)
        df = encode_categoricals(df)

        sklearn_model = getattr(manager, "sklearn_model", None)
        if (
            manager.runtime_type == "sklearn"
            and sklearn_model is not None
            and hasattr(sklearn_model, "feature_names_in_")
        ):
            df = align_to_feature_names(df, list(sklearn_model.feature_names_in_))

        precio = manager.predict(df)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Formatear respuesta
        metadata_path = settings.MODELS_DIR / "metadata.json"
        r2 = 0.0
        modelo_nombre = "Random Forest"
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            r2 = float(metadata.get("r2", 0.0))
            modelo_nombre = str(metadata.get("modelo", "Random Forest"))

        logger = get_logger()
        if logger:
            logger.info(
                "prediction_completed",
                request_id=request_id,
                latency_ms=round(elapsed_ms, 3),
                model_version=manager.model_version,
                runtime_type=manager.runtime_type,
            )

        return PrediccionOutput(
            precio_predicho=round(float(precio), 2),
            precio_formateado=f"${precio:,.0f} USD",
            modelo_usado=modelo_nombre,
            confianza_r2=r2,
            model_version=manager.model_version,
            runtime_type=manager.runtime_type,
        )

    except ValueError as e:
        logger = get_logger()
        if logger:
            logger.warning("prediction_bad_request", error=str(e))
        raise HTTPException(
            status_code=400, detail="Datos de entrada inválidos para predecir"
        ) from e
    except RuntimeError as e:
        logger = get_logger()
        if logger:
            logger.warning("prediction_model_unavailable", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible temporalmente. Reentrena o reinicia el servicio.",
        ) from e
    except Exception as e:
        logger = get_logger()
        if logger:
            logger.exception("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error interno en la predicción") from e
