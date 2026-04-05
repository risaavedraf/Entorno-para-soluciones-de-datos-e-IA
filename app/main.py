"""
app/main.py
-----------
API FastAPI para predicción de precios inmobiliarios.
Carga el modelo entrenado (Random Forest) y expone endpoint /predict.

Uso:
    uvicorn app.main:app --reload
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────
# 1. RUTAS Y CONFIGURACIÓN
# ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
STATIC_DIR = BASE_DIR / "static"

# Variables globales (se cargan al iniciar)
modelo = None
metadata = None


# ────────────────────────────────────────────────────
# 2. LIFESPAN: Cargar modelo al iniciar la app
# ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API."""
    global modelo, metadata

    print("🔄 Cargando modelo...")

    if not MODEL_PATH.exists():
        print(f"⚠️  Modelo no encontrado en: {MODEL_PATH}")
        print("   Ejecutá: python scripts/entrenamiento.py")
    else:
        modelo = joblib.load(MODEL_PATH)
        print(f"✅ Modelo cargado: {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"📋 Metadata cargada: {metadata['modelo']} (R²={metadata['r2']:.4f})")

    yield  # La app está corriendo

    print("👋 Cerrando API...")


# ────────────────────────────────────────────────────
# 3. APP FASTAPI
# ────────────────────────────────────────────────────
app = FastAPI(
    title="🏠 API Predicción Precios Inmobiliarios",
    description="Predice el valor de mercado de propiedades en Ames, Iowa usando Random Forest",
    version="1.0.0",
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


# ────────────────────────────────────────────────────
# 5. ENDPOINTS
# ────────────────────────────────────────────────────
@app.get("/")
def read_root():
    """Sirve el frontend HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    # Fallback si no hay HTML
    return {"mensaje": "API de Predicción Inmobiliaria", "estado": "operativo"}


@app.get("/api")
def api_info():
    """Devuelve info de la API en JSON."""
    return {
        "mensaje": "🏠 API de Predicción Inmobiliaria",
        "estado": "operativo",
        "modelo": metadata["modelo"] if metadata else "No cargado",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "info": "GET /model/info",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health_check():
    """Health check para Render y monitoreo."""
    modelo_cargado = modelo is not None
    return {"status": "ok" if modelo_cargado else "degraded", "modelo_cargado": modelo_cargado}


@app.get("/model/info")
def model_info():
    """Devuelve información del modelo entrenado."""
    if not metadata:
        raise HTTPException(
            status_code=404, detail="Modelo no encontrado. Ejecutá: python scripts/entrenamiento.py"
        )
    return metadata


@app.post("/predict", response_model=PrediccionOutput)
def predict(propiedad: PropiedadInput):
    """
    Predice el precio de una propiedad.

    Recibe las características de la propiedad y devuelve el precio estimado.
    """
    if modelo is None:
        raise HTTPException(
            status_code=503, detail="Modelo no cargado. Ejecutá: python scripts/entrenamiento.py"
        )

    try:
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

        # Feature engineering (igual que en entrenamiento.py)
        df["ratio_area_banos"] = df["gr_liv_area"] / (df["full_bath"] + 1)
        df["area_por_habitacion"] = df["gr_liv_area"] / (df["bedroom_abvgr"] + 1)
        df["tiene_sotano"] = (df["total_bsmt_sf"] > 0).astype(int)
        df["tiene_garage"] = (df["garage_cars"] > 0).astype(int)

        # One-Hot Encoding para neighborhood
        # Necesitamos TODAS las columnas que el modelo espera
        # Las columnas nbh_XXX se crean con get_dummies
        df["neighborhood"] = propiedad.neighborhood
        df = pd.get_dummies(df, columns=["neighborhood"], prefix="nbh", drop_first=True)

        # Asegurar que tenemos TODAS las columnas que el modelo espera
        # Las que faltan se llenan con 0 (no pertenece a ese barrio)
        if hasattr(modelo, "feature_names_in_"):
            for col in modelo.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            # Reordenar columnas según lo que espera el modelo
            df = df[modelo.feature_names_in_]

        # Predecir
        precio = modelo.predict(df)[0]

        # Formatear respuesta
        r2 = metadata["r2"] if metadata else 0.0
        modelo_nombre = metadata["modelo"] if metadata else "Random Forest"

        return PrediccionOutput(
            precio_predicho=round(float(precio), 2),
            precio_formateado=f"${precio:,.0f} USD",
            modelo_usado=modelo_nombre,
            confianza_r2=r2,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}") from e
