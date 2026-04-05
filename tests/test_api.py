"""
tests/test_api.py
-----------------
Tests para la API de Predicción Inmobiliaria.
Usa mocks para no depender del modelo real (aislamiento de tests).

Uso:
    pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np


# ────────────────────────────────────────────────────
# 1. FIXTURE: Mock del modelo
# ────────────────────────────────────────────────────
@pytest.fixture
def mock_model():
    """Crea un mock del pipeline de scikit-learn."""
    mock = MagicMock()

    # Simular que el modelo predice un precio
    mock.predict.return_value = np.array([175000.0])

    # Simular feature_names_in_ (columnas que espera el modelo)
    mock.feature_names_in_ = [
        'overall_qual', 'gr_liv_area', 'total_bsmt_sf', 'full_bath',
        'bedroom_abvgr', 'garage_cars', 'garage_area', 'lot_frontage',
        'ratio_area_banos', 'area_por_habitacion', 'tiene_sotano', 'tiene_garage',
        'nbh_Blueste', 'nbh_BrDale', 'nbh_BrkSide', 'nbh_ClearCr',
        'nbh_CollgCr', 'nbh_Crawfor', 'nbh_Edwards', 'nbh_Gilbert',
        'nbh_IDOTRR', 'nbh_MeadowV', 'nbh_Mitchel', 'nbh_NAmes',
        'nbh_NPkVill', 'nbh_NWAmes', 'nbh_NoRidge', 'nbh_NridgHt',
        'nbh_OldTown', 'nbh_SWISU', 'nbh_Sawyer', 'nbh_SawyerW',
        'nbh_Somerst', 'nbh_StoneBr', 'nbh_Timber', 'nbh_Veenker'
    ]

    return mock


@pytest.fixture
def mock_metadata():
    """Crea un mock de la metadata del modelo."""
    return {
        'modelo': 'Random Forest',
        'r2': 0.9649,
        'mae': 9210.0,
        'rmse': 13808.0
    }


@pytest.fixture
def client(mock_model, mock_metadata):
    """
    Crea un TestClient con el modelo mockeado.
    Esto aísla los tests del modelo real.
    """
    with patch('app.main.modelo', mock_model), \
         patch('app.main.metadata', mock_metadata):
        from app.main import app
        yield TestClient(app)


# ────────────────────────────────────────────────────
# 2. DATOS DE PRUEBA
# ────────────────────────────────────────────────────
VALID_INPUT = {
    "overall_qual": 7,
    "gr_liv_area": 1500,
    "total_bsmt_sf": 800,
    "full_bath": 2,
    "bedroom_abvgr": 3,
    "garage_cars": 2,
    "garage_area": 500,
    "lot_frontage": 60,
    "neighborhood": "NAmes"
}


# ────────────────────────────────────────────────────
# 3. TESTS DEL ENDPOINT RAÍZ
# ────────────────────────────────────────────────────
class TestRoot:
    def test_root_returns_200(self, client):
        """GET / debe retornar 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_mensaje(self, client):
        """GET / debe contener el campo 'mensaje'."""
        response = client.get("/")
        data = response.json()
        assert "mensaje" in data
        assert "API" in data["mensaje"]

    def test_root_contains_estado(self, client):
        """GET / debe contener estado operativo."""
        response = client.get("/")
        data = response.json()
        assert data["estado"] == "operativo"


# ────────────────────────────────────────────────────
# 4. TESTS DEL HEALTH CHECK
# ────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        """GET /health debe retornar 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self, client):
        """GET /health debe contener status ok cuando modelo está cargado."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["modelo_cargado"] is True


# ────────────────────────────────────────────────────
# 5. TESTS DEL ENDPOINT /predict
# ────────────────────────────────────────────────────
class TestPredict:
    def test_predict_returns_200(self, client):
        """POST /predict con datos válidos debe retornar 200."""
        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 200

    def test_predict_contains_precio(self, client):
        """POST /predict debe retornar precio_predicho."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "precio_predicho" in data
        assert isinstance(data["precio_predicho"], (int, float))
        assert data["precio_predicho"] > 0

    def test_predict_contains_formato(self, client):
        """POST /predict debe retornar precio_formateado."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "precio_formateado" in data
        assert "$" in data["precio_formateado"]
        assert "USD" in data["precio_formateado"]

    def test_predict_contains_modelo(self, client):
        """POST /predict debe retornar el modelo usado."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "modelo_usado" in data
        assert data["modelo_usado"] == "Random Forest"

    def test_predict_contains_confianza(self, client):
        """POST /predict debe retornar confianza R²."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "confianza_r2" in data
        assert 0 <= data["confianza_r2"] <= 1

    def test_predict_precio_rango_realista(self, client):
        """El precio debe estar en un rango realista ($35K - $500K)."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        precio = data["precio_predicho"]
        assert 35000 <= precio <= 500000, f"Precio {precio} fuera de rango"

    def test_predict_missing_field_returns_422(self, client):
        """POST /predict sin campo obligatorio debe retornar 422."""
        invalid_input = {
            "gr_liv_area": 1500,
            # Falta overall_qual (obligatorio)
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_invalid_qual_returns_422(self, client):
        """POST /predict con overall_qual fuera de rango debe retornar 422."""
        invalid_input = VALID_INPUT.copy()
        invalid_input["overall_qual"] = 15  # Debe ser 1-10
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_negative_area_returns_422(self, client):
        """POST /predict con superficie negativa debe retornar 422."""
        invalid_input = VALID_INPUT.copy()
        invalid_input["gr_liv_area"] = -100
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


# ────────────────────────────────────────────────────
# 6. TESTS DEL ENDPOINT /model/info
# ────────────────────────────────────────────────────
class TestModelInfo:
    def test_model_info_returns_200(self, client):
        """GET /model/info debe retornar 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_metrics(self, client):
        """GET /model/info debe contener métricas del modelo."""
        response = client.get("/model/info")
        data = response.json()
        assert "modelo" in data
        assert "r2" in data
        assert "mae" in data
        assert data["r2"] > 0.9  # Esperamos al menos 90%


# ────────────────────────────────────────────────────
# 7. TESTS DE DOCUMENTACIÓN
# ────────────────────────────────────────────────────
class TestDocs:
    def test_docs_available(self, client):
        """GET /docs debe retornar la documentación Swagger."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_available(self, client):
        """GET /openapi.json debe retornar el schema OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict" in data["paths"]


# ────────────────────────────────────────────────────
# 8. TEST: MODELO NO CARGADO
# ────────────────────────────────────────────────────
class TestModelNotLoaded:
    def test_predict_returns_503_without_model(self):
        """POST /predict sin modelo cargado debe retornar 503."""
        with patch('app.main.modelo', None):
            from app.main import app
            test_client = TestClient(app)
            response = test_client.post("/predict", json=VALID_INPUT)
            assert response.status_code == 503

    def test_model_info_returns_404_without_model(self):
        """GET /model/info sin metadata debe retornar 404."""
        with patch('app.main.metadata', None):
            from app.main import app
            test_client = TestClient(app)
            response = test_client.get("/model/info")
            assert response.status_code == 404
