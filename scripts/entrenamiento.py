"""
scripts/entrenamiento.py
------------------------
Entrena un modelo de Regresión para predecir precios inmobiliarios.
Lee datos desde PostgreSQL (vista limpia), entrena, serializa el modelo.

Uso:
    python scripts/entrenamiento.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Scikit-Learn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ────────────────────────────────────────────────────
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/housing_db")

# Rutas
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPTS_DIR, "..")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "processed")


# ────────────────────────────────────────────────────
# 2. CARGAR DATOS
# ────────────────────────────────────────────────────
def cargar_datos(engine) -> pd.DataFrame:
    """Lee datos directamente desde la vista limpia de PostgreSQL."""
    print("📥 Cargando datos desde vw_properties_clean...")
    query = "SELECT * FROM vw_properties_clean"
    df = pd.read_sql(query, engine)
    print(f"✅ {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


# ────────────────────────────────────────────────────
# 3. PREPARAR FEATURES
# ────────────────────────────────────────────────────
def preparar_features(df: pd.DataFrame) -> tuple:
    """
    Prepara X e y para el modelo.
    Incluye encoding categórico y feature engineering.
    """
    print("🔧 Preparando features...")

    df = df.copy()

    # Feature engineering
    df['ratio_area_banos'] = df['gr_liv_area'] / (df['full_bath'] + 1)
    df['area_por_habitacion'] = df['gr_liv_area'] / (df['bedroom_abvgr'] + 1)
    df['tiene_sotano'] = (df['total_bsmt_sf'] > 0).astype(int)
    df['tiene_garage'] = (df['garage_cars'] > 0).astype(int)

    # One-Hot Encoding para neighborhood
    df = pd.get_dummies(df, columns=['neighborhood'], prefix='nbh', drop_first=True)

    # Separar X e y
    cols_excluir = ['order_id', 'saleprice']
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    y = df['saleprice']

    print(f"✅ X: {X.shape[1]} features, y: {y.name}")
    return X, y


# ────────────────────────────────────────────────────
# 4. ENTRENAR MODELOS
# ────────────────────────────────────────────────────
def entrenar_modelos(X_train, y_train, X_test, y_test) -> dict:
    """
    Entrena múltiples modelos y compara rendimiento.
    Esto demuestra que entendés de ML, no solo de código.
    """
    print("\n🏋️ Entrenando modelos...\n")

    modelos = {
        "Linear Regression": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Rellena NaN con mediana
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ]),
    }

    resultados = {}

    for nombre, pipeline in modelos.items():
        print(f"  🔄 Entrenando {nombre}...")

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Predecir
        y_pred = pipeline.predict(X_test)

        # Métricas
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        resultados[nombre] = {
            'pipeline': pipeline,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }

        print(f"     MAE:  ${mae:,.0f}")
        print(f"     RMSE: ${rmse:,.0f}")
        print(f"     R²:   {r2:.4f}")
        print()

    return resultados


# ────────────────────────────────────────────────────
# 5. SELECCIONAR Y GUARDAR EL MEJOR MODELO
# ────────────────────────────────────────────────────
def guardar_mejor_modelo(resultados: dict) -> str:
    """Selecciona el modelo con mejor R² y lo serializa."""
    # Encontrar el mejor (mayor R²)
    mejor_nombre = max(resultados, key=lambda k: resultados[k]['r2'])
    mejor_modelo = resultados[mejor_nombre]

    print(f"🏆 Mejor modelo: {mejor_nombre}")
    print(f"   R²: {mejor_modelo['r2']:.4f}")
    print(f"   MAE: ${mejor_modelo['mae']:,.0f}")

    # Crear directorio models/ si no existe
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Serializar con joblib
    modelo_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(mejor_modelo['pipeline'], modelo_path)
    print(f"\n💾 Modelo guardado en: {modelo_path}")

    # Guardar metadata del modelo
    metadata = {
        'modelo': mejor_nombre,
        'r2': mejor_modelo['r2'],
        'mae': mejor_modelo['mae'],
        'rmse': mejor_modelo['rmse'],
        'features': list(resultados[mejor_nombre]['pipeline'].feature_names_in_) if hasattr(resultados[mejor_nombre]['pipeline'], 'feature_names_in_') else []
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata guardada en: {metadata_path}")

    return modelo_path


# ────────────────────────────────────────────────────
# 6. MAIN
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        engine = create_engine(DATABASE_URL)
        print(f"🔌 Conectando a: {DATABASE_URL.split('@')[-1]}\n")

        # Pipeline completo
        df = cargar_datos(engine)
        X, y = preparar_features(df)

        # Split train/test (80/20)
        print("\n📊 Dividiendo datos: 80% train, 20% test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Train: {X_train.shape[0]} registros")
        print(f"   Test:  {X_test.shape[0]} registros")

        # Entrenar
        resultados = entrenar_modelos(X_train, y_train, X_test, y_test)

        # Guardar el mejor
        modelo_path = guardar_mejor_modelo(resultados)

        print("\n" + "="*50)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*50)
        print("\n🚀 Para usar el modelo, ejecutá:")
        print("   uvicorn app.main:app --reload")
        print("\n   POST /predict con JSON:")
        print('   {"overall_qual": 7, "gr_liv_area": 1500, ...}')

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        raise
