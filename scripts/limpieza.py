"""
scripts/limpieza.py
-------------------
Lee datos de la vista limpia de PostgreSQL y prepara features para el modelo.
Hace lo que SQL no puede: encoding categórico, escalamiento, feature engineering.

Uso:
    python scripts/limpieza.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ────────────────────────────────────────────────────
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/housing_db")


# ────────────────────────────────────────────────────
# 2. LEER DATOS DESDE LA VISTA LIMPIA
# ────────────────────────────────────────────────────
def leer_datos_limpios(engine) -> pd.DataFrame:
    """Lee la vista vw_properties_clean de PostgreSQL."""
    query = "SELECT * FROM vw_properties_clean"
    print("📥 Leyendo datos desde vw_properties_clean...")
    df = pd.read_sql(query, engine)
    print(f"✅ Datos leídos: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ────────────────────────────────────────────────────
def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables que mejoren el poder predictivo del modelo.
    Esto es lo que diferencia un modelo mediocre de uno bueno.
    """
    print("🔧 Creando features derivadas...")

    # Antigüedad de la casa al momento de venta
    df['antiguedad'] = df['yr_sold'] - df['year_built']
    # Nota: yr_sold y year_built no están en la vista limpia
    # Las agregamos desde properties_raw si es necesario

    # Ratio superficie habitable / baños
    df['ratio_area_banos'] = df['gr_liv_area'] / (df['full_bath'] + 1)  # +1 para evitar división por 0

    # Superficie por habitación
    df['area_por_habitacion'] = df['gr_liv_area'] / (df['bedroom_abvgr'] + 1)

    # Flag: tiene sótano
    df['tiene_sotano'] = (df['total_bsmt_sf'] > 0).astype(int)

    # Flag: tiene garage
    df['tiene_garage'] = (df['garage_cars'] > 0).astype(int)

    print(f"✅ Features creadas: {df.shape[1]} columnas totales")
    return df


# ────────────────────────────────────────────────────
# 4. ENCODING CATEGÓRICO
# ────────────────────────────────────────────────────
def encode_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas a numéricas.
    Scikit-learn no entiende strings — necesita números.
    """
    print("🔄 Encoding variables categóricas...")

    # One-Hot Encoding para neighborhood (muchas categorías)
    # Usamos get_dummies de pandas — simple y efectivo
    df_encoded = pd.get_dummies(df, columns=['neighborhood'], prefix='nbh', drop_first=True)

    print(f"✅ Encoding completo: {df_encoded.shape[1]} columnas")
    return df_encoded


# ────────────────────────────────────────────────────
# 5. SEPARAR FEATURES Y TARGET
# ────────────────────────────────────────────────────
def separar_xy(df: pd.DataFrame) -> tuple:
    """
    Separa el dataset en X (features) e y (target).
    order_id no es feature — es solo identificador.
    """
    print("✂️  Separando features (X) y target (y)...")

    # Columnas que NO son features
    cols_excluir = ['order_id', 'saleprice']

    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    y = df['saleprice']

    print(f"✅ X: {X.shape[1]} features, y: {y.name} (target)")
    return X, y


# ────────────────────────────────────────────────────
# 6. MAIN
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        engine = create_engine(DATABASE_URL)
        print(f"🔌 Conectando a: {DATABASE_URL.split('@')[-1]}")

        # Pipeline de limpieza
        df = leer_datos_limpios(engine)
        df = crear_features(df)
        df = encode_categoricas(df)
        X, y = separar_xy(df)

        # Guardar datos procesados localmente para entrenamiento
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
        os.makedirs(output_dir, exist_ok=True)

        X.to_csv(os.path.join(output_dir, "X.csv"), index=False)
        y.to_csv(os.path.join(output_dir, "y.csv"), index=False)

        print(f"\n💾 Datos guardados en {output_dir}/")
        print(f"   X.csv: {X.shape}")
        print(f"   y.csv: {y.shape}")
        print("\n✅ Limpieza completada. Listo para entrenamiento.")

    except Exception as e:
        print(f"❌ Error durante la limpieza: {e}")
        raise
