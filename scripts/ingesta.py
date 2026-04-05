"""
scripts/ingesta.py
------------------
Carga el dataset AmesHousing.csv en la tabla 'properties_raw' de PostgreSQL.
Lee la conexión desde variables de entorno (.env) para ser portable.

Uso:
    python scripts/ingesta.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE CONEXIÓN
# ────────────────────────────────────────────────────
load_dotenv()  # Lee el archivo .env automáticamente

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/housing_db")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "database", "AmesHousing.csv")


# ────────────────────────────────────────────────────
# 2. LECTURA Y NORMALIZACIÓN DEL CSV
# ────────────────────────────────────────────────────
def cargar_csv(ruta: str) -> pd.DataFrame:
    """Lee el CSV y normaliza los nombres de columna para SQL."""
    print(f"📂 Leyendo dataset desde: {ruta}")
    df = pd.read_csv(ruta)

    # Limpiar nombres de columna: minúsculas, sin espacios ni caracteres especiales
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


# ────────────────────────────────────────────────────
# 3. SELECCIÓN DE COLUMNAS RELEVANTES
# ────────────────────────────────────────────────────
COLUMNAS_SELECCIONADAS = [
    "order", "pid",
    "ms_subclass", "ms_zoning", "lot_frontage", "lot_area", "street",
    "lot_shape", "neighborhood", "overall_qual", "overall_cond",
    "year_built", "year_remod_add", "house_style", "bldg_type",
    "gr_liv_area", "total_bsmt_sf", "1st_flr_sf", "2nd_flr_sf",
    "full_bath", "half_bath", "bedroom_abvgr", "kitchen_abvgr", "totrms_abvgrd",
    "garage_type", "garage_yr_blt", "garage_cars", "garage_area",
    "wood_deck_sf", "open_porch_sf", "pool_area",
    "mo_sold", "yr_sold", "sale_type", "sale_condition", "saleprice"
]

# Mapeo para alinear nombres del CSV con el schema.sql
RENOMBRAR = {
    "order":        "order_id",
    "1st_flr_sf":   "first_flr_sf",
    "2nd_flr_sf":   "second_flr_sf",
}


# ────────────────────────────────────────────────────
# 4. CARGA A POSTGRESQL
# ────────────────────────────────────────────────────
def cargar_a_postgres(df: pd.DataFrame, engine) -> None:
    """Elimina la tabla anterior y sube los datos frescos."""
    with engine.connect() as conn:
        print("🗑️  Ejecutando schema.sql (DROP + CREATE TABLE)...")
        schema_path = os.path.join(os.path.dirname(__file__), "..", "database", "schema.sql")
        with open(schema_path, encoding="utf-8") as f:
            schema_sql = f.read()
        # Ejecutar cada statement del schema por separado
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            # Ignorar: vacíos, solo comentarios (-- ...), o solo espacios
            if stmt and not stmt.startswith("--"):
                conn.execute(text(stmt))
        conn.commit()

    print("📤 Cargando datos en 'properties_raw'...")
    df.to_sql(
        name="properties_raw",
        con=engine,
        if_exists="append",  # El schema.sql ya recreó la tabla limpia
        index=False
    )
    print(f"✅ {len(df)} registros cargados exitosamente en PostgreSQL.")


# ────────────────────────────────────────────────────
# 5. MAIN
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df_raw = cargar_csv(DATA_PATH)

        # Filtrar solo las columnas que existen en el CSV
        cols_existentes = [c for c in COLUMNAS_SELECCIONADAS if c in df_raw.columns]
        df_filtrado = df_raw[cols_existentes].rename(columns=RENOMBRAR)

        engine = create_engine(DATABASE_URL)
        print(f"🔌 Conectando a: {DATABASE_URL.split('@')[-1]}")  # Oculta credenciales en el log

        cargar_a_postgres(df_filtrado, engine)

    except Exception as e:
        print(f"❌ Error durante la ingesta: {e}")
        raise
