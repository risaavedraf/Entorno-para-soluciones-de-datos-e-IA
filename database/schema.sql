-- =====================================================
-- Schema: Pipeline de Predicción de Precios Inmobiliarios
-- Dataset: Ames Housing Dataset (82 columnas, 2930 filas)
-- Asignatura: ITY1101 - Gestión de Datos para IA
-- Última actualización: 2026-03-26
-- =====================================================

-- Eliminar tabla si ya existe (útil para re-ingestas)
DROP TABLE IF EXISTS properties_raw;

-- Tabla principal con los datos en bruto desde el CSV
CREATE TABLE properties_raw (
    -- Identificadores
    order_id        INTEGER,
    pid             BIGINT,

    -- Caracteristicas del lote y zona
    ms_subclass     INTEGER,
    ms_zoning       VARCHAR(10),
    lot_frontage    NUMERIC(8,2),       -- Nulos posibles (490 registros)
    lot_area        INTEGER,
    street          VARCHAR(10),
    lot_shape       VARCHAR(5),
    neighborhood    VARCHAR(30),

    -- Calidad y construccion
    overall_qual    SMALLINT,           -- Escala 1-10. Variable clave.
    overall_cond    SMALLINT,
    year_built      SMALLINT,
    year_remod_add  SMALLINT,
    house_style     VARCHAR(15),
    bldg_type       VARCHAR(10),

    -- Superficie habitable (features primarias para el modelo)
    gr_liv_area     INTEGER,            -- Superficie habitable sobre tierra (pies cuadrados)
    total_bsmt_sf   NUMERIC(8,2),
    first_flr_sf    INTEGER,
    second_flr_sf   INTEGER,

    -- Baños y habitaciones
    full_bath       SMALLINT,
    half_bath       SMALLINT,
    bedroom_abvgr   SMALLINT,
    kitchen_abvgr   SMALLINT,
    totrms_abvgrd   SMALLINT,

    -- Garage
    garage_type     VARCHAR(20),        -- Nulos posibles (157 registros)
    garage_yr_blt   NUMERIC(6,1),
    garage_cars     NUMERIC(4,1),
    garage_area     NUMERIC(8,2),

    -- Amenidades exteriores
    wood_deck_sf    INTEGER,
    open_porch_sf   INTEGER,
    pool_area       INTEGER,

    -- Venta  (Variable Objetivo)
    mo_sold         SMALLINT,
    yr_sold         SMALLINT,
    sale_type       VARCHAR(20),
    sale_condition  VARCHAR(20),
    saleprice       INTEGER NOT NULL    -- Target: Precio de venta en dólares
);

-- =====================================================
-- Vista de datos limpios para entrenamiento del modelo
-- Excluye nulos en columnas criticas y filtra outliers
-- =====================================================
CREATE OR REPLACE VIEW vw_properties_clean AS
SELECT
    order_id,
    neighborhood,
    overall_qual,
    year_built,
    gr_liv_area,
    total_bsmt_sf,
    full_bath,
    bedroom_abvgr,
    COALESCE(garage_cars, 0)    AS garage_cars,
    COALESCE(garage_area, 0)    AS garage_area,
    COALESCE(lot_frontage, 0)   AS lot_frontage,
    saleprice
FROM properties_raw
WHERE
    saleprice > 0
    AND gr_liv_area < 4000      -- Eliminar outliers de superficie extrema
    AND saleprice < 500000;     -- Eliminar outliers de precio
