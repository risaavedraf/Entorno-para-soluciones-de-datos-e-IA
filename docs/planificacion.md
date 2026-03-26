# Documento de Planificación del Proyecto (ITY1101)
_Última actualización: 2026-03-26_

## 1. Introducción
El presente proyecto tiene como objetivo desarrollar un **pipeline de gestión de datos** automatizado y un modelo de Inteligencia Artificial (IA) para la predicción del valor de mercado de propiedades inmobiliarias. El foco principal de esta solución es la **gestión eficiente, persistente y estructurada de los datos** utilizando PostgreSQL como núcleo de almacenamiento para el posterior modelaje predictivo.

## 2. Alcance del Proyecto
### Qué incluye:
* **Ingesta de Datos**: Obtención de un dataset público (formato CSV) y su carga automatizada en un motor relacional.
* **Gestión de Base de Datos**: Creación de un esquema en **PostgreSQL** para asegurar la integridad y persistencia de la información.
* **Data Wrangling**: Limpieza y transformación de datos (manejo de nulos, outliers y codificación) directamente involucrando la base de datos.
* **Modelo Predictivo**: Entrenamiento de un modelo de Regresión que consume los datos **directamente desde PostgreSQL**.
* **API de Consulta**: Implementación de una API (FastAPI) para obtener predicciones basadas en el modelo entrenado.
* **Contenerización y Despliegue**: Uso de Docker para asegurar la portabilidad entre el entorno local y Render (nube).

### Qué NO incluye:
* Recolección manual o web scraping de sitios inmobiliarios.
* Desarrollo de un frontend complejo para usuarios finales.

## 3. Entregables del Proyecto
1. Repositorio GitHub con historial de versiones.
2. Documentación técnica de planificación (PMBOK) y diseño de datos.
3. Scripts de Ingesta (`scripts/ingesta.py`) y Carga (`database/schema.sql`).
4. Script de Entrenamiento que consume de SQL (`scripts/entrenamiento.py`).
5. Modelo serializado (`models/model.joblib`).
6. API operativa en Render (URL pública).

## 4. Cronograma (WBS / EDT)
| Semana | Fase | Actividad Clave | Hito |
|---|---|---|---|
| 1 | Planificación y Diseño | Definición de Alcance y Diseño de Esquema SQL | Hito 1: Plan Aprobado |
| 2 | Ingesta y Carga | **Carga de CSV a PostgreSQL** mediante Python | Hito 2: DB Poblada |
| 3 | Limpieza y Análisis | Transformación de datos en DB y Análisis Exploratorio | Hito 3: Datos Limpios |
| 4 | Modelado y Entrenamiento | Entrenamiento de Regresión **leyendo de SQL** | Hito 4: Modelo Entrenado |
| 5 | Despliegue y Cierre | Integración en FastAPI, Docker y Despliegue en Render | Hito 5: Entrega Final |

## 5. Riesgos y Mitigación
1. **Pérdida de Integridad en Carga SQL**: Errores de tipos de datos al pasar de CSV a Postgres. *Mitigación: Validación de tipos con Pandas antes de la carga.*
2. **Entornos Diferentes**: Diferencias entre el Postgres local y el de Render. *Mitigación: Uso de variables de entorno y Docker (Variables `DATABASE_URL`).*
3. **Métrica de Error Elevada**: Bajo rendimiento del modelo inicial. *Mitigación: Fine-tuning de hiperparámetros y feature engineering en SQL.*
