# 🏠 Pipeline de Predicción de Precios Inmobiliarios
**Asignatura:** Gestión de Datos para IA (ITY1101) — DuocUC  
**Dataset:** Ames Housing Dataset (2930 propiedades, 82 variables)

## 📝 Descripción
Pipeline de gestión de datos e Inteligencia Artificial para la predicción del valor de mercado de propiedades inmobiliarias. El proyecto abarca desde la ingesta de datos desde un CSV público, su almacenamiento en PostgreSQL, hasta el entrenamiento de un modelo de Regresión y su exposición através de una API REST.

## 🏗️ Arquitectura
```
CSV (Ames Housing)
      │
      ▼ scripts/ingesta.py
      │
 [PostgreSQL]  ◄── database/schema.sql
      │
      ▼ scripts/entrenamiento.py
      │
 [Modelo .joblib]
      │
      ▼ app/main.py (FastAPI)
      │
   [Render]
```

## 📂 Estructura del Proyecto
```
/
├── app/
│   └── main.py              # API FastAPI (endpoints /predict y /health)
├── data/
│   └── AmesHousing.csv      # Dataset fuente (ignorado en git)
├── database/
│   └── schema.sql           # Esquema y vistas de PostgreSQL
├── docs/
│   ├── planificacion.md     # Planificación PMBOK del proyecto
│   └── diseno_tecnico.md    # Arquitectura y diseño técnico
├── models/                  # Modelo entrenado serializado (.joblib)
├── scripts/
│   ├── ingesta.py           # Carga CSV → PostgreSQL
│   ├── limpieza.py          # (Próximamente) Transformación de datos
│   └── entrenamiento.py     # (Próximamente) Entrenamiento del modelo
├── .env.example             # Plantilla de variables de entorno
├── .gitignore
├── dockerfile               # Imagen Docker para Render
├── requirements.txt         # Dependencias Python
└── README.md
```

## 🛠️ Stack Tecnológico
| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.9+ |
| API | FastAPI / Uvicorn |
| Base de Datos | PostgreSQL + SQLAlchemy |
| Machine Learning | Scikit-Learn, Pandas, NumPy |
| Contenerización | Docker |
| CI/CD | GitHub Actions |
| Despliegue | Render |

## 🚀 Cómo ejecutar localmente
### Prerrequisitos
- Python 3.9+
- PostgreSQL corriendo (local o Docker)

### 1. Clonar y configurar entorno
```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con los datos de tu base de datos local
```

### 3. Ejecutar ingesta de datos
```bash
python scripts/ingesta.py
```

### 4. Levantar la API
```bash
uvicorn app.main:app --reload
```

## 🌐 Despliegue en Render
La aplicación se despliega automáticamente en cada push a `main` mediante el Dockerfile.  
**URL:** [https://entorno-para-soluciones-de-datos-e-ia.onrender.com/](https://entorno-para-soluciones-de-datos-e-ia.onrender.com/)

## 🧪 CI/CD
Cada push a `main` ejecuta el pipeline de GitHub Actions que valida la instalación de dependencias e importación de la app.

## 📚 Documentación
- [Planificación PMBOK](docs/planificacion.md)
- [Diseño Técnico](docs/diseno_tecnico.md)
