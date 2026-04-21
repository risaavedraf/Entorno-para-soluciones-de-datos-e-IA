# Dockerfile
# ==========
# Imagen Docker para la API de Predicción Inmobiliaria
# Uso: docker build -t housing-api .

FROM python:3.12-slim AS base

# Instalar dependencias del sistema necesarias para psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root por seguridad
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copiar requirements primero (cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ app/
COPY pipeline/ pipeline/
COPY models/ models/
# model.onnx (if present) is included via models/ copy
COPY static/ static/

# Ensure ONNX runtime is explicitly present
RUN pip install --no-cache-dir onnxruntime

# Cambiar al usuario no-root
USER appuser

# Healthcheck para Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Render usa la variable PORT. Si no existe, usamos 8000.
EXPOSE ${PORT:-8000}

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
