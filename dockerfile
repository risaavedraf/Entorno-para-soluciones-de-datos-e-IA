FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render usa la variable PORT. Si no existe, usamos 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}