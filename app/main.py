from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensaje": "¡Entorno de IA configurado con éxito!", "estado": "operativo"}

@app.get("/health")
def health_check():
    return {"status": "ok"}