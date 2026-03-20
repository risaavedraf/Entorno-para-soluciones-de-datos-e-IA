# Entorno Técnico para Soluciones de Datos e IA
**Asignatura:** Gestión de Datos para IA (ITY1101)  

## 📝 Descripción
Este proyecto consiste en la configuración de un entorno técnico completo en la nube para el desarrollo de soluciones de IA. Se integra el control de versiones, la contenerización con Docker, automatización CI/CD y despliegue continuo para asegurar escalabilidad y trazabilidad.

## 🛠️ Herramientas Utilizadas
* **GitHub:** Control de versiones y repositorio central.
* **Docker:** Contenerización de la aplicación para asegurar que funcione en cualquier entorno.
* **GitHub Actions:** Pipeline de CI/CD para automatización de pruebas e integración.
* **Render:** Plataforma de despliegue en la nube para la API.
* **FastAPI:** Framework de Python utilizado para la aplicación simple.

## 🚀 Despliegue
La aplicación se encuentra operativa en la siguiente URL:
> **[[TU_URL_AQUÍ](https://entorno-para-soluciones-de-datos-e-ia.onrender.com/)]**

## 📂 Estructura del Proyecto
* `app/`: Contiene la lógica de la API (`main.py`).
* `Dockerfile`: Instrucciones para la imagen de contenedor.
* `.github/workflows/`: Configuración de la automatización CI/CD.
* `.env.example`: Plantilla de variables de entorno.
* `.gitignore`: Archivos excluidos del control de versiones.

Imagen
<img width="3334" height="491" alt="imagen" src="https://github.com/user-attachments/assets/23b19361-2c9e-45b6-8381-082d8635a0d6" />


## 🧪 Decisiones Técnicas

1. **Enfoque en Contenedores:** Se utilizó Docker para garantizar la portabilidad de la solución de IA.
2. **Automatización:** Se implementó un flujo en GitHub Actions que valida el código en cada `push`.
3. **Escalabilidad:** Se eligió Render por su facilidad para escalar servicios web basados en contenedores.

## 🔍 Alternativas Investigadas
Como parte del trabajo autónomo, se identificaron las siguientes alternativas para el despliegue:
* **Vercel / Railway:** Para despliegues rápidos de aplicaciones web.
* **Fly.io:** Enfocado en despliegues globales cerca del usuario.
* **Supabase:** Excelente alternativa si el proyecto requiere una base de datos gestionada.
