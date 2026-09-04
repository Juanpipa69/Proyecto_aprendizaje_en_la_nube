FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# Entrenar el modelo DENTRO del contenedor, para que las rutas de MLflow
# queden consistentes con el propio contenedor (evita rutas absolutas de Windows)
RUN python -c "from src.churn.flows.training_pipeline import pipeline_entrenamiento; pipeline_entrenamiento()"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]