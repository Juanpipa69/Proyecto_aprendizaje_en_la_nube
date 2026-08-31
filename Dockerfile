FROM python:3.11-slim

WORKDIR /app

# Copiar solo requirements primero (aprovecha el cache de Docker si no cambian)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY src/ src/
COPY data/ data/
COPY mlflow.db .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]