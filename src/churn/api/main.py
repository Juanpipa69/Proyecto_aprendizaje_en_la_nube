"""
API REST para servir predicciones del modelo de Churn.
Carga el modelo @champion desde MLflow y expone un endpoint de predicción.
"""

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="API de Predicción de Churn",
    description="Predice si un cliente de telecomunicaciones va a cancelar su servicio.",
    version="1.0.0",
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_URI = "models:/churn-random-forest@champion"
modelo = None


@app.on_event("startup")
def cargar_modelo():
    """Carga el modelo @champion al iniciar la API (no en cada petición)."""
    global modelo
    modelo = mlflow.sklearn.load_model(MODEL_URI)


class ClienteInput(BaseModel):
    """Datos de un cliente para predecir si va a cancelar el servicio."""

    call_failure: int = Field(..., ge=0, alias="Call  Failure")
    complains: int = Field(..., ge=0, le=1, alias="Complains")
    subscription_length: int = Field(..., gt=0, alias="Subscription  Length")
    charge_amount: int = Field(..., ge=0, alias="Charge  Amount")
    seconds_of_use: int = Field(..., ge=0, alias="Seconds of Use")
    frequency_of_use: int = Field(..., ge=0, alias="Frequency of use")
    frequency_of_sms: int = Field(..., ge=0, alias="Frequency of SMS")
    distinct_called_numbers: int = Field(..., ge=0, alias="Distinct Called Numbers")
    age_group: int = Field(..., ge=1, alias="Age Group")
    tariff_plan: int = Field(..., alias="Tariff Plan")
    status: int = Field(..., alias="Status")
    age: int = Field(..., gt=0, alias="Age")
    customer_value: float = Field(..., ge=0, alias="Customer Value")

    class Config:
        populate_by_name = True


class PrediccionOutput(BaseModel):
    churn_predicho: int
    probabilidad_churn: float
    mensaje: str


@app.get("/")
def raiz():
    return {"mensaje": "API de Predicción de Churn activa. Ver /docs para probarla."}


@app.get("/health")
def salud():
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.post("/predict", response_model=PrediccionOutput)
def predecir(cliente: ClienteInput):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado.")

    datos = cliente.dict(by_alias=True)
    df = pd.DataFrame([datos])

    # Mismo feature engineering aplicado en el entrenamiento
    df["Uso_Promedio_Mensual"] = df["Seconds of Use"] / df[
        "Subscription  Length"
    ].replace(0, 1)

    prediccion = int(modelo.predict(df)[0])
    probabilidad = float(modelo.predict_proba(df)[0][1])

    mensaje = (
        "Alto riesgo de cancelación"
        if prediccion == 1
        else "Bajo riesgo de cancelación"
    )

    return PrediccionOutput(
        churn_predicho=prediccion,
        probabilidad_churn=round(probabilidad, 4),
        mensaje=mensaje,
    )
