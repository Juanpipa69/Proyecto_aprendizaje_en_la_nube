"""
Pipeline de entrenamiento automatizado para el modelo de Churn.
Convierte en un flow de Prefect el proceso que se validó manualmente
en notebooks/01_eda_y_baseline.ipynb.
"""

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlflow.client import MlflowClient
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@task(name="cargar_datos", log_prints=True)
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el dataset crudo desde data/raw/."""
    df = pd.read_csv(ruta)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


@task(name="validar_datos", log_prints=True)
def validar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Valida que el dataset tenga las columnas y tipos esperados antes de procesar."""
    columnas_esperadas = {
        "Call  Failure",
        "Complains",
        "Subscription  Length",
        "Charge  Amount",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Age Group",
        "Tariff Plan",
        "Status",
        "Age",
        "Customer Value",
        "Churn",
    }
    columnas_reales = set(df.columns)

    faltantes = columnas_esperadas - columnas_reales
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {faltantes}")

    if not set(df["Churn"].unique()).issubset({0, 1}):
        raise ValueError("La columna Churn debe contener solo valores 0 o 1")

    if df.shape[0] == 0:
        raise ValueError("El dataset está vacío")

    print(f"Validación OK: {df.shape[0]} filas, todas las columnas esperadas presentes")
    return df


@task(name="limpiar_datos", log_prints=True)
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos (hallazgo del EDA: ~9.5% del dataset)."""
    filas_antes = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados eliminados: {filas_antes - df.shape[0]}")
    return df


@task(name="preparar_features", log_prints=True)
def preparar_features(df: pd.DataFrame):
    """Split train/test, escalado y balanceo con SMOTE (solo en train)."""
    # Feature engineering: uso promedio por mes de suscripción
    # (un cliente con mucho uso pero suscripción corta se comporta distinto
    # a uno con el mismo uso pero muchos meses de antigüedad)
    df = df.copy()
    df["Uso_Promedio_Mensual"] = df["Seconds of Use"] / df[
        "Subscription  Length"
    ].replace(0, 1)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    print(f"Train balanceado: {X_train_bal.shape[0]} filas")
    return X_train_bal, X_test_scaled, y_train_bal, y_test


@task(name="entrenar_y_registrar_modelo", log_prints=True)
def entrenar_y_registrar_modelo(X_train_bal, X_test_scaled, y_train_bal, y_test):
    """Entrena Random Forest, lo trackea en MLflow y lo promueve a @champion."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="random_forest_pipeline"):
        modelo = RandomForestClassifier(random_state=42, n_estimators=100)
        modelo.fit(X_train_bal, y_train_bal)

        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("modelo", "random_forest")
        mlflow.log_param("balanceo", "SMOTE")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            modelo, name="modelo", registered_model_name="churn-random-forest"
        )

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"F1-Score: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")

    client = MlflowClient()
    latest_version = client.get_latest_versions("churn-random-forest")[0].version
    client.set_registered_model_alias(
        name="churn-random-forest", alias="champion", version=latest_version
    )
    print(f"Modelo versión {latest_version} promovido a @champion")

    return {"f1_score": f1, "roc_auc": roc_auc, "version": latest_version}


@flow(name="pipeline-entrenamiento-churn", log_prints=True)
def pipeline_entrenamiento(ruta_datos: str = "data/raw/Customer Churn.csv"):
    """Flow principal: orquesta todo el pipeline de entrenamiento."""
    df = cargar_datos(ruta_datos)
    df = validar_datos(df)
    df_limpio = limpiar_datos(df)
    X_train_bal, X_test_scaled, y_train_bal, y_test = preparar_features(df_limpio)
    resultado = entrenar_y_registrar_modelo(
        X_train_bal, X_test_scaled, y_train_bal, y_test
    )
    return resultado


if __name__ == "__main__":
    # Ejecución directa (para pruebas)
    # pipeline_entrenamiento()

    # Deployment con scheduling automático: corre todos los días a las 2:00 AM
    pipeline_entrenamiento.serve(
        name="entrenamiento-diario-churn",
        cron="0 2 * * *",
    )
