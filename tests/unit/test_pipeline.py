import pandas as pd
import pytest

from src.churn.flows.training_pipeline import limpiar_datos, preparar_features, validar_datos

COLUMNAS = [
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
]


def _df_valido(n_filas: int = 20) -> pd.DataFrame:
    data = {col: [i % 5 + 1 for i in range(n_filas)] for col in COLUMNAS}
    data["Churn"] = [i % 2 for i in range(n_filas)]
    return pd.DataFrame(data)


def test_validar_datos_ok():
    df = _df_valido()
    validar_datos.fn(df)  # no debe lanzar excepcion


def test_validar_datos_columna_faltante():
    df = _df_valido().drop(columns=["Age"])
    with pytest.raises(ValueError):
        validar_datos.fn(df)


def test_validar_datos_target_invalido():
    df = _df_valido()
    df["Churn"] = [0, 2] * (len(df) // 2)
    with pytest.raises(ValueError):
        validar_datos.fn(df)


def test_limpiar_datos_elimina_duplicados():
    df = _df_valido(n_filas=10)
    df_con_duplicados = pd.concat([df, df.iloc[:3]], ignore_index=True)
    df_limpio = limpiar_datos.fn(df_con_duplicados)
    assert df_limpio.shape[0] == 10


def test_preparar_features_agrega_columna_derivada():
    df = _df_valido(n_filas=20)
    X_train, X_test, y_train, y_test = preparar_features.fn(df)
    assert "Uso_Promedio_Mensual" in X_train.columns
    assert len(X_train) + len(X_test) == 20