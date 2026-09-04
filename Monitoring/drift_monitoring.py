import pandas as pd
import numpy as np
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Customer Churn.csv"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR DATASET
# ============================================================

print("=" * 60)
print("MONITOREO DE DATA DRIFT - CUSTOMER CHURN")
print("=" * 60)

print("\n1. Cargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset cargado correctamente.")
print(f"Número de registros: {len(df)}")
print(f"Número de columnas: {len(df.columns)}")


# ============================================================
# 3. LIMPIAR NOMBRES DE COLUMNAS
# ============================================================

# El dataset contiene algunos nombres con espacios dobles.
# Los normalizamos para trabajar de forma más segura.

df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

print("\nColumnas normalizadas:")

for column in df.columns:
    print(f" - {column}")


# ============================================================
# 4. DEFINIR VARIABLE OBJETIVO
# ============================================================

TARGET = "Churn"

if TARGET not in df.columns:
    raise ValueError(
        f"No se encontró la variable objetivo '{TARGET}'"
    )


# ============================================================
# 5. SEPARAR VARIABLES PREDICTORAS
# ============================================================

features = [
    column for column in df.columns
    if column != TARGET
]

df_features = df[features].copy()


# ============================================================
# 6. CREAR DATASET DE REFERENCIA Y DATOS NUEVOS
# ============================================================

print("\n2. Creando conjuntos de referencia y datos nuevos...")

np.random.seed(42)

# 80% de los datos serán nuestra referencia
reference_data = df_features.sample(
    frac=0.80,
    random_state=42
)

# 20% serán los datos nuevos
current_data = df_features.drop(
    reference_data.index
).copy()


reference_data = reference_data.reset_index(drop=True)
current_data = current_data.reset_index(drop=True)


print(f"Datos de referencia: {reference_data.shape}")
print(f"Datos nuevos: {current_data.shape}")


# ============================================================
# 7. SIMULAR DATA DRIFT
# ============================================================

print("\n3. Simulando cambios en los datos nuevos...")

# IMPORTANTE:
# Esto se hace únicamente para demostrar que el sistema
# de monitoreo detecta cambios en la distribución.
#
# En producción, estos serían datos reales que llegan
# posteriormente.

# Cambio en Charge_Amount
if "Charge_Amount" in current_data.columns:

    current_data["Charge_Amount"] = (
        current_data["Charge_Amount"] + 1
    )

    print(" - Cambio simulado en Charge_Amount")


# Cambio en Customer_Value
if "Customer_Value" in current_data.columns:

    current_data["Customer_Value"] = (
        current_data["Customer_Value"] * 1.20
    )

    print(" - Cambio simulado en Customer_Value")


# Cambio en Frequency_of_use
if "Frequency_of_use" in current_data.columns:

    current_data["Frequency_of_use"] = (
        current_data["Frequency_of_use"] * 1.15
    )

    print(" - Cambio simulado en Frequency_of_use")


# ============================================================
# 8. GENERAR REPORTE DE EVIDENTLY
# ============================================================

print("\n4. Generando reporte de Data Drift...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ============================================================
# 9. GUARDAR REPORTE HTML
# ============================================================

report_path = REPORT_DIR / "data_drift_report.html"

result.save_html(str(report_path))


# ============================================================
# 10. RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("MONITOREO FINALIZADO")
print("=" * 60)

print(f"\nReporte generado correctamente en:")

print(report_path)

print("\nAbre el archivo HTML en tu navegador.")