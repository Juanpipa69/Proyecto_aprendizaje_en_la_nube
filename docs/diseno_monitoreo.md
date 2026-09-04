# Diseño de Monitoreo — Modelo de Predicción de Churn

## 1. Objetivo

Detectar cuándo los datos que recibe el modelo en producción se alejan de los
datos con los que fue entrenado (`data/raw/Customer Churn.csv`), para saber
cuándo el modelo puede estar perdiendo precisión sin que nadie se dé cuenta.

## 2. Qué se mide

| Tipo de monitoreo | Qué evalúa | Herramienta |
|---|---|---|
| **Data Drift** | Si la distribución de las variables predictoras (`Complains`, `Customer Value`, `Frequency of use`, etc.) cambia respecto a los datos de entrenamiento | Evidently (`DataDriftPreset`) |
| **Prediction Drift** | Si la proporción de predicciones "va a cancelar" vs. "se queda" cambia de forma inusual en el tiempo | Evidently, comparando distribución de `churn_predicho` |
| **Model Performance** (cuando haya datos reales de churn confirmado) | Si el F1-Score y ROC-AUC del modelo en producción se degradan respecto a los valores de validación (F1 ≈ 0.87, ROC-AUC ≈ 0.98) | scikit-learn, comparado contra el baseline registrado en MLflow |

## 3. Con qué frecuencia

- **Data Drift:** cada vez que se acumulen 100 predicciones nuevas, o semanalmente (lo que ocurra primero).
- **Model Performance:** mensualmente, una vez se tenga confirmación real de qué clientes cancelaron ese mes.

## 4. Umbrales de alerta

| Métrica | Umbral | Acción |
|---|---|---|
| % de variables con drift detectado (Evidently) | Más del 30% de las variables muestran drift significativo | Alerta amarilla: revisar manualmente el reporte |
| Drift en `Complains` o `Customer Value` (las variables más predictivas, según el EDA) | Cualquier drift detectado en estas dos variables específicas | Alerta roja: revisión prioritaria, aunque el % global no supere el 30% |
| F1-Score en producción | Cae por debajo de 0.75 (frente al 0.87 de validación) | Alerta roja: considerar reentrenamiento |
| ROC-AUC en producción | Cae por debajo de 0.90 (frente al 0.98 de validación) | Alerta roja: considerar reentrenamiento |

## 5. Qué se hace si se dispara una alerta

1. **Alerta amarilla (drift moderado):** el equipo revisa el reporte HTML de Evidently para entender qué variables cambiaron y por qué (¿cambió el negocio? ¿un error en la fuente de datos?).
2. **Alerta roja (drift en variable crítica, o caída de performance):**
   - Pausar temporalmente las decisiones automáticas basadas en el modelo (usarlo solo como referencia, no como decisión final).
   - Ejecutar el pipeline de reentrenamiento (`src/churn/flows/training_pipeline.py`) con los datos más recientes disponibles.
   - Comparar el modelo reentrenado contra el `@champion` actual en MLflow; si el nuevo modelo mejora las métricas, promoverlo a `@champion`.
3. **Documentar** cada incidente de drift (fecha, variables afectadas, acción tomada) para llevar un historial.

## 6. Estado actual

- ✅ Reporte de Data Drift implementado y probado (`Monitoring/drift_monitoring.py`, resultado en `reports/data_drift_report.html`).
- ⏳ Prediction Drift y Model Performance: diseñados en este documento, implementación pendiente para una fase futura del proyecto (no obligatoria en el alcance actual del curso).