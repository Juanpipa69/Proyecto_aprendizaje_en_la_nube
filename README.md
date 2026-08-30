# Predicción de Churn (Cancelación) de Clientes

**Predicción de Churn de Clientes** es un pipeline de machine learning end-to-end diseñado para identificar clientes con alto riesgo de cancelar su servicio, permitiendo implementar estrategias de retención basadas en datos.

**Descripción del Dataset**

El proyecto utiliza el archivo `Customer Churn.csv` (Iranian Churn Dataset, UCI), con datos de comportamiento de uso de un operador de telecomunicaciones para predecir el abandono de clientes. Son 3,150 registros, 13 variables predictoras, todas numéricas desde el origen (sin texto que codificar) y sin nulos.

| Categoría de Características | Descripción | Atributos Clave |
| :--- | :--- | :--- 
| **Uso del servicio** | Volumen de llamadas, SMS y datos | `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers` |
| **Calidad de servicio** | Fallos e incidencias | `Call Failure`, `Complains` |
| **Plan y cuenta** | Antigüedad, tarifa y estado | `Subscription Length`, `Charge Amount`, `Tariff Plan`, `Status` |
| **Demografía y valor** | Perfil del cliente | `Age`, `Age Group`, `Customer Value` |
| **Variable Objetivo** | Indicador de abandono del cliente | `Churn` (0 = se queda, 1 = abandona; ya viene codificada) |

**Flujo de Trabajo de Machine Learning**

* **Limpieza y Preprocesamiento de Datos:** Revisión de duplicados (300 filas duplicadas detectadas, ~9.5% del dataset) y escalado con `StandardScaler` sobre variables numéricas. No se requiere One-Hot Encoding porque el dataset no tiene variables categóricas de texto.
* **Análisis Exploratorio de Datos (EDA):** Identificación de indicadores clave de abandono, como la duración de la suscripción (`Subscription Length`), la frecuencia de uso y las quejas del cliente (`Complains`).
* **Manejo del Desbalanceo de Clases:** Aplicación de SMOTE (Técnica de Sobremuestreo de Minorías Sintéticas) para equilibrar la distribución de la variable objetivo.
* **Selección de Modelos:** Entrenamiento y evaluación de Regresión Logística y Random Forest (XGBoost queda como mejora futura opcional).
* **Optimización de Hiperparámetros:** Pendiente — se usaron los hiperparámetros por defecto de scikit-learn; ajuste fino con `GridSearchCV` queda como mejora futura.

 **Comparativa de Modelos**

| Modelo | F1-Score | ROC-AUC | Notas |
| :--- | :--- | :--- | :--- |
| Baseline (Dummy Classifier) | 0.00 | 0.50 | Punto de referencia mínimo |
| Regresión Logística | 0.63 | 0.92 | Con SMOTE sobre el train |
| **Random Forest** | **0.85–0.87** | **0.98** | Modelo ganador, registrado como `@champion` en MLflow |

> Validado con 5-fold cross-validation: F1 promedio 0.972 (±0.006), ROC-AUC promedio 0.994 (±0.003) sobre el set de entrenamiento balanceado. XGBoost y GridSearchCV quedan como mejora futura opcional.

**Estructura del Repositorio**

* `data/raw/` — Dataset original `Customer Churn.csv`.
* `notebooks/` — EDA, experimentos y baseline (`01_eda_y_baseline.ipynb`).
* `src/churn/flows/` — Pipeline de entrenamiento automatizado (`training_pipeline.py`, Prefect).
* `src/churn/api/`, `features/`, `models/`, `monitoring/` — Módulos para las siguientes fases (Deployment y Monitoreo).
* `mlflow.db` — Base de datos local de experimentos MLflow (no se sube a Git).
* `tests/`, `configs/`, `docs/`, `.github/workflows/` — Estructura recomendada del curso, lista para las fases 4-6.

**Guía de Instalación y Uso**

* **Clonar el repositorio:** `git clone https://github.com/tu-usuario/Proyecto_aprendizaje_en_la_nube.git`
* **Navegar a la carpeta:** `cd Proyecto_aprendizaje_en_la_nube`
* **Instalar dependencias:** `pip install -r requirements.txt`
* **Ejecutar el pipeline de entrenamiento:** `python src/churn/flows/training_pipeline.py`
* **Ver los experimentos en MLflow:** `mlflow ui --backend-store-uri sqlite:///mlflow.db`

# Fase 1: Planificación y Setup del Proyecto

## 1.1 Selección del Proyecto
* **Problema Elegido:** Predicción de Abandono (Churn) de Clientes en Telecomunicaciones. Es un problema de clasificación binaria.
* **Problema de Negocio (Hipotético):** La empresa de telecomunicaciones está perdiendo ingresos debido a una alta tasa de cancelación de contratos. Retener a un cliente existente es mucho más barato que adquirir uno nuevo. Este modelo permitirá al equipo de retención identificar clientes de alto riesgo y ofrecerles promociones preventivas personalizadas, aumentando el LTV (Life Time Value) y reduciendo la pérdida de ingresos.
* **Métricas de Éxito:** 
  * **Técnica:** Maximizar el `F1-Score` y el `ROC-AUC` (dado que el dataset suele estar desbalanceado). Un Recall alto es crucial para no dejar escapar falsos negativos (clientes que se van pero el modelo no detectó).
  * **Negocio:** Reducir la tasa de churn mensual en un 15% en el segmento de alto riesgo.
* **Alcance del Proyecto:**
  * **MVP (Producto Mínimo Viable):** Modelo baseline de Machine Learning (Regresión Logística/Random Forest) entrenado con datos limpios y empaquetado en una estructura de MLOps básica, expuesto posiblemente mediante un script de predicción simple.

### Estado del Proyecto (según las 6 fases del curso)

| Fase | Contenido | Responsable | Estado |
| :--- | :--- | :--- | :--- |
| **Fase 1:** Planificación, Setup y EDA | Selección de dataset, entorno, análisis exploratorio, baseline | Mariana / Ceneida | ✅ Completa |
| **Fase 2:** Experiment Tracking | MLflow, comparación de modelos, model registry (`@champion`) | Ceneida | ✅ Completa |
| **Fase 3:** Pipeline de Entrenamiento | Automatización con Prefect, validación de datos, feature engineering, scheduling | Ceneida | ✅ Completa |
| **Fase 4:** Deployment | Dockerfile, API REST con FastAPI | Mariana / Juan | ⏳ Pendiente |
| **Fase 5:** Monitoreo | Reporte de drift, diseño de monitoreo | Por asignar | ⏳ Pendiente |
| **Fase 6:** Testing y Best Practices | Unit tests, linter, pre-commit | Por asignar | ⏳ Pendiente |
