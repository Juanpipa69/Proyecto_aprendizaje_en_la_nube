# Predicción de Churn (Cancelación) de Clientes

**Predicción de Churn de Clientes** es un pipeline de machine learning end-to-end diseñado para identificar clientes con alto riesgo de cancelar su servicio, permitiendo implementar estrategias de retención basadas en datos.

**Descripción del Dataset**

El proyecto utiliza el archivo `Customer Churn.csv`, el cual registra datos demográficos, metadatos de cuenta y patrones de uso de servicios para predecir la pérdida de clientes.

| Categoría de Características | Descripción | Atributos Clave |
| :--- | :--- | :--- |
| **Demografía** | Detalles personales del cliente | `Gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Metadatos de Cuenta** | Historial contractual y financiero | `Tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges` |
| **Servicios Suscritos** | Productos activos y agregados | `InternetService`, `OnlineSecurity`, `TechSupport` |
| **Variable Objetivo** | Indicador de abandono del cliente | `Churn` (Sí/No mapeado a valores binarios) |

**Flujo de Trabajo de Machine Learning**

* **Limpieza y Preprocesamiento de Datos:** Manejo de valores nulos o faltantes, escalado con `StandardScaler` y aplicación de One-Hot Encoding para variables categóricas.
* **Análisis Exploratorio de Datos (EDA):** Identificación de indicadores clave de abandono, como la duración del contrato, el tipo de pago y la antigüedad (`tenure`).
* **Manejo del Desbalanceo de Clases:** Aplicación de SMOTE (Técnica de Sobremuestreo de Minorías Sintéticas) para equilibrar la distribución de la variable objetivo.
* **Selección de Modelos:** Entrenamiento y evaluación de múltiples algoritmos de clasificación, incluyendo Regresión Logística, Random Forest y XGBoost.
* **Optimización de Hiperparámetros:** Ajuste fino de parámetros mediante `GridSearchCV` priorizando métricas como Recall y ROC-AUC.

**Comparativa de Modelos**

| Modelo | Exactitud (Accuracy) | Precisión | Exhaustividad (Recall) | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística** | 0.79 | 0.64 | 0.55 | 0.59 | 0.82 |
| **Random Forest** | 0.81 | 0.67 | 0.59 | 0.63 | 0.85 |
| **XGBoost (Optimizado)** | **0.84** | **0.73** | **0.71** | **0.72** | **0.89** |

**Estructura del Repositorio**

* `data/` — Dataset original `Customer Churn.csv` y datos procesados.
* `notebooks/` — Notebooks de Jupyter con el EDA y experimentos con modelos.
* `src/` — Scripts modulares de Python para preprocesamiento, entrenamiento y evaluación.
* `models/` — Artefactos de modelos entrenados y guardados (`.pkl` / `.joblib`).

**Guía de Instalación y Uso**

* **Clonar el repositorio:** `git clone https://github.com/tu-usuario/Proyecto_aprendizaje_en_la_nube.git`
* **Navegar a la carpeta:** `cd Proyecto_aprendizaje_en_la_nube`
* **Instalar dependencias:** `pip install -r requirements.txt`
* **Ejecutar el entrenamiento:** `python src/train.py`
* **Generar predicciones:** `python src/predict.py --input data/Customer\ Churn.csv`

# Fase 1: Planificación y Setup del Proyecto

## 1.1 Selección del Proyecto
* **Problema Elegido:** Predicción de Abandono (Churn) de Clientes en Telecomunicaciones. Es un problema de clasificación binaria.
* **Problema de Negocio (Hipotético):** La empresa de telecomunicaciones está perdiendo ingresos debido a una alta tasa de cancelación de contratos. Retener a un cliente existente es mucho más barato que adquirir uno nuevo. Este modelo permitirá al equipo de retención identificar clientes de alto riesgo y ofrecerles promociones preventivas personalizadas, aumentando el LTV (Life Time Value) y reduciendo la pérdida de ingresos.
* **Métricas de Éxito:** 
  * **Técnica:** Maximizar el `F1-Score` y el `ROC-AUC` (dado que el dataset suele estar desbalanceado). Un Recall alto es crucial para no dejar escapar falsos negativos (clientes que se van pero el modelo no detectó).
  * **Negocio:** Reducir la tasa de churn mensual en un 15% en el segmento de alto riesgo.
* **Alcance del Proyecto:**
  * **MVP (Producto Mínimo Viable):** Modelo baseline de Machine Learning (Regresión Logística/Random Forest) entrenado con datos limpios y empaquetado en una estructura de MLOps básica, expuesto posiblemente mediante un script de predicción simple.

### Timeline del Proyecto (MVP)

| Tarea | Duración Estimada | Responsable | Estado |
| :--- | :--- | :--- | :--- |
| **Fase 1:** Setup, Planificación y EDA | Semana 1 | Mariana | En progreso |
| **Fase 2:** Preprocesamiento e Ing. Características | Semana 2 | Juan | Pendiente |
| **Fase 3:** Entrenamiento y Selección de Modelos | Semana 3 | Ceneida | Pendiente |
| **Fase 4:** Refactorización MLOps y Despliegue (API) | Semana 4 | Todos | Pendiente |
