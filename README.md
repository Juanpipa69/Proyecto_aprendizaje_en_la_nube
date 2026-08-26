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
