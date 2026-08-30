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
* **Selección de Modelos:** Entrenamiento y evaluación de múltiples algoritmos de clasificación, incluyendo Regresión Logística, Random Forest y XGBoost.
* **Optimización de Hiperparámetros:** Ajuste fino de parámetros mediante `GridSearchCV` priorizando métricas como Recall y ROC-AUC.

 **Comparativa de Modelos**

>  **Pendiente.** Los modelos aún no se han entrenado (Fase 3 del proyecto, en curso). Esta tabla se completará con resultados reales de Regresión Logística, Random Forest y XGBoost una vez finalizado el entrenamiento y la optimización de hiperparámetros.

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
