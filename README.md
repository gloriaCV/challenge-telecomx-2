# 📊 Proyecto Telecom X - Parte 2
<img width="2816" height="811" alt="portada" src="https://github.com/user-attachments/assets/6368db64-70fd-4ffc-a11d-2bcceae87da2" />

## 🚀 Propósito del Análisis
El objetivo principal de este proyecto es **predecir el churn (abandono/cancelación) de clientes** en una empresa de telecomunicaciones, utilizando técnicas de análisis de datos y machine learning.  
Se busca identificar **las variables más relevantes** que influyen en la decisión de los clientes de cancelar el servicio, y así proponer **estrategias de retención**.

---

## 📂 Estructura del Proyecto

```
TelecomX_Parte2/
├── telecomx_2.ipynb           # Cuaderno principal del análisis
├── datos_tratados.csv         # Dataset preprocesado utilizado en el notebook
├── imagenes/                  # Carpeta con visualizaciones generadas
│ ├── barplots_variables_categoricas.png
│ ├── grafico_curva_precall_rf.png
│ ├── grafico_curva_precall_rl.png
│ ├── grafico_curva_roc_rf.png
│ ├── grafico_curva_roc_rl.png
│ ├── grafico_donut_abandono.png
│ ├── histogramas_variables_numericas.png
│ ├── mapa_calor_correlacion_1.png
│ ├── mapa_calor_correlacion_2.png
│ ├── matriz_confusion_NearMiss_rf.png
│ ├── matriz_confusion_Original_rf.png
│ ├── matriz_confusion_Original_rl.png
│ ├── matriz_confusion_RL_NearMiss_rl.png
│ ├── matriz_confusion_SMOTE_rf.png
│ ├── matriz_confusion_SMOTE_rl.png
│ ├── pairplot_variables_numericas.png
│ └── portada.png
├── LICENSE
└── README.md                  # Documentación del proyecto
```

---

## 🛠️ Preparación de los Datos

El dataset original contenía **7043 registros y 32 variables**.  
Se realizó un proceso de limpieza y transformación de variables que incluyó:

- **Clasificación de variables**
  - Numéricas: `Antiguedad_Meses`, `Cargos_Mensuales`
  - Categóricas transformadas en dummies (booleanas): variables de contrato, tipo de internet, métodos de pago, servicios adicionales, etc.
  - Binarias: `Adulto_Mayor`, `Pareja`, `Dependientes`, `Servicio_Telefonico`, `Facturacion_Sin_Papel`

- **Codificación**
  - Se aplicó *One-Hot Encoding* (via `get_dummies`) para las variables categóricas, generando columnas binarias (`Yes/No`).

- **Normalización**
  - Las variables numéricas no se escalaron en esta etapa, dado que los modelos elegidos no lo requieren estrictamente (Random Forest es invariante al escalado, y Regresión Logística puede trabajar directamente con variables en la escala original).

- **Separación de datos**
  - Los datos se dividieron en conjuntos de **entrenamiento (70%) y prueba (30%)** para validar los modelos.
---

## 📊 Exploración

Durante el **Análisis Exploratorio de Datos (EDA)** se aplicaron diversos gráficos:

- **Donut chart**: proporción de clientes que abandonan vs. los que permanecen.
    <img width="500" height="500" alt="grafico_donut_abandono" src="https://github.com/user-attachments/assets/733c8059-8629-46da-89b8-85f1cb1169f9" />

- **Pairplot**: relación entre variables numéricas.
   <img width="890" height="750" alt="pairplot_variables_numericas" src="https://github.com/user-attachments/assets/01058920-794b-48b5-94d2-d0ccc75b068a" />

- **Histogramas**: distribución de cargos mensuales y antigüedad.
   <img width="1800" height="600" alt="histogramas_variables_numericas" src="https://github.com/user-attachments/assets/7b31599b-f6bc-43de-a32f-0dabba73ddc5" />

- **Barras agrupadas**: churn según tipo de contrato y servicio de internet.
   <img width="1800" height="600" alt="barplots_variables_categoricas" src="https://github.com/user-attachments/assets/738e77e8-fd71-466e-8660-6caa8f77e69c" />

- **Matriz de correlación**: detección de multicolinealidad.  
   <img width="2000" height="1800" alt="mapa_calor_correlacion_2" src="https://github.com/user-attachments/assets/26c5ec91-e75e-4700-9109-306bfab074ee" />

### Hallazgos principales:
- Los clientes con **menor antigüedad en la compañía** presentan la mayor tasa de abandono, especialmente si tienen **cargos mensuales altos**.  
- Los clientes con **contratos mes a mes** abandonan más, mientras que los de **contratos a dos años** son los más estables.  
- Los clientes con **servicio de internet por fibra óptica** muestran un nivel de churn significativamente mayor.  
---

## 🤖 Modelado Predictivo

Se implementaron dos enfoques de clasificación:

1. **Regresión Logística**  
   - Modelo base, elegido por su interpretabilidad y capacidad para manejar problemas de clasificación binaria.  
   - Permite analizar la influencia individual de cada variable sobre la probabilidad de churn.

2. **Random Forest**  
   - Modelo basado en *ensembles*, elegido por su robustez frente a variables correlacionadas y su capacidad para capturar relaciones no lineales.  
   - Permite extraer métricas de **importancia de variables**.
     
Se aplicaron técnicas de **balanceo de clases** como **SMOTE** y **NearMiss**, para enfrentar el desbalance entre clientes que abandonan y los que permanecen.


---
## **Diagrama de Flujo del Modelo Predictivo**

```
[Preparar Datos] 
       ↓
[Entrenar Modelo Inicial] → [Identificar Características Importantes]
       ↓
[Probar con 10-13 Características] → [Seleccionar Mejor Cantidad]
       ↓
[Optimizar Hiperparámetros] → [Original | SMOTE | NearMiss]
       ↓
[Entrenar Modelo Final] → [Aplicar NearMiss solo a Entrenamiento]
       ↓
[Evaluar en Prueba] → [Diagnosticar Ajuste]
```
### ¿Qué Hace Cada Parte del Flujo de Trabajo?

#### 1. Selección de Características

🔍 **Identificación de predictores clave**  
- Encuentra las variables que mejor predicen el abandono de clientes  
- Ordena las características por su importancia estadística
  
📊 **Optimización de cantidad**  
- Prueba con diferentes números de características  
- Selecciona la cantidad óptima que maximiza la detección de abandonos  

#### 2. Preprocesamiento de Datos

📏 **Escalado** 
- Si el modelo es sensible a escalas requiere normalización  

⚖️ **Tres estrategias comparadas**  
| Estrategia  | Método                     | Impacto                                  |
|-------------|----------------------------|------------------------------------------|
| **Original**| Sin modificación           | Trabaja con datos desbalanceados reales  |
| **SMOTE**   | Sobremuestreo              | Genera nuevos casos sintéticos de abandono |
| **NearMiss**| Submuestreo                | Reduce casos de clientes fieles          |

#### 3. Optimización del Modelo
⚙️ **Búsqueda de mejor configuración**  
- Prueba automáticamente combinaciones de parámetros  
- Ajusta elementos clave:
    - **Regresión Logística**:  
       - C (regularización)
       - penalty (L1/L2) 
       - liblinear, saga

    - **Random Forest**:  
       - Número de árboles 
       - Profundidad máxima  
       - Muestras por hoja

  
🎯 **Foco en detección de abandonos**  
- Usa Recall como métrica principal de optimización  

#### 4. Evaluación Final
📈 **Medición de desempeño**  
- Recall: Capacidad de detectar abandonos reales  
- Comparación entrenamiento vs. prueba

🔍 **Diagnóstico de problemas**  
| Situación        | Indicador                           | Solución potencial               |
|------------------|-------------------------------------|----------------------------------|
| **Underfitting** | Recall bajo en ambos conjuntos      | Modelo más complejo o más datos |
| **Overfitting**  | Recall alto en entrenamiento, bajo en prueba | Regularización o simplificación |

### Objetivo Principal 🎯

**Maximizar la detección de clientes en riesgo de abandono**  
- Prioriza identificar la mayor cantidad posible de abandonos reales  
- Acepta cierto margen de error en falsas alarmas  


---
## ⚙️ Tecnologías y Librerías Utilizadas

- **Python 3.11.13**
- **Pandas 2.2.2**
- **NumPy 2.0.2**
- **Matplotlib 3.10.0**
- **Seaborn 0.13.2**
- **Scikit-learn 1.6.1**
- **Imbalanced-learn 0.13.0**
- **Google Colab** (entorno de ejecución)

---
## 🚀 Cómo Usar Este Repositorio

Para visualizar y ejecutar el análisis, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/gloriaCV/challenge-telecomx-2.git
cd challenge-telecomx-2
```

### 2. Abrir el archivo principal
Puedes abrir el archivo `telecomx_2.ipynb` de dos formas:

#### Opción A: En Google Colab (recomendado)
- Abre [Google Colab](https://colab.research.google.com/).
- Selecciona **Archivo > Abrir cuaderno > GitHub**.
- Pega la URL del repositorio:  
```
https://github.com/gloriaCV/challenge-telecomx-2
```
- Abre `telecomx_2.ipynb` y ejecuta las celdas.

#### Opción B: En Jupyter Notebook
- Asegúrate de tener **Jupyter** instalado:
```bash
pip install notebook
```
- Inicia Jupyter:
```
jupyter notebook
```
- Abre `telecomx_2.ipynb` desde el navegador

---

## 📌 Resultados Principales

- **Modelos entrenados**: Regresión Logística y Random Forest.  
- **Mejor desempeño**: Random Forest, gracias a su capacidad para capturar relaciones no lineales.  
- **Variables más influyentes**:
  1. Antigüedad del cliente.  
  2. Tipo de contrato.  
  3. Facturación mensual.  
  4. Servicios adicionales.

 ---

## 📄 Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

---


## 🚀 Autor

**Gloria Cisternas**  
[GitHub](https://github.com/gloriaCV) 
