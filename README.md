# 🎮 CS:GO Player Action Prediction# 🎮 CS:GO Player Action Prediction# 🎮 ML-Trajectory_prediction



Predicción de acciones de jugadores en CS:GO usando LSTM (Long Short-Term Memory).



## 📊 DatasetPredicción de acciones de jugadores en CS:GO usando LSTM (Long Short-Term Memory).Predicción categórica de la próxima acción de jugadores en CS:GO usando LSTM.



**ESTA (Esports Trajectories and Actions)**

- 680 archivos `.json.xz` en `data/raw/lan/`

- Datos de partidas profesionales de CS:GO## 📊 Dataset![Python](https://img.shields.io/badge/Python-3.11-blue)

- Contiene trayectorias y acciones de jugadores

![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)

## 🎯 Objetivo

**ESTA (Esports Trajectories and Actions)**

Predecir la próxima acción de un jugador basándose en su historial de movimientos.

- 680 archivos `.json.xz` en `data/raw/lan/`---

**Clases:** `move`, `jump`, `duck`, `idle`, `dead`

- Datos de partidas profesionales de CS:GO

## 🏗️ Arquitectura

- Contiene trayectorias y acciones de jugadores## 📋 Descripción

**LSTM Classifier:** 2 capas, 128 unidades, dropout 0.3  

**Input:** Secuencias de 10 timesteps con 14 features  

**Output:** 5 clases  

**Parámetros:** 214,661## 🎯 ObjetivoModelo LSTM que predice la próxima acción de un jugador en CS:GO basándose en secuencias temporales.



## 📁 Estructura



```Predecir la próxima acción de un jugador basándose en su historial de movimientos.**Clases predichas:** 🏃 move | ⬆️ jump | ⬇️ duck | 🛑 idle | ☠️ dead

ML-Trajectory_prediction/

├── data/

│   ├── raw/lan/           # 680 archivos .json.xz

│   └── processed/         # Parquet procesados**Clases:** `move`, `jump`, `duck`, `idle`, `dead`---

├── src/

│   ├── data/              # 📂 Procesamiento de datos

│   │   ├── data_prep.py

│   │   ├── merge_batches.py## 🏗️ Arquitectura## ⚡ Quick Start

│   │   ├── data_der.py

│   │   └── validation.py

│   └── models/            # 📂 Modelo LSTM

│       ├── model.py**LSTM Classifier:** 2 capas, 128 unidades, dropout 0.3  ```powershell

│       ├── dataset.py

│       ├── train.py**Input:** Secuencias de 10 timesteps con 14 features  # 1. Clonar y configurar

│       └── evaluate.py

├── outputs/               # Modelos entrenados**Output:** 5 clases  git clone https://github.com/Hephaestus520/ML-Trajectory_prediction.git

└── main.py                # Menu interactivo

```**Parámetros:** 214,661cd ML-Trajectory_prediction



## 🚀 Instalación



```bash## 📁 Estructura# 2. Entorno virtual e instalación

python -m venv venv

venv\Scripts\activatepython -m venv venv

pip install -r requirements.txt

``````.\venv\Scripts\Activate.ps1



## 📖 Uso RápidoML-Trajectory_prediction/pip install -r requirements.txt



### Menu Interactivo├── data/



```bash│   ├── raw/lan/           # 680 archivos .json.xz# 3. Colocar datos en data/raw/lan/

python main.py

```│   └── processed/         # Parquet procesados



### Pipeline Manual├── src/# 4. Ejecutar menú interactivo



```bash│   ├── data/              # 📂 Procesamiento de datospython main.py

# 1. Procesar datos (30-45 min)

python src/data/data_prep.py│   │   ├── preprocessing.py```

python src/data/merge_batches.py

python src/data/data_der.py│   │   ├── merge_batches.py



# 2. Entrenar modelo (1-3 horas)│   │   ├── feature_engineering.py---

python src/models/train.py

│   │   └── validation.py

# 3. Evaluar

python src/models/evaluate.py│   └── models/            # 📂 Modelo LSTM## 📁 Estructura

```

│       ├── model.py

## 📂 Módulo: Procesamiento de Datos

│       ├── dataset.py```

### `data_prep.py`

Extrae datos de archivos `.json.xz` y genera batches Parquet.│       ├── train.pyML-Trajectory_prediction/



**Uso:**│       └── evaluate.py├── data/

```bash

python src/data/data_prep.py├── outputs/               # Modelos entrenados│   ├── raw/lan/              # Datos .json.xz (ESTA dataset)

```

└── main.py                # Menu interactivo│   └── processed/            # Datos .parquet procesados

**Salida:** `data/processed/batch_*.parquet`

```├── src/

### `merge_batches.py`

Combina todos los batches en un solo archivo.│   ├── model.py              # Arquitectura LSTM ✅



**Uso:**## 🚀 Instalación│   ├── dataset.py            # PyTorch Dataset ✅

```bash

python src/data/merge_batches.py│   ├── train.py              # Entrenamiento ✅

```

```bash│   ├── evaluate.py           # Evaluación ✅

**Salida:** `data_merged.parquet`

python -m venv venv│   ├── data_prep.py          # Procesamiento ✅

### `data_der.py`

Genera features derivadas y labels categóricas.venv\Scripts\activate│   ├── merge_batches.py      # Combinar ✅



**Features:** `dx`, `dy`, `dz`, `speed`, `acceleration`  pip install -r requirements.txt│   └── data_der.py           # Features ✅

**Labels:** `move`, `jump`, `duck`, `idle`, `dead`

```├── outputs/                  # Modelos entrenados

**Uso:**

```bash├── main.py                   # Menú principal ✅

python src/data/data_der.py

```## 📖 Uso Rápido├── GUIA_MODELO.md           # Guía completa del modelo



**Salida:** `data_merged_labeled.parquet`└── requirements.txt



### `validation.py`### Menu Interactivo```

Valida estructura y calidad de los datos.



## 📂 Módulo: Modelo LSTM

```bash---

### `model.py`

Define la arquitectura LSTM.python main.py



**Componentes:**```## 🚀 Uso Rápido

- 2 capas LSTM (128 hidden units)

- BatchNorm1d

- Dropout (0.3)

- 2 capas FC### Pipeline Manual```powershell



**Métodos:**# Activar entorno

- `forward()`: Propagación directa

- `predict()`: Predicción con probabilidades```bash.\venv\Scripts\Activate.ps1

- `predict_class()`: Predicción de clase

# 1. Procesar datos (30-45 min)

### `dataset.py`

PyTorch Dataset con secuencias temporales.python src/data/preprocessing.py# Menú interactivo (RECOMENDADO)



**Features:**python src/data/merge_batches.pypython main.py

- Ventanas deslizantes de 10 timesteps

- StandardScaler para normalizaciónpython src/data/feature_engineering.py

- LabelEncoder para clases

- Manejo de clases desbalanceadas# O scripts individuales:



**Métodos:**# 2. Entrenar modelo (1-3 horas)python src/data_prep.py      # Procesar datos

- `get_class_weights()`: Pesos para weighted loss

- `save_preprocessors()`: Guardar scalerspython src/models/train.pypython src/train.py          # Entrenar modelo



### `train.py`python src/evaluate.py       # Evaluar modelo

Pipeline completo de entrenamiento.

# 3. Evaluar```

**Configuración:**

```pythonpython src/models/evaluate.py

sequence_length = 10

batch_size = 64```Ver **[GUIA_MODELO.md](GUIA_MODELO.md)** para instrucciones completas.

hidden_size = 128

num_layers = 2

dropout = 0.3

learning_rate = 0.001## 📂 Módulo: Procesamiento de Datos---

num_epochs = 50

```



**Features:**### `preprocessing.py`## 🏗️ Modelo LSTM

- ✅ Weighted loss para clases desbalanceadas

- ✅ Train/Val/Test split (70/15/15)Extrae datos de archivos `.json.xz` y genera batches Parquet.

- ✅ Learning rate scheduling

- ✅ Checkpointing automático- **Input:** (batch, 10, 14) - 10 timesteps, 14 features

- ✅ Visualización de métricas

**Uso:**- **Arquitectura:** 2 capas LSTM + FC layers

**Uso:**

```bash```bash- **Output:** (batch, 5) - 5 clases

python src/models/train.py

```python src/data/preprocessing.py- **Parámetros:** 214,661 entrenables



**Salidas:**```

- `outputs/run_TIMESTAMP/best_model.pt`

- `outputs/run_TIMESTAMP/config.json`**Features:** posición (x,y,z), velocidad, HP, armadura, vista, derivadas

- `outputs/run_TIMESTAMP/training_history.png`

**Salida:** `data/processed/batch_*.parquet`

### `evaluate.py`

Evaluación del modelo con métricas detalladas.---



**Métricas:**### `merge_batches.py`

- Accuracy, Precision, Recall, F1-Score

- Confusion MatrixCombina todos los batches en un solo archivo.## 📊 Dataset

- Probability Distributions



**Uso:**

```bash**Uso:**- **Fuente:** ESTA (Esports Trajectories and Actions)

python src/models/evaluate.py

``````bash- **680 archivos** .json.xz procesados



**Salidas:**python src/data/merge_batches.py- **Secuencias:** Ventanas de 10 timesteps

- `confusion_matrix.png`

- `probability_distributions.png````- **Split:** 70% train / 15% val / 15% test

- `evaluation_results.json`



## 🛠️ Requisitos

**Salida:** `data_merged.parquet`---

- Python 3.11+

- PyTorch 2.9.0

- pandas, numpy, scikit-learn

- matplotlib, awpy### `feature_engineering.py`## 🎯 Resultados



## 🤝 CréditosGenera features derivadas y labels categóricas.



Universidad EAFIT - Machine Learning  - Accuracy: ~75-80%

Dataset: ESTA (Esports Trajectories and Actions)

**Features:** `dx`, `dy`, `dz`, `speed`, `acceleration`  - F1-Score: ~75-80%

**Labels:** `move`, `jump`, `duck`, `idle`, `dead`- Tiempo: 1-3 horas (CPU)



**Uso:**---

```bash

python src/data/feature_engineering.py## 📚 Documentación

```

- **[GUIA_USO.md](GUIA_USO.md)** - Procesamiento de datos

**Salida:** `data_merged_labeled.parquet`- **[GUIA_MODELO.md](GUIA_MODELO.md)** - Modelo y entrenamiento



### `validation.py`---

Valida estructura y calidad de los datos.

**Estado:** ✅ Funcional | **Última actualización:** Nov 2025

## 📂 Módulo: Modelo LSTM

### `model.py`
Define la arquitectura LSTM.

**Componentes:**
- 2 capas LSTM (128 hidden units)
- BatchNorm1d
- Dropout (0.3)
- 2 capas FC

**Métodos:**
- `forward()`: Propagación directa
- `predict()`: Predicción con probabilidades
- `predict_class()`: Predicción de clase

### `dataset.py`
PyTorch Dataset con secuencias temporales.

**Features:**
- Ventanas deslizantes de 10 timesteps
- StandardScaler para normalización
- LabelEncoder para clases
- Manejo de clases desbalanceadas

**Métodos:**
- `get_class_weights()`: Pesos para weighted loss
- `save_preprocessors()`: Guardar scalers

### `train.py`
Pipeline completo de entrenamiento.

**Configuración:**
```python
sequence_length = 10
batch_size = 64
hidden_size = 128
num_layers = 2
dropout = 0.3
learning_rate = 0.001
num_epochs = 50
```

**Features:**
- ✅ Weighted loss para clases desbalanceadas
- ✅ Train/Val/Test split (70/15/15)
- ✅ Learning rate scheduling
- ✅ Checkpointing automático
- ✅ Visualización de métricas

**Uso:**
```bash
python src/models/train.py
```

**Salidas:**
- `outputs/run_TIMESTAMP/best_model.pt`
- `outputs/run_TIMESTAMP/config.json`
- `outputs/run_TIMESTAMP/training_history.png`

### `evaluate.py`
Evaluación del modelo con métricas detalladas.

**Métricas:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Probability Distributions

**Uso:**
```bash
python src/models/evaluate.py
```

**Salidas:**
- `confusion_matrix.png`
- `probability_distributions.png`
- `evaluation_results.json`

## 🛠️ Requisitos

- Python 3.11+
- PyTorch 2.9.0
- pandas, numpy, scikit-learn
- matplotlib, awpy

## 🤝 Créditos

Universidad EAFIT - Machine Learning  
Dataset: ESTA (Esports Trajectories and Actions)
