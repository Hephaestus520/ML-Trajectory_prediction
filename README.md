# ML-Trajectory_prediction

├── data/                     # Dataset (local, no se sube a GitHub)
│   ├── raw/                  # Datos originales descargados
│   └── processed/            # Datos listos para entrenar
│
├── notebooks/                # Exploración y prototipos (EDA, modelos)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_baseline.ipynb
│
├── src/                      # Código fuente del proyecto
│   ├── __init__.py
│   ├── data_prep.py          # Limpieza y generación de features
│   ├── dataset.py            # Clase Dataset de PyTorch
│   ├── model.py              # Definición del modelo (LSTM, etc.)
│   ├── train.py              # Bucle de entrenamiento
│   ├── evaluate.py           # Métricas y validación
│   └── utils.py              # Funciones auxiliares
│
├── outputs/                  # Modelos entrenados, logs, resultados
│
├── requirements.txt
├── README.md
└── .gitignore
