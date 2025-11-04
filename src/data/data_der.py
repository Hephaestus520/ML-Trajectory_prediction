"""
data_prep.py
-------------
Prepara el dataset ESTA (Esports Trajectories and Actions) para el modelo
de predicción cualitativa de trayectorias en CS:GO.

Pipeline:
1. Carga los datos combinados (data_merged.parquet)
2. Calcula variables derivadas: velocidad, cambios en altura (dz)
3. Genera etiquetas cualitativas: move, jump, duck, idle
4. Guarda dataset limpio y etiquetado
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# 🔧 Parámetros globales
# -------------------------------------------------------
DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "data_merged.parquet"
OUTPUT_FILE = DATA_DIR / "data_merged_labeled.parquet"

# Thresholds empíricos (ajustables)
VEL_THRESH = 5        # velocidad mínima para considerar movimiento
Z_JUMP_THRESH = 20    # diferencia en z para detectar salto
Z_DUCK_THRESH = -10   # diferencia en z para detectar agacharse


# -------------------------------------------------------
# 🧩 Funciones de preparación
# -------------------------------------------------------

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables derivadas como velocidad y cambios verticales.
    """
    df = df.sort_values(["map", "round", "player", "tick"])
    
    # Calcular diferencias por jugador
    df["dx"] = df.groupby(["map", "round", "player"])["x"].diff().fillna(0)
    df["dy"] = df.groupby(["map", "round", "player"])["y"].diff().fillna(0)
    df["dz"] = df.groupby(["map", "round", "player"])["z"].diff().fillna(0)
    
    # Magnitud de la velocidad (si no está directamente disponible)
    if "velocity" not in df.columns:
        df["velocity"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)
    
    # Magnitud horizontal (más relevante para movimiento)
    df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    return df


def derive_action_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera etiquetas cualitativas basadas en el movimiento y altura.
    """
    df["action_label"] = "idle"

    # Movimiento horizontal
    df.loc[df["speed"] > VEL_THRESH, "action_label"] = "move"
    
    # Saltar (aumento brusco en Z)
    df.loc[df["dz"] > Z_JUMP_THRESH, "action_label"] = "jump"
    
    # Agacharse (descenso leve en Z y baja velocidad)
    df.loc[(df["dz"] < Z_DUCK_THRESH) & (df["speed"] < VEL_THRESH), "action_label"] = "duck"
    
    # Opcional: morir / sin vida
    if "isAlive" in df.columns:
        df.loc[df["isAlive"] == 0, "action_label"] = "dead"
    
    return df


def process_dataset(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    """
    Pipeline principal: carga, deriva features, genera etiquetas y guarda.
    """
    print(f"📥 Cargando dataset desde {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"✅ Dataset cargado con {len(df):,} filas y {len(df.columns)} columnas")

    print("⚙️ Derivando características de movimiento ...")
    df = derive_features(df)

    print("🎯 Generando etiquetas cualitativas ...")
    df = derive_action_labels(df)

    print("📊 Distribución de etiquetas:")
    print(df["action_label"].value_counts(normalize=True).round(3) * 100)

    print(f"💾 Guardando dataset final en {output_path} ...")
    df.to_parquet(output_path, index=False)
    print("✅ Dataset etiquetado guardado con éxito.")


# -------------------------------------------------------
# 🏁 Ejecución directa (CLI)
# -------------------------------------------------------
if __name__ == "__main__":
    process_dataset()
