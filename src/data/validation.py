import pandas as pd

def process_dataset():
    df = pd.read_parquet("data/processed/data_merged.parquet")

    print(df.columns)
    print(df.head())
    print(df.describe())

    # Ejemplos rápidos
    print("Mapas únicos:", df["map"].unique())
    print("Límites de posición X/Y/Z:", df[["x", "y", "z"]].describe())

    df = df.sort_values(["map", "round", "player", "tick"])
    df["tick_diff"] = df.groupby(["map", "round", "player"])["tick"].diff()

    print(df["tick_diff"].describe())
    print("Saltos grandes (> 64 ticks):", (df["tick_diff"] > 64).sum())
