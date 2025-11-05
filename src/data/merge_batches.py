import pandas as pd
import glob
import os

def merge_batches(
    input_dir="data/processed",
    output_file="data/processed/data_merged_small.parquet",
    round_fraction=0.001,   # ← porcentaje de rondas/jugadores que se conservan
    random_state=42
):
    """
    Une múltiples archivos parquet en un solo DataFrame y toma una muestra
    temporalmente coherente: conserva rondas completas por jugador.

    ✅ Mantiene coherencia temporal (no corta secuencias)
    ✅ Representativo en mapas y jugadores
    ✅ Ideal para entrenamiento de modelos secuenciales (LSTM, GRU)
    """

    # Buscar archivos parquet de batches procesados
    files = sorted(glob.glob(os.path.join(input_dir, "esta_clean_part*.parquet")))
    print(f"🔎 Archivos encontrados: {len(files)}")

    valid_dfs = []
    skipped = []

    for file in files:
        try:
            df = pd.read_parquet(file)
            if len(df) == 0:
                raise ValueError("Archivo vacío")
            valid_dfs.append(df)
            print(f"✅ Cargado {os.path.basename(file)} con {len(df):,} filas")
        except Exception as e:
            print(f"⚠️ Saltado {os.path.basename(file)} ({e})")
            skipped.append(file)

    # Concatenar todos los DataFrames
    if not valid_dfs:
        raise RuntimeError("❌ No se pudieron cargar archivos válidos.")

    df_full = pd.concat(valid_dfs, ignore_index=True)
    print(f"\n📊 Dataset combinado: {len(df_full):,} filas, {df_full.shape[1]} columnas")

    # --------------------------------------------------------
    # 🧠 Muestreo temporalmente coherente
    # --------------------------------------------------------
    # Identificar combinaciones únicas de mapa, ronda y jugador
    unique_keys = df_full[["map", "round", "player"]].drop_duplicates()

    # Muestrear una fracción de rondas completas
    subset_keys = unique_keys.sample(frac=round_fraction, random_state=random_state)

    print(f"🎯 Manteniendo {len(subset_keys):,} combinaciones únicas "
          f"({round_fraction*100:.1f}% del total)")

    # Filtrar el dataset completo
    df_sampled = df_full.merge(subset_keys, on=["map", "round", "player"], how="inner")
    print(f"✅ Dataset reducido a {len(df_sampled):,} filas "
          f"({len(df_sampled)/len(df_full)*100:.2f}% del total)")

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------
    print("\n🔍 Validaciones:")
    print(f" - Duplicados: {df_sampled.duplicated().sum()}")
    print(f" - Valores nulos:\n{df_sampled.isnull().sum()}")

    # --------------------------------------------------------
    # Guardar resultado
    # --------------------------------------------------------
    df_sampled.to_parquet(output_file, index=False)
    print(f"\n💾 Guardado dataset reducido en: {output_file}")

    if skipped:
        print("\n⚠️ Archivos ignorados:")
        for s in skipped:
            print(f" - {os.path.basename(s)}")

    return df_sampled


if __name__ == "__main__":
    # 🔹 Ajusta round_fraction según tu RAM y capacidad de entrenamiento
    # Ejemplo: 0.02 = 2% de las rondas/jugadores
    merge_batches()
