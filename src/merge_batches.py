import pandas as pd
import glob
import os

def merge_batches(input_dir="data/processed", output_file="data/processed/data_merged.parquet"):
    """
    Une múltiples archivos parquet en un solo DataFrame,
    ignorando archivos corruptos o incompletos.
    """

    # Buscar archivos parquet
    files = sorted(glob.glob(os.path.join(input_dir, "esta_clean*.parquet")))
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

    df_final = pd.concat(valid_dfs, ignore_index=True)
    print(f"\n📊 Dataset combinado: {len(df_final):,} filas, {df_final.shape[1]} columnas")

    # Validaciones básicas
    print("\n🔍 Validaciones:")
    print(f" - Duplicados: {df_final.duplicated().sum()}")
    print(f" - Valores nulos:\n{df_final.isnull().sum()}")

    # Guardar resultado final
    df_final.to_parquet(output_file, index=False)
    print(f"\n💾 Guardado dataset final en: {output_file}")

    if skipped:
        print("\n⚠️ Archivos ignorados:")
        for s in skipped:
            print(f" - {os.path.basename(s)}")

    return df_final


if __name__ == "__main__":
    df = merge_batches()
