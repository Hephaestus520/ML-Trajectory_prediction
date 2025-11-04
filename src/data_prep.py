import lzma
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_trajectory_files(data_dir):
    json_files = list(Path(data_dir).rglob("*.json.xz"))
    print(f"Archivos de trayectorias encontrados: {len(json_files)}")
    print(json_files)
    all_data = []

    for file in tqdm(json_files, desc="Cargando JSONs comprimidos"):
        try:
            with lzma.open(file, "rt") as f:  # 't' = texto, no binario
                data = json.load(f)
            # Aquí asumimos que el contenido tiene estructura tipo 'positions'
            if "positions" in data:
                for rec in data["positions"]:
                    if isinstance(rec, dict) and "pos" in rec:
                        all_data.append({
                            "player_id": rec.get("player_id"),
                            "tick": rec.get("tick"),
                            "x": rec["pos"].get("x"),
                            "y": rec["pos"].get("y"),
                            "z": rec["pos"].get("z")
                        })
        except Exception as e:
            print(f"⚠️ Error al leer {file.name}: {e}")

    df = pd.DataFrame(all_data)
    return df


def main():
    data_dir = Path("E:/ML/ML-Trajectory_prediction/data/lan")
    df = load_trajectory_files(data_dir)

    print("Columnas detectadas:", df.columns.tolist())
    print(df.head())

    if not df.empty:
        df = df.dropna(subset=["x", "y", "z"])
        df = df.sort_values(by=["player_id", "tick"]).reset_index(drop=True)
        output = Path("E:/ML/ML-Trajectory_prediction/data/processed/lan_clean.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"✅ Data procesada y guardada en {output}")
    else:
        print("⚠️ DataFrame vacío: revisa la estructura interna de los JSONs.")


if __name__ == "__main__":
    main()
