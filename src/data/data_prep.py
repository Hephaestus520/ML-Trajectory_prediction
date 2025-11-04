# src/data_prep.py
import os
import json
import lzma
import pandas as pd
from tqdm import tqdm
import gc

RAW_DIR = "data/raw/lan"
PROCESSED_DIR = "data/processed"

def load_json_xz(file_path: str):
    """Carga un archivo comprimido .json.xz de manera segura."""
    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ No se pudo leer {file_path}: {e}")
        return None

def extract_player_data(demo_data):
    if not demo_data or not isinstance(demo_data, dict):
        return []

    all_rows = []
    map_name = demo_data.get("mapName", "unknown")

    for rnd in (demo_data.get("gameRounds") or []):
        round_num = rnd.get("roundNum")

        for fr in (rnd.get("frames") or []):
            tick = fr.get("tick")
            for side_key in ("ct", "t"):
                side_data = fr.get(side_key) or {}
                for player in (side_data.get("players") or []):
                    try:
                        vx = player.get("velocityX", 0) or 0
                        vy = player.get("velocityY", 0) or 0
                        vz = player.get("velocityZ", 0) or 0
                        velocity = (vx**2 + vy**2 + vz**2)**0.5

                        all_rows.append({
                            "map": map_name,
                            "round": round_num,
                            "tick": tick,
                            "side": player.get("side", side_key.upper()),
                            "player": player.get("name"),
                            "x": player.get("x"),
                            "y": player.get("y"),
                            "z": player.get("z"),
                            "hp": player.get("hp"),
                            "armor": player.get("armor"),
                            "velocity": velocity,
                            "viewX": player.get("viewX"),
                            "viewY": player.get("viewY"),
                            "isAlive": player.get("isAlive"),
                            "activeWeapon": player.get("activeWeapon"),
                        })
                    except Exception:
                        continue
    return all_rows

def process_all_files(batch_size=100):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_data = []
    batch_idx = 0

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json.xz")]

    for i, file_name in enumerate(tqdm(files)):
        file_path = os.path.join(RAW_DIR, file_name)
        demo_data = load_json_xz(file_path)
        if demo_data:
            rows = extract_player_data(demo_data)
            if rows:
                all_data.extend(rows)

        # Guardar cada cierto número de archivos procesados
        if (i + 1) % batch_size == 0:
            if all_data:
                df = pd.DataFrame(all_data)
                output_path = os.path.join(PROCESSED_DIR, f"esta_clean_part{batch_idx}.parquet")
                df.to_parquet(output_path)
                print(f"💾 Guardado batch {batch_idx} con {len(df)} filas.")
                batch_idx += 1
                all_data.clear()
                gc.collect()

    # Guardar lo que quede
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = os.path.join(PROCESSED_DIR, f"esta_clean_part{batch_idx}.parquet")
        df.to_parquet(output_path)
        print(f"💾 Guardado batch final {batch_idx} con {len(df)} filas.")

    print("✅ Procesamiento completo.")

if __name__ == "__main__":
    process_all_files(batch_size=100)
