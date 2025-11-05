import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from .model import ActionClassifierLSTM

# -------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------
RUN_DIR = Path("outputs/run_20251105_005017")
SEQ_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# 1. CARGAR MODELO Y PREPROCESADORES
# -------------------------------------------------------------------
print(f"📦 Cargando modelo desde {RUN_DIR}")

with open(RUN_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open(RUN_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model_files = list(RUN_DIR.glob("*.pt")) + list(RUN_DIR.glob("*.pth"))
if not model_files:
    raise FileNotFoundError(f"No se encontró ningún archivo .pt o .pth en {RUN_DIR}")
model_path = model_files[0]
print(f"✅ Archivo de modelo encontrado: {model_path.name}")

checkpoint = torch.load(model_path, map_location=DEVICE)

# -------------------------------------------------------------------
# 2. DETECTAR FORMATO DEL CHECKPOINT
# -------------------------------------------------------------------
if "model_state" in checkpoint:
    print("🧠 Formato: {'model_state', 'model_params'} detectado")
    model_params = checkpoint.get("model_params", {})
    model = ActionClassifierLSTM(**model_params)
    model.load_state_dict(checkpoint["model_state"])

elif "model_state_dict" in checkpoint:
    print("🧩 Formato: checkpoint de entrenamiento detectado")
    # Usa los parámetros del entrenamiento; ajusta si cambiaste en train.py
    model = ActionClassifierLSTM(
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_classes=len(label_encoder.classes_),
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

else:
    print("⚙️ Formato: state_dict puro detectado")
    model = ActionClassifierLSTM(
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_classes=len(label_encoder.classes_),
        dropout=0.3,
    )
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("✅ Modelo cargado correctamente")

# -------------------------------------------------------------------
# 3. EJEMPLO DE SECUENCIA
# -------------------------------------------------------------------
feature_cols = [
    'x', 'y', 'z',
    'velocity',
    'hp', 'armor',
    'viewX', 'viewY',
    'dx', 'dy', 'dz',
    'speed',
    'acceleration'
]

example_sequence = np.random.rand(SEQ_LEN, len(feature_cols)).astype(np.float32)
example_sequence = scaler.transform(example_sequence)
X = torch.tensor(example_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# -------------------------------------------------------------------
# 4. HACER PREDICCIÓN
# -------------------------------------------------------------------
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

print("\n🔮 RESULTADOS DE PREDICCIÓN")
print(f"Acción predicha: {pred_label}")
print("Probabilidades por clase:")
for cls, p in zip(label_encoder.classes_, probs):
    print(f"   {cls:10s}: {p:.3f}")
