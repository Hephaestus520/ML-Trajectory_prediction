"""
evaluate.py
-----------
Script para evaluar el modelo entrenado con métricas detalladas.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from .model import ActionClassifierLSTM
from .dataset import ActionSequenceDataset


def evaluate_model(model_dir, data_path=None, batch_size=64):
    """
    Evalúa un modelo entrenado.
    
    Args:
        model_dir: Directorio con el modelo entrenado
        data_path: Ruta opcional a datos de test (usa val si no se provee)
        batch_size: Tamaño del batch
    """
    model_dir = Path(model_dir)
    
    print("=" * 70)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    # Cargar configuración
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)
    
    print(f"\n📁 Modelo: {model_dir.name}")
    print(f"📅 Timestamp: {config['timestamp']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")
    
    # Cargar preprocessors
    scaler, label_encoder = ActionSequenceDataset.load_preprocessors(model_dir)
    
    # Cargar dataset
    if data_path is None:
        data_path = "data/processed/data_merged_labeled.parquet"
    
    print(f"\n📥 Cargando datos desde: {data_path}")
    
    # Crear dataset de test
    test_dataset = ActionSequenceDataset(
        data_path=data_path,
        sequence_length=config['sequence_length'],
        feature_cols=config['feature_cols'],
        scaler=scaler,
        label_encoder=label_encoder,
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Crear y cargar modelo
    model = ActionClassifierLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # Cargar pesos del mejor modelo
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✅ Modelo cargado (época {checkpoint['epoch']})")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Val Acc: {checkpoint['val_acc']:.2f}%")
    
    # Evaluación
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\n🔄 Evaluando modelo...")
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"📈 MÉTRICAS GENERALES")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    # Reporte por clase
    print(f"\n{'='*70}")
    print(f"📊 REPORTE POR CLASE")
    print(f"{'='*70}")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=config['classes'],
        digits=4
    ))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config['classes'],
        yticklabels=config['classes']
    )
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    cm_path = model_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Matriz de confusión guardada en: {cm_path}")
    plt.close()
    
    # Graficar distribución de probabilidades por clase
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(config['classes']):
        if i < len(axes):
            # Probabilidades cuando la etiqueta real es esta clase
            class_mask = all_labels == i
            class_probs = all_probs[class_mask, i]
            
            axes[i].hist(class_probs, bins=50, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Probabilidades - Clase: {class_name}')
            axes[i].set_xlabel('Probabilidad')
            axes[i].set_ylabel('Frecuencia')
            axes[i].grid(True, alpha=0.3)
    
    # Ocultar ejes sobrantes
    for i in range(len(config['classes']), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    probs_path = model_dir / "probability_distributions.png"
    plt.savefig(probs_path, dpi=150, bbox_inches='tight')
    print(f"💾 Distribuciones de probabilidad guardadas en: {probs_path}")
    plt.close()
    
    # Guardar resultados
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': config['classes']
    }
    
    with open(model_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Resultados guardados en: {model_dir / 'evaluation_results.json'}")
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUACIÓN COMPLETADA")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Buscar el modelo más reciente
        output_dir = Path("outputs")
        if output_dir.exists():
            runs = sorted(output_dir.glob("run_*"), key=lambda x: x.name, reverse=True)
            if runs:
                model_dir = runs[0]
                print(f"🔍 Usando modelo más reciente: {model_dir}")
            else:
                print("❌ No se encontraron modelos entrenados en outputs/")
                sys.exit(1)
        else:
            print("❌ Directorio outputs/ no existe")
            sys.exit(1)
    
    evaluate_model(model_dir)
