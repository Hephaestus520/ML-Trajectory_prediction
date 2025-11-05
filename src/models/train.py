"""
train.py
--------
Script de entrenamiento del modelo LSTM para clasificación de acciones.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import ActionClassifierLSTM, get_model_summary
from .dataset import ActionSequenceDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for sequences, labels in progress_bar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar barra de progreso
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Valida el modelo en el conjunto de validación."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for sequences, labels in progress_bar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def plot_training_history(history, save_path):
    """Grafica la historia de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Gráfica guardada en {save_path}")
    plt.close()


def train_model(
    data_path="data/processed/data_merged_labeled.parquet",
    sequence_length=10,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    batch_size=64,
    num_epochs=50,
    learning_rate=0.001,
    train_split=0.7,
    val_split=0.15,
    output_dir="outputs",
    use_class_weights=True
):
    print("debug1")
    """
    Entrena el modelo LSTM.
    
    Args:
        data_path: Ruta a los datos etiquetados
        sequence_length: Longitud de las secuencias
        hidden_size: Tamaño de la capa oculta
        num_layers: Número de capas LSTM
        dropout: Tasa de dropout
        batch_size: Tamaño del batch
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        train_split: Proporción de datos para entrenamiento
        val_split: Proporción de datos para validación
        output_dir: Directorio para guardar resultados
        use_class_weights: Si se usan pesos de clase para balancear
    """
    
    # Crear directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("debug2")
    # Timestamp para identificar el entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    print("debug3")
    print("=" * 70)
    print(f"🚀 ENTRENAMIENTO DEL MODELO LSTM")
    print("=" * 70)
    print(f"📁 Directorio de salida: {run_dir}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}\n")
    
    # Cargar dataset completo
    print("📊 Cargando dataset...")
    full_dataset = ActionSequenceDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        train=True
    )
    
    # Guardar preprocessors
    full_dataset.save_preprocessors(run_dir)
    
    # Calcular pesos de clase
    class_weights = None
    if use_class_weights:
        class_weights = full_dataset.get_class_weights().to(device)
        print(f"\n✅ Usando pesos de clase para balancear entrenamiento")
    
    # Split train/val/test
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Split de datos:")
    print(f"   Train: {len(train_dataset):,} secuencias ({train_split*100:.0f}%)")
    print(f"   Val:   {len(val_dataset):,} secuencias ({val_split*100:.0f}%)")
    print(f"   Test:  {len(test_dataset):,} secuencias ({(1-train_split-val_split)*100:.0f}%)")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 para Windows
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Crear modelo
    input_size = len(full_dataset.feature_cols)
    num_classes = len(full_dataset.classes)
    
    model = ActionClassifierLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    print(f"\n")
    get_model_summary(model, input_size, sequence_length)
    
    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Guardar configuración
    config = {
        "sequence_length": sequence_length,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "classes": list(full_dataset.classes),
        "feature_cols": full_dataset.feature_cols,
        "device": str(device),
        "timestamp": timestamp
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Historia de entrenamiento
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0
    
    # Entrenamiento
    print(f"\n{'='*70}")
    print(f"🎯 INICIANDO ENTRENAMIENTO")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Guardar métricas
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Imprimir resumen
        print(f"\n📊 Resumen Época {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, run_dir / "best_model.pt")
            print(f"   ✅ Mejor modelo guardado! (Val Loss: {val_loss:.4f})")
    
    # Guardar último modelo
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, run_dir / "last_model.pt")
    
    # Guardar historia
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
    
    # Graficar historia
    plot_training_history(history, run_dir / "training_history.png")
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"📁 Resultados guardados en: {run_dir}")
    print(f"🏆 Mejor Val Loss: {best_val_loss:.4f}")
    print(f"🏆 Mejor Val Accuracy: {best_val_acc:.2f}%")
    print(f"\n📝 Archivos generados:")
    print(f"   • best_model.pt - Mejor modelo")
    print(f"   • last_model.pt - Último modelo")
    print(f"   • config.json - Configuración")
    print(f"   • history.json - Historia de entrenamiento")
    print(f"   • training_history.png - Gráficas")
    print(f"   • scaler.pkl - Normalizador de features")
    print(f"   • label_encoder.pkl - Codificador de labels")
    print("=" * 70)
    
    return model, history, run_dir


if __name__ == "__main__":
    # Entrenar modelo con configuración por defecto
    model, history, run_dir = train_model(
        sequence_length=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        batch_size=64,
        num_epochs=50,
        learning_rate=0.001,
        use_class_weights=True
    )
