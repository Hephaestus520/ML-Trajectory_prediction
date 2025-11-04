"""
dataset.py
----------
PyTorch Dataset para cargar secuencias temporales de acciones de jugadores.

Este dataset crea secuencias de N timesteps para predecir la acción en el timestep N+1.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


class ActionSequenceDataset(Dataset):
    """
    Dataset de secuencias temporales para predicción de acciones.
    
    Args:
        data_path: Ruta al archivo parquet con datos etiquetados
        sequence_length: Número de timesteps en cada secuencia
        feature_cols: Lista de columnas a usar como features
        target_col: Nombre de la columna con las etiquetas
        scaler: StandardScaler pre-entrenado (opcional)
        label_encoder: LabelEncoder pre-entrenado (opcional)
        train: Si es True, ajusta scaler y encoder. Si es False, usa los proporcionados
    """
    
    def __init__(
        self,
        data_path,
        sequence_length=10,
        feature_cols=None,
        target_col="action_label",
        scaler=None,
        label_encoder=None,
        train=True
    ):
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # Cargar datos
        print(f"📥 Cargando datos desde {data_path}...")
        self.df = pd.read_parquet(data_path)
        print(f"✅ Datos cargados: {len(self.df):,} filas")
        
        # Features por defecto
        if feature_cols is None:
            self.feature_cols = [
                'x', 'y', 'z',              # Posición
                'velocity',                  # Velocidad total
                'hp', 'armor',              # Estado del jugador
                'viewX', 'viewY',           # Dirección de la vista
                'dx', 'dy', 'dz',           # Cambios en posición
                'speed',                    # Velocidad horizontal
                'acceleration',             # Aceleración
            ]
        else:
            self.feature_cols = feature_cols
        
        # Verificar que las columnas existan
        missing_cols = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Label Encoder para las clases
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.df[target_col + '_encoded'] = self.label_encoder.fit_transform(self.df[target_col])
            self.classes = self.label_encoder.classes_
            print(f"📊 Clases detectadas: {list(self.classes)}")
        else:
            self.label_encoder = label_encoder
            self.classes = self.label_encoder.classes_
            self.df[target_col + '_encoded'] = self.label_encoder.transform(self.df[target_col])
        
        # StandardScaler para normalizar features
        if scaler is None and train:
            self.scaler = StandardScaler()
            self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
            print(f"✅ Scaler ajustado a los datos de entrenamiento")
        else:
            self.scaler = scaler
            if self.scaler is not None:
                self.df[self.feature_cols] = self.scaler.transform(self.df[self.feature_cols])
                print(f"✅ Features normalizadas con scaler existente")
        
        # Crear secuencias
        self.sequences = []
        self.labels = []
        
        print(f"🔄 Creando secuencias de longitud {sequence_length}...")
        self._create_sequences()
        print(f"✅ Creadas {len(self.sequences):,} secuencias")
        
    def _create_sequences(self):
        """Crea secuencias temporales agrupadas por jugador y ronda."""
        # Agrupar por mapa, ronda y jugador
        grouped = self.df.groupby(['map', 'round', 'player'])
        
        for name, group in grouped:
            # Ordenar por tick
            group = group.sort_values('tick')
            
            # Crear secuencias deslizantes
            for i in range(len(group) - self.sequence_length):
                # Secuencia de features
                sequence = group[self.feature_cols].iloc[i:i + self.sequence_length].values
                
                # Label (acción en el siguiente timestep)
                label = group[self.target_col + '_encoded'].iloc[i + self.sequence_length]
                
                self.sequences.append(sequence)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Retorna una secuencia y su label.
        
        Returns:
            sequence: Tensor de forma (sequence_length, num_features)
            label: Tensor con el índice de la clase
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return sequence, label
    
    def get_class_weights(self):
        """
        Calcula pesos de clases para balancear el entrenamiento.
        
        Returns:
            Tensor con pesos por clase
        """
        labels = np.array(self.labels)
        unique, counts = np.unique(labels, return_counts=True)
        
        # Peso inversamente proporcional a la frecuencia
        total = len(labels)
        weights = total / (len(unique) * counts)
        
        print("\n📊 Distribución de clases:")
        for i, (cls, count, weight) in enumerate(zip(self.classes, counts, weights)):
            pct = count / total * 100
            print(f"   {cls:8s}: {count:>8,} ({pct:>5.2f}%) - peso: {weight:.3f}")
        
        return torch.FloatTensor(weights)
    
    def save_preprocessors(self, save_dir):
        """
        Guarda el scaler y label encoder para uso futuro.
        
        Args:
            save_dir: Directorio donde guardar los archivos
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(save_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"💾 Preprocessors guardados en {save_dir}")
    
    @staticmethod
    def load_preprocessors(save_dir):
        """
        Carga el scaler y label encoder guardados.
        
        Args:
            save_dir: Directorio donde están guardados los archivos
            
        Returns:
            scaler, label_encoder
        """
        save_dir = Path(save_dir)
        
        with open(save_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        with open(save_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        print(f"📥 Preprocessors cargados desde {save_dir}")
        return scaler, label_encoder


if __name__ == "__main__":
    # Ejemplo de uso (requiere datos procesados)
    print("🧪 Testing ActionSequenceDataset\n")
    
    data_path = "data/processed/data_merged_labeled.parquet"
    
    if Path(data_path).exists():
        # Crear dataset
        dataset = ActionSequenceDataset(
            data_path=data_path,
            sequence_length=10,
            train=True
        )
        
        print(f"\n📊 Dataset info:")
        print(f"   Total sequences: {len(dataset):,}")
        print(f"   Sequence shape: {dataset[0][0].shape}")
        print(f"   Number of features: {dataset[0][0].shape[1]}")
        print(f"   Number of classes: {len(dataset.classes)}")
        
        # Mostrar una muestra
        sequence, label = dataset[0]
        print(f"\n📝 Muestra:")
        print(f"   Sequence shape: {sequence.shape}")
        print(f"   Label: {label.item()} ({dataset.classes[label]})")
        
        # Calcular pesos de clase
        weights = dataset.get_class_weights()
        
        print("\n✅ Dataset test completed!")
    else:
        print(f"⚠️ Archivo no encontrado: {data_path}")
        print("   Primero debes procesar los datos ejecutando:")
        print("   python src/data_prep.py")
        print("   python src/merge_batches.py")
        print("   python src/data_der.py")
