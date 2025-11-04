"""
model.py
--------
Modelo LSTM para clasificación categórica de acciones de jugadores en CS:GO.

Arquitectura:
    Input: Secuencia de features temporales
    → LSTM layers
    → Fully Connected layers
    → Output: Probabilidades de 5 clases (move, jump, duck, idle, dead)
"""

import torch
import torch.nn as nn


class ActionClassifierLSTM(nn.Module):
    """
    Clasificador LSTM para predecir la siguiente acción del jugador.
    
    Args:
        input_size: Número de features de entrada por timestep
        hidden_size: Número de unidades ocultas en el LSTM
        num_layers: Número de capas LSTM apiladas
        num_classes: Número de clases de salida (5: move, jump, duck, idle, dead)
        dropout: Tasa de dropout entre capas LSTM
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.2
    ):
        super(ActionClassifierLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de normalización
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Capas fully connected
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de forma (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor de forma (batch_size, num_classes) con logits
        """
        # LSTM
        # out: (batch, seq, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # Tomar la salida del último timestep
        # out[:, -1, :] → (batch, hidden_size)
        last_output = out[:, -1, :]
        
        # Batch normalization
        last_output = self.batch_norm(last_output)
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Predicción con probabilidades usando softmax.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor con probabilidades de cada clase
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def predict_class(self, x):
        """
        Predicción de la clase con mayor probabilidad.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor con índices de las clases predichas
        """
        probabilities = self.predict(x)
        return torch.argmax(probabilities, dim=1)


def get_model_summary(model, input_size, sequence_length):
    """
    Imprime un resumen del modelo.
    
    Args:
        model: Modelo de PyTorch
        input_size: Tamaño de features de entrada
        sequence_length: Longitud de la secuencia
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"\nInput shape: (batch_size, {sequence_length}, {input_size})")
    print(f"\nArchitecture:")
    print(model)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Testing ActionClassifierLSTM model\n")
    
    # Parámetros de ejemplo
    batch_size = 32
    sequence_length = 10
    input_size = 14  # x, y, z, velocity, hp, armor, viewX, viewY, dx, dy, dz, speed, acceleration, etc.
    num_classes = 5
    
    # Crear modelo
    model = ActionClassifierLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )
    
    # Mostrar resumen
    get_model_summary(model, input_size, sequence_length)
    
    # Crear datos de prueba
    x = torch.randn(batch_size, sequence_length, input_size)
    
    print("\n🔄 Forward pass test:")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) sample:\n{output[0]}")
    
    # Predicción con probabilidades
    probs = model.predict(x)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Probabilities sample (should sum to 1.0):\n{probs[0]}")
    print(f"Sum: {probs[0].sum():.4f}")
    
    # Predicción de clases
    predictions = model.predict_class(x)
    print(f"\nPredicted classes shape: {predictions.shape}")
    print(f"Predicted classes sample: {predictions[:10]}")
    
    print("\n✅ Model test completed successfully!")
