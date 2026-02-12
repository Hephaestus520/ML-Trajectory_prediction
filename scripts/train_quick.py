"""
train_quick.py
--------------
Script de entrenamiento rápido optimizado para ejecución en 1 hora.

Configuración:
- 20% de los datos (6.8M filas)
- Máximo 1M de secuencias
- 20 épocas
- Batch size 256
- Dropout 0.3
"""

from src.models.train import train_model

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 ENTRENAMIENTO RÁPIDO - Configuración Optimizada")
    print("=" * 70)
    print("\n📊 Configuración:")
    print("   • Datos: 20% del dataset (~6.8M filas)")
    print("   • Secuencias máximas: 1,000,000")
    print("   • Épocas: 20")
    print("   • Batch size: 256")
    print("   • Hidden size: 128")
    print("   • Layers: 2")
    print("   • Dropout: 0.3")
    print("   • Learning rate: 0.001")
    print("\n⏱️ Tiempo estimado: 40-50 minutos")
    print("=" * 70)
    
    input("\n⏸️  Presiona ENTER para iniciar el entrenamiento...")
    
    # Entrenar modelo
    model, history, run_dir = train_model(
        sequence_length=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        batch_size=256,
        num_epochs=20,
        learning_rate=0.001,
        use_class_weights=True
    )
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"\n📁 Resultados guardados en: {run_dir}")
    print("\n📝 Próximos pasos:")
    print("   1. Revisar gráficas en: outputs/run_*/training_history.png")
    print("   2. Evaluar modelo: python main.py → Opción 5")
    print("   3. Hacer predicciones: python main.py → Opción 6")
    print("=" * 70)
