"""
main.py
-------
Script principal para ejecutar todo el pipeline:
1. Procesamiento de datos
2. Entrenamiento del modelo
3. Evaluación del modelo
"""

import sys
from pathlib import Path


def main():
    """Menú principal del proyecto."""
    print("\n" + "=" * 70)
    print("🎮 ML TRAJECTORY PREDICTION - CS:GO")
    print("Predicción Categórica de Acciones de Jugadores")
    print("=" * 70)
    
    print("\n📋 Opciones:")
    print("   1. Procesar datos (data_prep.py → merge_batches.py → data_der.py)")
    print("   2. Validar datos procesados")
    print("   3. Probar modelo LSTM")
    print("   4. Entrenar modelo")
    print("   5. Evaluar modelo")
    print("   6. Salir")
    
    choice = input("\n👉 Selecciona una opción (1-6): ").strip()
    
    if choice == "1":
        process_data()
    elif choice == "2":
        validate_data()
    elif choice == "3":
        test_model()
    elif choice == "4":
        train_model_menu()
    elif choice == "5":
        evaluate_model_menu()
    elif choice == "6":
        print("\n👋 ¡Hasta luego!")
        sys.exit(0)
    else:
        print("\n❌ Opción inválida")


def process_data():
    """Ejecuta el pipeline completo de procesamiento de datos."""
    print("\n" + "=" * 70)
    print("📊 PROCESAMIENTO DE DATOS")
    print("=" * 70)
    
    # Verificar que existan archivos raw
    raw_dir = Path("data/raw/lan")
    if not raw_dir.exists():
        print(f"\n❌ Directorio {raw_dir} no existe")
        return
    
    files = list(raw_dir.glob("*.json.xz"))
    if not files:
        print(f"\n❌ No hay archivos .json.xz en {raw_dir}")
        return
    
    print(f"\n✅ Encontrados {len(files)} archivos .json.xz")
    confirm = input(f"\n¿Procesar todos los archivos? (s/n): ").strip().lower()
    
    if confirm != 's':
        print("Operación cancelada")
        return
    
    print("\n🔄 Iniciando procesamiento...")
    print("   Esto puede tardar 30-60 minutos dependiendo de tu hardware.")
    print("   Asegúrate de tener suficiente espacio en disco (~5-10 GB)")
    
    # Paso 1: Procesar archivos raw
    print("\n" + "-" * 70)
    print("PASO 1/3: Procesando archivos raw → batches Parquet")
    print("-" * 70)
    
    from src.data.data_prep import process_all_files
    process_all_files(batch_size=100)
    
    # Paso 2: Combinar batches
    print("\n" + "-" * 70)
    print("PASO 2/3: Combinando batches")
    print("-" * 70)
    
    from src.data.merge_batches import merge_batches
    merge_batches()
    
    # Paso 3: Generar features y etiquetas
    print("\n" + "-" * 70)
    print("PASO 3/3: Generando features y etiquetas categóricas")
    print("-" * 70)
    
    from src.data.data_der import process_dataset
    process_dataset()
    
    print("\n" + "=" * 70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 70)
    print("\nArchivos generados en data/processed/:")
    print("   • esta_clean_part*.parquet - Batches procesados")
    print("   • data_merged.parquet - Datos combinados")
    print("   • data_merged_labeled.parquet - Datos con etiquetas")


def validate_data():
    """Valida los datos procesados."""
    print("\n" + "=" * 70)
    print("🔍 VALIDACIÓN DE DATOS")
    print("=" * 70)
    
    data_file = Path("data/processed/data_merged_labeled.parquet")
    
    if not data_file.exists():
        print(f"\n❌ Archivo no encontrado: {data_file}")
        print("   Primero debes procesar los datos (opción 1)")
        return
    
    from src.data.validation import process_dataset
    process_dataset()


def test_model():
    """Prueba la arquitectura del modelo."""
    print("\n" + "=" * 70)
    print("🧪 PRUEBA DEL MODELO LSTM")
    print("=" * 70)
    
    from src.models.model import ActionClassifierLSTM, get_model_summary
    import torch
    
    # Parámetros
    batch_size = 32
    sequence_length = 10
    input_size = 14
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
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    probs = model.predict(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    
    print("\n✅ Modelo funcionando correctamente!")


def train_model_menu():
    """Menú para entrenar el modelo."""
    print("\n" + "=" * 70)
    print("🎯 ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    # Verificar que existan datos procesados
    data_file = Path("data/processed/data_merged_labeled.parquet")
    
    if not data_file.exists():
        print(f"\n❌ Archivo no encontrado: {data_file}")
        print("   Primero debes procesar los datos (opción 1)")
        return
    
    print("\n⚙️ Configuración del entrenamiento:")
    print("   (Presiona Enter para usar valores por defecto)")
    
    try:
        seq_len = input("\n   Longitud de secuencia [10]: ").strip()
        sequence_length = int(seq_len) if seq_len else 10
        
        h_size = input("   Hidden size [128]: ").strip()
        hidden_size = int(h_size) if h_size else 128
        
        n_layers = input("   Número de capas LSTM [2]: ").strip()
        num_layers = int(n_layers) if n_layers else 2
        
        b_size = input("   Batch size [64]: ").strip()
        batch_size = int(b_size) if b_size else 64
        
        epochs = input("   Número de épocas [50]: ").strip()
        num_epochs = int(epochs) if epochs else 50
        
        lr = input("   Learning rate [0.001]: ").strip()
        learning_rate = float(lr) if lr else 0.001
        
    except ValueError:
        print("\n❌ Valor inválido. Usando configuración por defecto.")
        sequence_length = 10
        hidden_size = 128
        num_layers = 2
        batch_size = 64
        num_epochs = 50
        learning_rate = 0.001
    
    print(f"\n📋 Configuración final:")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    confirm = input("\n¿Iniciar entrenamiento? (s/n): ").strip().lower()
    
    if confirm != 's':
        print("Entrenamiento cancelado")
        return
    
    # Entrenar
    from src.models.train import train_model
    
    model, history, run_dir = train_model(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_class_weights=True
    )
    
    print(f"\n✅ Modelo guardado en: {run_dir}")


def evaluate_model_menu():
    """Menú para evaluar el modelo."""
    print("\n" + "=" * 70)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    # Buscar modelos entrenados
    output_dir = Path("outputs")
    
    if not output_dir.exists():
        print("\n❌ No hay modelos entrenados")
        print("   Primero debes entrenar un modelo (opción 4)")
        return
    
    runs = sorted(output_dir.glob("run_*"), key=lambda x: x.name, reverse=True)
    
    if not runs:
        print("\n❌ No se encontraron modelos entrenados")
        return
    
    print("\n📁 Modelos disponibles:")
    for i, run in enumerate(runs, 1):
        print(f"   {i}. {run.name}")
    
    choice = input(f"\nSelecciona un modelo (1-{len(runs)}) o Enter para el más reciente: ").strip()
    
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                model_dir = runs[idx]
            else:
                print("❌ Opción inválida. Usando el más reciente.")
                model_dir = runs[0]
        except ValueError:
            print("❌ Opción inválida. Usando el más reciente.")
            model_dir = runs[0]
    else:
        model_dir = runs[0]
    
    print(f"\n📂 Evaluando modelo: {model_dir.name}")
    
    from src.models.evaluate import evaluate_model
    evaluate_model(model_dir)


if __name__ == "__main__":
    while True:
        try:
            main()
            
            print("\n" + "-" * 70)
            continue_choice = input("¿Realizar otra operación? (s/n): ").strip().lower()
            if continue_choice != 's':
                print("\n👋 ¡Hasta luego!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            retry = input("\n¿Intentar de nuevo? (s/n): ").strip().lower()
            if retry != 's':
                break
