"""
Script de prueba simple para verificar el pipeline de procesamiento de datos.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Prueba que todos los imports necesarios funcionen."""
    print("=" * 60)
    print("🧪 TEST 1: Verificando imports")
    print("=" * 60)
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
        
        import numpy as np
        print(f"✅ numpy {np.__version__}")
        
        import torch
        print(f"✅ torch {torch.__version__}")
        
        from sklearn import __version__ as sklearn_version
        print(f"✅ scikit-learn {sklearn_version}")
        
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}")
        
        from tqdm import tqdm
        print(f"✅ tqdm")
        
        import awpy
        print(f"✅ awpy")
        
        print("\n✅ Todos los paquetes importados correctamente!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Error de import: {e}\n")
        return False


def test_data_structure():
    """Verifica que la estructura de datos esté correcta."""
    print("=" * 60)
    print("🧪 TEST 2: Verificando estructura de carpetas")
    print("=" * 60)
    
    required_dirs = {
        "data": "Carpeta principal de datos",
        "data/raw": "Datos originales",
        "data/raw/lan": "Archivos .json.xz del dataset ESTA",
        "data/processed": "Datos procesados",
        "src": "Código fuente",
        "outputs": "Modelos y resultados",
    }
    
    all_exist = True
    for dir_path, description in required_dirs.items():
        path = Path(dir_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        
        extra_info = ""
        if exists and dir_path == "data/raw/lan":
            # Contar archivos .json.xz
            files = list(path.glob("*.json.xz"))
            extra_info = f" ({len(files)} archivos .json.xz)"
        
        print(f"{status} {dir_path:<20} - {description}{extra_info}")
        
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ Estructura de carpetas correcta!\n")
    else:
        print("\n⚠️ Algunas carpetas no existen (se crearán al procesar).\n")
    
    return all_exist


def test_scripts_exist():
    """Verifica que los scripts de procesamiento existan."""
    print("=" * 60)
    print("🧪 TEST 3: Verificando scripts de procesamiento")
    print("=" * 60)
    
    scripts = {
        "src/data/data_prep.py": "Procesamiento de archivos raw",
        "src/data/merge_batches.py": "Combinar batches",
        "src/data/data_der.py": "Features y etiquetas",
        "src/data/validation.py": "Validación de datos",
    }
    
    all_exist = True
    for script_path, description in scripts.items():
        path = Path(script_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {script_path:<25} - {description}")
        
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ Todos los scripts existen!\n")
    else:
        print("\n❌ Algunos scripts no existen.\n")
    
    return all_exist


def test_single_file_load():
    """Intenta cargar un archivo .json.xz si existe."""
    print("=" * 60)
    print("🧪 TEST 4: Probando carga de un archivo de demo")
    print("=" * 60)
    
    raw_dir = Path("data/raw/lan")
    
    if not raw_dir.exists():
        print(f"⚠️ Directorio {raw_dir} no existe.")
        print("   Coloca tus archivos .json.xz allí antes de procesar.\n")
        return False
    
    files = list(raw_dir.glob("*.json.xz"))
    
    if not files:
        print(f"⚠️ No hay archivos .json.xz en {raw_dir}")
        print("   Coloca tus archivos .json.xz allí antes de procesar.\n")
        return False
    
    print(f"📂 Encontrados {len(files)} archivos .json.xz")
    print(f"   Probando con: {files[0].name}\n")
    
    try:
        # Intentar cargar el primer archivo
        sys.path.insert(0, str(Path.cwd()))
        from src.data.data_prep import load_json_xz, extract_player_data
        
        print("   1️⃣ Cargando archivo...")
        demo_data = load_json_xz(str(files[0]))
        
        if not demo_data:
            print("   ❌ No se pudo cargar el archivo\n")
            return False
        
        print("   ✅ Archivo cargado correctamente")
        
        print("   2️⃣ Extrayendo datos de jugadores...")
        rows = extract_player_data(demo_data)
        
        if not rows:
            print("   ❌ No se pudieron extraer datos\n")
            return False
        
        print(f"   ✅ Extraídos {len(rows):,} registros")
        
        # Mostrar una muestra de los datos
        if rows:
            print("\n   📊 Muestra de datos extraídos:")
            print(f"      • Mapa: {rows[0].get('map', 'N/A')}")
            print(f"      • Ronda: {rows[0].get('round', 'N/A')}")
            print(f"      • Jugador: {rows[0].get('player', 'N/A')}")
            print(f"      • Posición: ({rows[0].get('x', 0):.2f}, {rows[0].get('y', 0):.2f}, {rows[0].get('z', 0):.2f})")
            print(f"      • Velocidad: {rows[0].get('velocity', 0):.2f}")
        
        print("\n✅ Test de carga completado exitosamente!\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 60)
    print("🚀 TESTS DEL PIPELINE DE PROCESAMIENTO")
    print("=" * 60 + "\n")
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"📁 Directorio de trabajo: {project_root}\n")
    
    results = []
    
    # Ejecutar tests
    results.append(("Imports de paquetes", test_imports()))
    results.append(("Estructura de carpetas", test_data_structure()))
    results.append(("Scripts de procesamiento", test_scripts_exist()))
    results.append(("Carga de archivo demo", test_single_file_load()))
    
    # Resumen
    print("=" * 60)
    print("📋 RESUMEN DE TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "⚠️ SKIP/FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   Resultado: {passed}/{total} tests pasados")
    
    if passed >= 3:  # Al menos los primeros 3 tests deben pasar
        print("\n" + "=" * 60)
        print("✅ PIPELINE LISTO PARA USAR")
        print("=" * 60)
        print("\n📝 Próximos pasos para procesar tus datos:")
        print("\n1. Asegúrate de tener archivos .json.xz en data/raw/lan/")
        print("\n2. Ejecuta los scripts en orden:")
        print("   python src/data/data_prep.py       # Procesa archivos raw → batches")
        print("   python src/data/merge_batches.py   # Une batches")
        print("   python src/data/data_der.py        # Genera features y etiquetas")
        print("   python src/data/validation.py      # Valida datos procesados")
        print("\n3. Los datos procesados estarán en data/processed/")
        print("   - esta_clean_part*.parquet  (batches)")
        print("   - data_merged.parquet        (datos combinados)")
        print("   - data_merged_labeled.parquet (con etiquetas)")
        print()
    else:
        print("\n⚠️ Algunos tests fallaron. Revisa los errores arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
