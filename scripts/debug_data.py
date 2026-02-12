"""
Script para verificar la estructura del archivo parquet
"""
import pandas as pd
from pathlib import Path

data_path = "data/processed/data_merged_labeled.parquet"

if Path(data_path).exists():
    print("📂 Leyendo archivo parquet...")
    df = pd.read_parquet(data_path)
    
    print(f"\n📊 INFORMACIÓN DEL DATAFRAME:")
    print(f"   Filas: {len(df):,}")
    print(f"   Columnas: {len(df.columns):,}")
    print(f"   Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 COLUMNAS ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        print(f"   {i:3d}. {col:30s} - {dtype:10s} ({nulls:,} nulos)")
    
    print(f"\n🔍 PRIMERAS 5 FILAS:")
    print(df.head())
    
    print(f"\n📈 INFO DEL DATAFRAME:")
    print(df.info(memory_usage='deep'))
    
    # Verificar columnas esperadas
    expected_cols = [
        'map', 'tick', 'round', 'player',
        'x', 'y', 'z', 'velocity',
        'hp', 'armor', 'viewX', 'viewY',
        'dx', 'dy', 'dz', 'speed', 'acceleration',
        'action_label'
    ]
    
    print(f"\n✅ COLUMNAS ESPERADAS:")
    for col in expected_cols:
        status = "✅" if col in df.columns else "❌"
        print(f"   {status} {col}")
    
    # Columnas extra
    extra_cols = set(df.columns) - set(expected_cols)
    if extra_cols:
        print(f"\n⚠️ COLUMNAS EXTRA ({len(extra_cols)}):")
        for col in sorted(extra_cols)[:20]:  # Mostrar solo las primeras 20
            print(f"   - {col}")
        if len(extra_cols) > 20:
            print(f"   ... y {len(extra_cols) - 20} más")
else:
    print(f"❌ Archivo no encontrado: {data_path}")
