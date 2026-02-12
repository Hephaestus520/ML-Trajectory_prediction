📊 EVALUACIÓN DEL MODELO
======================================================================

📁 Modelo: run_20251105_005017
📅 Timestamp: 20251105_005017
🖥️ Dispositivo: cpu
📥 Preprocessors cargados desde outputs\run_20251105_005017

📥 Cargando datos desde: data/processed/data_merged_labeled.parquet
📥 Cargando datos desde data/processed/data_merged_labeled.parquet...
✅ Datos cargados: 33,419 filas
✅ Features normalizadas con scaler existente
🔄 Creando secuencias de longitud 10...
✅ Creadas 32,989 secuencias

✅ Modelo cargado (época 11)
   Val Loss: 0.3907
   Val Acc: 86.70%

🔄 Evaluando modelo...

======================================================================
📈 MÉTRICAS GENERALES
======================================================================
Accuracy:  87.90%
Precision: 90.04%
Recall:    87.90%
F1-Score:  88.36%

======================================================================
📊 REPORTE POR CLASE
======================================================================
              precision    recall  f1-score   support

        dead     0.9706    0.9777    0.9741      9222
        duck     0.2250    0.9474    0.3636        19
        idle     0.5578    0.9473    0.7021      2145
        jump     0.7946    0.8770    0.8337      5535
        move     0.9430    0.8139    0.8737     16068

    accuracy                         0.8790     32989
   macro avg     0.6982    0.9126    0.7495     32989
weighted avg     0.9004    0.8790    0.8836     32989


💾 Matriz de confusión guardada en: outputs\run_20251105_005017\confusion_matrix.png
💾 Distribuciones de probabilidad guardadas en: outputs\run_20251105_005017\probability_distributions.png

💾 Resultados guardados en: outputs\run_20251105_005017\evaluation_results.json

======================================================================
✅ EVALUACIÓN COMPLETADA
======================================================================

----------------------------------------------------------------------