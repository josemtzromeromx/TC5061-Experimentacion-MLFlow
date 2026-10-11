# Mapeo del notebook baseline a la nueva arquitectura

Este documento explica cómo se refactoriza el notebook previo sin perder su lógica ni sus decisiones metodológicas.

| Bloque del notebook | Destino | Decisión |
|---|---|---|
| Carga de `Bank_Personal_Loan_Modelling.csv` | `src/dataset.py` | Se convierte en `load_dataset()` |
| Validación de unicidad de `ID` | `src/dataset.py` | Validación de esquema |
| Valores faltantes y reglas de calidad | Notebook + `dataset.py` | El diagnóstico detallado queda en EDA; las reglas necesarias para ejecución se validan |
| Eliminación de `Experience < 0` | `src/dataset.py` | Se preserva exactamente la decisión del notebook |
| Eliminación de duplicados | `src/dataset.py` | Limpieza determinista |
| Conversión de tipos para EDA | Notebook | Es relevante para visualización, no para la ejecución del modelo |
| Histogramas/KDE/boxplots/barras/scatter/heatmap | Notebook | EDA; no debe ejecutarse en cada entrenamiento |
| Selección de 11 features | `src/features.py` | `FEATURE_COLUMNS` centraliza la definición |
| Target `Personal Loan` | `src/features.py` | `TARGET_COLUMN` y `prepare_features()` |
| `StandardScaler` | `src/features.py` | Preprocesador reutilizable, ajustado dentro del Pipeline |
| `train_test_split(..., stratify=y)` | `src/train.py` | `split_data()` con `random_state` y `test_size` parametrizables |
| `LogisticRegression` | `src/train.py` | `build_model()` recibe hiperparámetros CLI |
| Predicciones | `src/train.py` | Parte de `evaluate_model()` |
| Accuracy/Precision/Recall/F1/ROC-AUC | `src/train.py` | Métricas centralizadas y registradas en MLflow |
| Matriz de confusión | `src/train.py` | Artefacto PNG reproducible |
| Curva ROC | `src/train.py` | Artefacto PNG reproducible |
| `classification_report` | `src/train.py` | Artefacto JSON reproducible |
| Configuración de MLflow | `src/train.py` | Tracking URI y experiment name configurables |
| Run único baseline | `src/train.py` | Se transforma en N corridas parametrizadas |
| Evidencias de MLflow | PDF final | Capturas de comparación y detalle de una corrida |

## Principio de diseño

El notebook queda como documento de exploración y razonamiento. La lógica repetible del sistema sale del notebook y queda en módulos Python que pueden ejecutarse desde la terminal, versionarse con Git y reutilizarse sin copiar y pegar código.
