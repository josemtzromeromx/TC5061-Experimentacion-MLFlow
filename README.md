# TC5061 - Experimentación con MLflow

Repositorio individual correspondiente a la actividad de la semana 4 de la asignatura **MLOps TC5061.10**.

El proyecto toma como base el notebook desarrollado en la actividad anterior y transforma el flujo exploratorio en una estructura modular, reproducible y parametrizable, utilizando **MLflow Tracking** para registrar parámetros, métricas, artefactos y el modelo entrenado en cada corrida.

---

## 1. Objetivo del proyecto

El objetivo es migrar el modelo baseline desarrollado originalmente en un notebook hacia un flujo de experimentación reproducible y ejecutable desde línea de comandos.

El proyecto permite:

- separar la carga y limpieza de datos de la lógica de entrenamiento;
- encapsular la preparación de variables y el preprocesamiento;
- ejecutar el entrenamiento desde `train.py`;
- modificar hiperparámetros sin editar el código fuente;
- registrar experimentos con MLflow;
- comparar múltiples corridas;
- guardar métricas, artefactos y el modelo;
- reproducir resultados utilizando una semilla fija;
- mantener una estructura de proyecto inspirada en **Cookiecutter Data Science**.

---

## 2. Modelo utilizado

El modelo utilizado es `LogisticRegression` y el preprocesamiento utiliza `StandardScaler`.

Ambos componentes se integran en un `Pipeline` de Scikit-learn para asegurar que el mismo preprocesamiento utilizado durante el entrenamiento también sea aplicado durante la evaluación.

La variable objetivo es:

```text
Personal Loan
```

---

## 3. Estructura del proyecto

```text
TC5061-Experimentacion-MLFlow/
│
├── data/
│   └── raw/
│       ├── Bank_Personal_Loan_Modelling.csv
│       └── README.md
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── TC5061_Notebook_Baseline_MartinezJose.ipynb
│
├── reports/
│   └── figures/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── features.py
│   └── train.py
│
├── .gitignore
├── Makefile
├── PROJECT_STRUCTURE.md
├── README.md
└── requirements.txt
```

---

## 4. Descripción de los archivos principales

### `notebooks/TC5061_Notebook_Baseline_MartinezJose.ipynb`

Conserva el notebook de la actividad anterior como referencia para:

- exploración de datos;
- análisis estadístico;
- visualizaciones;
- revisión de calidad de datos;
- documentación del modelo baseline.

La lógica repetible de entrenamiento ya no depende de ejecutar manualmente las celdas del notebook.

### `src/dataset.py`

Contiene la lógica relacionada con la carga, validación y limpieza del dataset.

Entre sus responsabilidades se encuentran:

- cargar el archivo CSV;
- validar que las columnas requeridas existan;
- validar la columna `ID`;
- eliminar registros con valores negativos en `Experience`;
- eliminar registros duplicados;
- validar variables utilizadas por el modelo;
- generar información de control sobre la limpieza realizada.

### `src/features.py`

Contiene la definición de las variables utilizadas por el modelo y la preparación de `X` e `y`.

Variables predictoras utilizadas:

```text
Age
Experience
Income
Family
CCAvg
Education
Mortgage
Securities Account
CD Account
Online
CreditCard
```

Variable objetivo:

```text
Personal Loan
```

También contiene la lógica necesaria para construir el preprocesamiento mediante `StandardScaler`.

### `src/train.py`

Es el punto de entrada ejecutable del proyecto.

Sus principales responsabilidades son:

- leer parámetros desde línea de comandos con `argparse`;
- cargar y limpiar los datos;
- preparar `X` e `y`;
- dividir datos de entrenamiento y prueba;
- construir el pipeline;
- configurar `LogisticRegression`;
- entrenar el modelo;
- generar predicciones;
- calcular métricas;
- generar artefactos;
- registrar parámetros, métricas, artefactos y modelo en MLflow;
- mostrar un resumen de resultados al finalizar cada corrida.

### `requirements.txt`

Contiene las versiones exactas de las librerías instaladas en el ambiente utilizado para ejecutar los experimentos.

Dependencias principales:

```text
mlflow==3.16.1
matplotlib==3.11.2
numpy==2.5.3
pandas==3.0.6
scikit-learn==1.9.1
seaborn==0.13.2
```

### `PROJECT_STRUCTURE.md`

Documenta la separación del notebook original hacia la estructura modular del proyecto.

### `.gitignore`

Evita versionar archivos temporales o locales, entre ellos:

```text
.DS_Store
__pycache__/
*.py[cod]
mlflow.db
mlruns/
mlartifacts/
venv/
.venv/
mlops/
*.log
```

---

## 5. Ambiente de ejecución

Se recomienda ejecutar el proyecto dentro de un ambiente virtual de Python.

En el desarrollo de esta actividad se utilizó un ambiente llamado:

```text
mlops
```

Ejemplo de activación en macOS/Linux:

```bash
source /ruta/al/ambiente/mlops/bin/activate
```

Después de activarlo, se puede verificar con:

```bash
which python
python --version
mlflow --version
```

---

## 6. Instalación de dependencias

Desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
```

Para validar que las librerías principales están disponibles:

```bash
python -c "import pandas, numpy, sklearn, mlflow, matplotlib; print('Dependencias OK')"
```

---

## 7. Verificación del script

Antes de ejecutar experimentos, se puede consultar la ayuda del script:

```bash
python src/train.py --help
```

El script expone los siguientes argumentos:

```text
--data-path
--C
--solver
--class-weight
--max-iter
--random-state
--test-size
--tracking-uri
--experiment-name
--run-name
```

---

## 8. Parámetros disponibles

### `--data-path`

Ruta al archivo CSV de entrada.

Valor por defecto:

```text
data/raw/Bank_Personal_Loan_Modelling.csv
```

### `--C`

Inverso de la fuerza de regularización de `LogisticRegression`.

Valor por defecto:

```text
1.0
```

### `--solver`

Algoritmo de optimización utilizado por `LogisticRegression`.

Opciones disponibles:

```text
lbfgs
liblinear
newton-cg
newton-cholesky
sag
saga
```

Valor por defecto:

```text
lbfgs
```

### `--class-weight`

Estrategia de ponderación de clases.

Opciones:

```text
none
balanced
```

Valor por defecto:

```text
none
```

### `--max-iter`

Número máximo de iteraciones permitidas al optimizador.

Valor por defecto:

```text
1000
```

### `--random-state`

Semilla aleatoria utilizada para garantizar reproducibilidad.

Valor utilizado en la actividad:

```text
4284
```

### `--test-size`

Proporción del dataset reservada para prueba.

Valor por defecto:

```text
0.33
```

### `--tracking-uri`

Dirección del servidor de MLflow Tracking.

Valor por defecto:

```text
http://127.0.0.1:5000
```

### `--experiment-name`

Nombre del experimento utilizado para agrupar todas las corridas.

Valor utilizado:

```text
TC5061_Bank_Personal_Loan_Experiments
```

### `--run-name`

Nombre específico de cada corrida.

Ejemplos:

```text
run01_baseline
run02_C_0_1
run03_C_10
run04_balanced
run05_liblinear
run06_reproducibility
```

---

## 9. Inicio del servidor de MLflow

MLflow debe estar activo antes de ejecutar las corridas.

Desde una terminal independiente:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

La interfaz queda disponible en:

```text
http://127.0.0.1:5000
```

La terminal donde se inicia MLflow debe mantenerse abierta mientras se ejecutan los experimentos.

MLflow puede generar localmente archivos como:

```text
mlflow.db
mlartifacts/
mlruns/
```

Estos archivos corresponden al estado local de los experimentos y están excluidos del repositorio mediante `.gitignore`.

---

## 10. Ejecución del baseline

El script puede ejecutarse utilizando todos los valores por defecto:

```bash
python src/train.py
```

Para registrar explícitamente la primera corrida del experimento:

```bash
python src/train.py \
  --run-name run01_baseline \
  --C 1.0 \
  --solver lbfgs \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

Resultados obtenidos:

```text
accuracy  = 0.9528
precision = 0.8522
recall    = 0.6203
f1_score  = 0.7179
roc_auc   = 0.9584
```

---

## 11. Corridas experimentales

Todas las corridas pertenecen al experimento:

```text
TC5061_Bank_Personal_Loan_Experiments
```

### Corrida 1 - Baseline

```bash
python src/train.py \
  --run-name run01_baseline \
  --C 1.0 \
  --solver lbfgs \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

### Corrida 2 - Mayor regularización

```bash
python src/train.py \
  --run-name run02_C_0_1 \
  --C 0.1 \
  --solver lbfgs \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

Resultados obtenidos:

```text
accuracy  = 0.9498
precision = 0.8519
recall    = 0.5823
f1_score  = 0.6917
roc_auc   = 0.9591
```

### Corrida 3 - Menor regularización

```bash
python src/train.py \
  --run-name run03_C_10 \
  --C 10.0 \
  --solver lbfgs \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

Resultados obtenidos:

```text
accuracy  = 0.9522
precision = 0.8333
recall    = 0.6329
f1_score  = 0.7194
roc_auc   = 0.9582
```

### Corrida 4 - Balanceo de clases

```bash
python src/train.py \
  --run-name run04_balanced \
  --C 1.0 \
  --solver lbfgs \
  --class-weight balanced \
  --max-iter 1000 \
  --random-state 4284
```

Resultados obtenidos:

```text
accuracy  = 0.8953
precision = 0.4781
recall    = 0.8987
f1_score  = 0.6242
roc_auc   = 0.9592
```

### Corrida 5 - Solver alternativo

```bash
python src/train.py \
  --run-name run05_liblinear \
  --C 1.0 \
  --solver liblinear \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

Resultados obtenidos:

```text
accuracy  = 0.9528
precision = 0.8522
recall    = 0.6203
f1_score  = 0.7179
roc_auc   = 0.9584
```

---

## 12. Resumen comparativo de las cinco corridas

| Corrida | C | Solver | Class weight | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `run01_baseline` | 1.0 | lbfgs | None | 0.9528 | 0.8522 | 0.6203 | 0.7179 | 0.9584 |
| `run02_C_0_1` | 0.1 | lbfgs | None | 0.9498 | 0.8519 | 0.5823 | 0.6917 | 0.9591 |
| `run03_C_10` | 10.0 | lbfgs | None | 0.9522 | 0.8333 | 0.6329 | 0.7194 | 0.9582 |
| `run04_balanced` | 1.0 | lbfgs | balanced | 0.8953 | 0.4781 | 0.8987 | 0.6242 | 0.9592 |
| `run05_liblinear` | 1.0 | liblinear | None | 0.9528 | 0.8522 | 0.6203 | 0.7179 | 0.9584 |

---

## 13. Comparación en la interfaz de MLflow

1. Abrir:

```text
http://127.0.0.1:5000
```

2. Seleccionar el experimento:

```text
TC5061_Bank_Personal_Loan_Experiments
```

3. Seleccionar:

```text
run01_baseline
run02_C_0_1
run03_C_10
run04_balanced
run05_liblinear
```

4. Utilizar la opción **Compare**.

La vista permite contrastar parámetros como:

```text
C
class_weight
solver
```

y métricas como:

```text
accuracy
precision
recall
f1_score
roc_auc
```

---

## 14. Artefactos registrados en MLflow

Cada corrida registra artefactos de evaluación dentro de:

```text
evaluation/
```

Artefactos principales:

```text
classification_report.json
confusion_matrix.png
metrics.json
roc_curve.png
```

También se registra el modelo entrenado mediante MLflow.

---

## 15. Métricas registradas

```text
accuracy
precision
recall
f1_score
roc_auc
```

---

## 16. Parámetros registrados

```text
model_type
preprocessing
C
solver
class_weight
max_iter
random_state
test_size
stratify
n_features
rows_loaded
negative_experience_removed
duplicate_rows_removed
rows_final
```

---

## 17. Prueba de reproducibilidad

Se realizó una corrida adicional con la misma configuración del baseline:

```bash
python src/train.py \
  --run-name run06_reproducibility \
  --C 1.0 \
  --solver lbfgs \
  --class-weight none \
  --max-iter 1000 \
  --random-state 4284
```

Resultados:

```text
accuracy  = 0.9528
precision = 0.8522
recall    = 0.6203
f1_score  = 0.7179
roc_auc   = 0.9584
```

Los valores coinciden con `run01_baseline`, confirmando la reproducibilidad cuando se usan los mismos argumentos y la misma semilla.

---

## 18. Interpretación general de los resultados

Tomando `F1-score` como métrica de referencia para comparar el balance entre precision y recall, la corrida `run03_C_10` obtuvo el valor más alto:

```text
F1-score = 0.7194
```

Configuración:

```text
C = 10.0
solver = lbfgs
class_weight = None
max_iter = 1000
random_state = 4284
```

El hiperparámetro con el mayor impacto observable fue `class_weight`.

Con `class_weight=balanced`, el recall aumentó:

```text
0.6203 -> 0.8987
```

pero la precision disminuyó:

```text
0.8522 -> 0.4781
```

Esto muestra el trade-off entre detectar una mayor proporción de clientes de la clase positiva y aumentar el número de falsos positivos.

El cambio de solver entre `lbfgs` y `liblinear` no produjo cambios apreciables en las métricas para esta configuración y partición de datos.

---

## 19. Control de versiones con Git

El desarrollo del proyecto se realizó utilizando la rama:

```text
feature/mlflow-experimentation-martinezjose
```

Flujo utilizado:

```text
main
  ↑
Pull Request
  ↑
feature/mlflow-experimentation-martinezjose
```

Los cambios desarrollados en la rama feature fueron integrados a `main` mediante un Pull Request en GitHub.

---

## 20. Flujo general del proyecto

```text
Bank_Personal_Loan_Modelling.csv
              │
              ▼
         dataset.py
              │
       carga + limpieza
              │
              ▼
         features.py
              │
 selección + StandardScaler
              │
              ▼
           train.py
              │
      train_test_split
              │
              ▼
        LogisticRegression
              │
              ▼
       Scikit-learn Pipeline
              │
              ▼
           evaluación
              │
        ┌─────┼─────┐
        │     │     │
     métricas │ artefactos
              │
              ▼
            MLflow
              │
        ┌─────┼─────┐
        │     │     │
     params metrics model
```

---

## 21. Resultado

El proyecto transforma el notebook baseline en un flujo de experimentación modular, parametrizable y reproducible.

La solución integra:

```text
Cookiecutter-inspired project structure
+
Feature Engineering
+
Scikit-learn Pipeline
+
Command Line Arguments
+
MLflow Tracking
+
Git Feature Branch
+
Pull Request
+
Reproducibility
```

De esta forma, el experimento puede ejecutarse nuevamente utilizando el código fuente, los parámetros registrados y las versiones de dependencias documentadas en el repositorio.
