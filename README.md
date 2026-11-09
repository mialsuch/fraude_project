# Fraude Project – Detección de fraude bancario con MLOps

Proyecto del curso *MLOps en Python – De la práctica a la producción*.
El objetivo es predecir si un cliente de una entidad bancaria comete **fraude** (`Fraude = 1`) o no,
con un flujo reproducible: entorno fijado con **Poetry**, orquestación con **Prefect** y trazabilidad con **MLflow**.

**Integrantes:** Miguel Suárez · Francisco Álvarez

## Caso de uso y datos

- **Fuente:** competencia de Kaggle [Modelos de Fraude 2026](https://www.kaggle.com/competitions/modelos-de-fraude-2026/data) (acceso por invitación).
- **Archivos:** `train.csv` (848 548 clientes, 9 columnas) y `test.csv` (200 027 clientes, sin la columna objetivo). Separados por `;`.
- **Objetivo:** `Fraude` (1 = fraude, 0 = no fraude). Clases desbalanceadas: **5,99 %** de fraude.
- **Métrica:** AUC-ROC, la métrica oficial de la competencia y adecuada para clases desbalanceadas.

| Campo | Descripción |
|---|---|
| `ID` | Cédula del cliente (identificador, no se usa para predecir) |
| `Num_Tarjetas` | Número de tarjetas del cliente |
| `Balance_Tj` | Balance de las tarjetas de crédito |
| `Total_Transac` | Total de transacciones a la fecha |
| `Trans_Int` | Transacciones internacionales a la fecha |
| `Estado` | Lugar de residencia |
| `Genero` | Género |
| `Tipo_Tarjeta` | Empresa con la que adquirió sus productos financieros |
| `Fraude` | Variable objetivo |

> **Los datos no se versionan en Git:** están sujetos a las reglas de la competencia, contienen cédulas y pesan ~50 MB.
> Descárgalos desde Kaggle (**Download All**) y copia `train.csv` y `test.csv` en la carpeta `data/`.

## Estructura

```
fraude_project/
├── configs/config.yaml     # rutas, separador, variables, división y MLflow (sin valores en el código)
├── data/                   # train.csv y test.csv (ignorados por Git)
├── docs/evidencias/        # capturas de MLflow y Prefect
├── flows/
│   ├── prueba_flow.py      # valida que Prefect y MLflow responden
│   └── datos_flow.py       # carga y valida los datos; registra su tamaño en Prefect y MLflow
├── services/               # cómo levantar MLflow y Prefect (Docker o sin Docker)
├── src/
│   ├── data/load_data.py   # carga y validación de los CSV
│   ├── models/             # (próximas actividades)
│   └── pipelines/          # (próximas actividades)
├── tests/                  # pruebas con pytest
├── pyproject.toml          # dependencias declaradas con versión fijada
└── poetry.lock             # versiones exactas resueltas (no editar a mano)
```

## 1. Requisitos

| Herramienta | Versión | Comprobación |
|---|---|---|
| Python | 3.12 a 3.14 | `py -3.12 --version` |
| Poetry | 2.x | `poetry --version` |
| Git | reciente | `git --version` |
| Docker Desktop (opcional) | reciente | `docker --version` |

## 2. Instalar el entorno

```powershell
git clone https://github.com/mialsuch/fraude_project.git
cd fraude_project
poetry env use 3.12
poetry install
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
```

`pyproject.toml` declara las dependencias y `poetry.lock` fija sus versiones exactas; se versionan ambos.
Si cambias dependencias, usa `poetry add` (nunca edites el lock a mano) y sube los dos archivos.

## 3. Levantar MLflow y Prefect

Ver [services/README.md](services/README.md). Sin Docker, en dos terminales desde `services/`:

```powershell
poetry run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --artifacts-destination mlflow_store/artifacts --serve-artifacts
```

```powershell
$env:PREFECT_HOME = "$PWD\prefect_home"
poetry run prefect server start --host 127.0.0.1 --port 4200
```

UIs: MLflow en http://localhost:5000 · Prefect en http://localhost:4200

## 4. Ejecutar los flows

Desde la raíz del repositorio, en una tercera terminal:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/prueba_flow.py      # run "prueba" + experimento "prueba-validacion"
poetry run python -m src.data.load_data     # valida los CSV y muestra su tamaño
poetry run python -m flows.datos_flow       # run "datos-fraude" + experimento "fraude-datos"
```

Sin `PREFECT_API_URL`, Prefect crea un servidor temporal y el run no aparece en la UI.

## Evidencias

Capturas en [docs/evidencias](docs/evidencias): UIs de MLflow y Prefect, run `prueba` (Completed, `Suma de 1 a 5 = 15`),
experimento `prueba-validacion` (`n = 5`, `suma = 15`), run `datos-fraude` y experimento `fraude-datos`
(`filas_train = 848548`, `filas_test = 200027`, `proporcion_fraude = 0.0599`).