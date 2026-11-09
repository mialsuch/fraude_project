"""Flow de datos: carga y valida los CSV de la competencia y registra su tamaño en Prefect y MLflow.

Pasos: cargar train → cargar test → registrar en MLflow → resumen (artefacto de Prefect)

Requisitos: servicios levantados (ver services/README.md) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz del repositorio):
    poetry run python -m flows.datos_flow
"""
import os

import mlflow
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from src.data.load_data import RUTA_CONFIG, cargar_datos, leer_config

#Carga
@task
def cargar(conjunto, cfg):
    """Carga y valida un conjunto ("train" o "test") con src/data/load_data.py."""
    df = cargar_datos(conjunto, cfg)
    print(f"{conjunto}: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

#Registro mlflow
@task
def registrar_en_mlflow(train, test, cfg):
    """Registra el tamaño y la proporción de fraude del dataset en MLflow. Devuelve el ID del run."""
    objetivo = cfg["variables"]["objetivo"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name="carga-datos") as run:
        mlflow.log_params(
            {
                "fuente": cfg["datos"]["fuente"],
                "separador": cfg["datos"]["separador"],
                "columnas_train": train.shape[1],
                "columnas_test": test.shape[1],
            }
        )
        mlflow.log_metrics(
            {
                "filas_train": train.shape[0],
                "filas_test": test.shape[0],
                "proporcion_fraude": float(train[objetivo].mean()),
            }
        )
        mlflow.log_artifact(RUTA_CONFIG)
        return run.info.run_id
    
#flow   
@flow(name="datos-fraude", log_prints=True)
def datos_fraude():
    cfg = leer_config()
    train = cargar("train", cfg)
    test = cargar("test", cfg)
    run_id = registrar_en_mlflow(train, test, cfg)

    proporcion = train[cfg["variables"]["objetivo"]].mean()
    create_markdown_artifact(
        key="datos-fraude",
        description="Resumen del dataset de fraude",
        markdown=(
            "## Dataset Modelos de Fraude 2026\n\n"
            f"- Train: {train.shape[0]} filas, {train.shape[1]} columnas\n"
            f"- Test: {test.shape[0]} filas, {test.shape[1]} columnas\n"
            f"- Proporción de fraude: **{proporcion:.2%}**\n"
            f"- Run de MLflow: `{run_id}`"
        ),
    )
    print(f"Proporción de fraude: {proporcion:.2%} | Run de MLflow: {run_id}")


if __name__ == "__main__":
    datos_fraude()