"""Pipeline de inferencia: cargar champion → lote con drift emulado → predicción → guardar → registrar.

Carga el modelo por su alias `champion`. El lote se toma de la parte de prueba de train.csv (la misma
división del entrenamiento), porque trae las etiquetas reales que necesita el drift de evento; test.csv
de Kaggle no las tiene. Al lote se le aplica el drift emulado de `inferencia.nivel_drift` (actividad 3)
y se guarda una copia en `monitoreo.carpeta_lotes` para los flows de drift.

Requisitos: servicios levantados, un champion registrado y las variables PREFECT_API_URL y MLFLOW_TRACKING_URI.
Ejecución (desde la raíz del repositorio):
    poetry run python -m src.pipelines.inferencia_flow
"""
import os
from datetime import date
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.data.load_data import cargar_datos, leer_config
from src.models.modelo import predecir
from src.monitoring.monitoreo import aplicar_drift, dividir, ruta_lote


@task(cache_policy=NO_CACHE)
def cargar_champion(cfg):
    """Carga el modelo con alias champion. Devuelve (modelo, versión, run de entrenamiento)."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Champion cargado: {nombre} v{version.version}")
    return modelo, version.version, version.run_id


@task(cache_policy=NO_CACHE)
def ingesta_lote(cfg):
    """Toma un lote de la parte de prueba, le aplica el drift emulado y guarda una copia para el monitoreo."""
    variables, inferencia = cfg["variables"], cfg["inferencia"]
    _, prueba = dividir(cargar_datos("train", cfg), cfg["division"], variables["objetivo"])
    base = prueba.sample(n=inferencia["tamano_lote"], random_state=inferencia["semilla_lote"])
    lote = aplicar_drift(
        base,
        inferencia["nivel_drift"],
        variables["numericas"],
        variables["objetivo"],
        cfg["monitoreo"]["subida_tasa_maxima"],
        semilla=inferencia["semilla_lote"],
    )

    ruta = ruta_lote(cfg["monitoreo"]["carpeta_lotes"], inferencia["nivel_drift"])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lote.to_csv(ruta, index=False)
    print(
        f"Lote con nivel de drift {inferencia['nivel_drift']} guardado en {ruta} "
        f"({len(lote)} filas, fraude real {lote[variables['objetivo']].mean():.1%})"
    )
    return lote


@task(cache_policy=NO_CACHE)
def predecir_y_guardar(modelo, lote, cfg, hoy):
    """Calcula la probabilidad de fraude y guarda el resultado en un CSV con la fecha y el nivel de drift."""
    resultado = predecir(modelo, lote, cfg["metricas"]["umbral"])
    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{hoy.isoformat()}_nivel_{cfg['inferencia']['nivel_drift']:.1f}.csv"
    resultado.to_csv(ruta, index=False)
    fraudes = int(resultado["prediccion"].sum())
    print(f"Predicciones guardadas en {ruta} | fraudes predichos: {fraudes} de {len(resultado)}")
    # Solo valores simples, para que Prefect los pase entre tareas sin problemas.
    return str(ruta), len(resultado), float(resultado["probabilidad_fraude"].mean()), fraudes


@task
def registrar_inferencia(version, run_id_entrenamiento, nivel_drift, ruta, filas, proba_media, fraudes, hoy):
    """Run de MLflow con tag fase=inferencia, enlazado a la versión del modelo y con el nivel de drift."""
    with mlflow.start_run(run_name=f"inferencia-{hoy.isoformat()}-nivel-{nivel_drift:.1f}"):
        mlflow.set_tags(
            {
                "fase": "inferencia",
                "modelo_version": version,
                "modelo_run_id": run_id_entrenamiento,
                "nivel_drift": nivel_drift,
            }
        )
        mlflow.log_metrics({"filas_predichas": filas, "probabilidad_media": proba_media, "fraudes_predichos": fraudes})
        mlflow.log_artifact(ruta)
    print(f"Run de inferencia registrado (modelo v{version}, nivel de drift {nivel_drift})")


@flow(name="inferencia-fraude", log_prints=True)
def inferencia_flow():
    cfg = leer_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento_modelo"])
    hoy = date.today()

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta_lote(cfg)
    ruta, filas, proba_media, fraudes = predecir_y_guardar(modelo, lote, cfg, hoy)
    registrar_inferencia(
        version, run_id_entrenamiento, cfg["inferencia"]["nivel_drift"], ruta, filas, proba_media, fraudes, hoy
    )


if __name__ == "__main__":
    inferencia_flow()