"""Pipeline de inferencia diaria: cargar champion → ingesta del lote → predicción → guardar → registrar (actividad 2).

Carga el modelo por su alias `champion` (no por un número de versión), así que cuando el
entrenamiento promueve un nuevo champion, la inferencia lo usa sin cambiar código.
El lote simula clientes nuevos con una muestra de test.csv (clientes sin etiqueta de fraude).

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


@task(cache_policy=NO_CACHE)
def cargar_champion(cfg):
    """Carga el modelo con alias champion. Devuelve (modelo, versión, run de entrenamiento)."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Champion cargado: {nombre} v{version.version}")
    return modelo, version.version, version.run_id


#Ingesta

@task(cache_policy=NO_CACHE)
def ingesta_lote(cfg, hoy):
    """Toma `tamano_lote` clientes de test.csv. La semilla es la fecha: el lote de un día es reproducible."""
    df = cargar_datos("test", cfg)
    lote = df.sample(n=cfg["inferencia"]["tamano_lote"], random_state=int(hoy.strftime("%Y%m%d")))
    print(f"Lote de {len(lote)} clientes tomado de test.csv")
    return lote


#Predicción
@task(cache_policy=NO_CACHE)
def predecir_y_guardar(modelo, lote, cfg, hoy):
    """Calcula la probabilidad de fraude y guarda el resultado en un CSV con la fecha de ejecución."""
    proba = modelo.predict_proba(lote[list(modelo.feature_names_in_)])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_fraude"] = proba
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{hoy.isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    fraudes = int(resultado["prediccion"].sum())
    print(f"Predicciones guardadas en {ruta} | fraudes predichos: {fraudes} de {len(resultado)}")
    # Solo valores simples, para que Prefect los pase entre tareas sin problemas.
    return str(ruta), len(resultado), float(proba.mean()), fraudes


#Registro
@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media, fraudes, hoy):
    """Run de MLflow con tag fase=inferencia, enlazado a la versión del modelo y a su run de entrenamiento."""
    with mlflow.start_run(run_name=f"inferencia-{hoy.isoformat()}"):
        mlflow.set_tags({"fase": "inferencia", "modelo_version": version, "modelo_run_id": run_id_entrenamiento})
        mlflow.log_metrics({"filas_predichas": filas, "probabilidad_media": proba_media, "fraudes_predichos": fraudes})
        mlflow.log_artifact(ruta)
    print(f"Run de inferencia registrado (modelo v{version})")


@flow(name="inferencia-fraude", log_prints=True)
def inferencia_flow():
    cfg = leer_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento_modelo"])
    hoy = date.today()

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta_lote(cfg, hoy)
    ruta, filas, proba_media, fraudes = predecir_y_guardar(modelo, lote, cfg, hoy)
    registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media, fraudes, hoy)


if __name__ == "__main__":
    inferencia_flow()
    
