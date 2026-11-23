"""Pipeline de drift de features (actividad 3).

Compara las variables de entrada del lote que puntuó la inferencia con los datos de entrenamiento usando
Evidently (DataDriftPreset). Guarda un reporte HTML por lote y emite una ALERTA si la fracción de variables
con drift supera `monitoreo.umbral_drift`. No necesita MLflow.

Ejecución (desde la raíz del repositorio, después de la inferencia con el mismo nivel_drift):
    poetry run python -m src.pipelines.drift_features_flow
"""
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.data.load_data import cargar_datos, leer_config
from src.monitoring.monitoreo import dividir, fraccion_desde_evidently, hay_alerta, ruta_lote


def columnas_modelo(cfg):
    return cfg["variables"]["numericas"] + cfg["variables"]["categoricas"]


@task(cache_policy=NO_CACHE)
def referencia(cfg):
    """Muestra de los datos de entrenamiento: la referencia contra la que se compara cada lote."""
    entrenamiento, _ = dividir(cargar_datos("train", cfg), cfg["division"], cfg["variables"]["objetivo"])
    muestra = entrenamiento.sample(n=cfg["monitoreo"]["tamano_referencia"], random_state=cfg["division"]["random_state"])
    print(f"Referencia: {len(muestra)} filas de entrenamiento")
    return muestra[columnas_modelo(cfg)]


@task(cache_policy=NO_CACHE)
def calcular_drift(datos_referencia, cfg):
    """Ejecuta el preset de drift de Evidently, guarda el reporte HTML y devuelve la fracción de variables con drift."""
    nivel = cfg["inferencia"]["nivel_drift"]
    ruta = ruta_lote(cfg["monitoreo"]["carpeta_lotes"], nivel)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}: ejecuta antes la inferencia con nivel_drift = {nivel}")
    lote = pd.read_csv(ruta)

    resultado = Report([DataDriftPreset()]).run(
        reference_data=datos_referencia, current_data=lote[columnas_modelo(cfg)]
    )
    carpeta = Path(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html = carpeta / f"drift_features_{ruta.stem}.html"
    resultado.save_html(str(ruta_html))
    print(f"Reporte de Evidently guardado en {ruta_html}")
    return fraccion_desde_evidently(resultado.dict())


@task
def alerta(fraccion, umbral):
    """Emite una ALERTA en los logs si la fracción de variables con drift supera el umbral."""
    if hay_alerta(fraccion, umbral):
        print(f"ALERTA: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
        return True
    print(f"Sin alerta: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    return False


@flow(name="drift-features", log_prints=True)
def drift_features_flow():
    cfg = leer_config()
    print(f"Nivel de drift del lote: {cfg['inferencia']['nivel_drift']}")
    fraccion = calcular_drift(referencia(cfg), cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"])


if __name__ == "__main__":
    drift_features_flow()