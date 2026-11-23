"""Pipeline de drift de evento, es decir, del target `Fraude` (actividad 3).

Compara la tasa de fraude del lote que puntuó la inferencia con la del entrenamiento (test chi-cuadrado).
Rechaza los lotes sin etiquetas con un mensaje claro: en producción las etiquetas reales llegan con retraso,
y sin ellas no hay drift de evento. Emite una ALERTA si el p-valor queda por debajo de `monitoreo.alfa_evento`.

Ejecución (desde la raíz del repositorio, después de la inferencia con el mismo nivel_drift):
    poetry run python -m src.pipelines.drift_evento_flow
"""
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.data.load_data import cargar_datos, leer_config
from src.monitoring.monitoreo import dividir, drift_de_target, ruta_lote, validar_lote


@task(cache_policy=NO_CACHE)
def referencia(cfg):
    """Variable objetivo de los datos de entrenamiento."""
    objetivo = cfg["variables"]["objetivo"]
    entrenamiento, _ = dividir(cargar_datos("train", cfg), cfg["division"], objetivo)
    return entrenamiento[objetivo]


@task(cache_policy=NO_CACHE)
def evaluar_evento(referencia_objetivo, cfg):
    """Compara la tasa de fraude del lote con la referencia. Falla si el lote no trae etiquetas."""
    objetivo = cfg["variables"]["objetivo"]
    lote = pd.read_csv(ruta_lote(cfg["monitoreo"]["carpeta_lotes"], cfg["inferencia"]["nivel_drift"]))
    validar_lote(lote, objetivo)
    resultado = drift_de_target(referencia_objetivo, lote[objetivo], cfg["monitoreo"]["alfa_evento"])
    print(
        f"Tasa de fraude: entrenamiento {resultado['tasa_referencia']:.1%}, lote {resultado['tasa_lote']:.1%} "
        f"(p-valor {resultado['p_valor']:.4g})"
    )
    return resultado


@task
def alerta(resultado, alfa):
    """Emite una ALERTA en los logs si el cambio en la tasa de fraude es significativo."""
    if resultado["drift"]:
        print(f"ALERTA: drift del target Fraude (p-valor {resultado['p_valor']:.4g} < {alfa})")
        return True
    print(f"Sin alerta: tasa de fraude estable (p-valor {resultado['p_valor']:.4g} >= {alfa})")
    return False


@flow(name="drift-evento", log_prints=True)
def drift_evento_flow():
    cfg = leer_config()
    print(f"Nivel de drift del lote: {cfg['inferencia']['nivel_drift']}")
    resultado = evaluar_evento(referencia(cfg), cfg)
    return alerta(resultado, cfg["monitoreo"]["alfa_evento"])


if __name__ == "__main__":
    drift_evento_flow()