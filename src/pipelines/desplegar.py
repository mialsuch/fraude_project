"""Despliegues en Prefect de las dos pipelines (actividad 2).

- entrenamiento-manual: sin horario; se lanza a mano desde la UI de Prefect.
- inferencia-diaria: todos los días a las 06:00, hora de Colombia.

Ejecución (desde la raíz del repositorio, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python -m src.pipelines.desplegar

Deja la terminal abierta: mientras `serve` esté activo, Prefect puede lanzar las ejecuciones.
"""
import os

# Las ejecuciones de `serve` corren en un subproceso. En Windows su salida usa cp1252, que no admite
# los emojis que imprime MLflow (🏃) y la tarea falla con UnicodeEncodeError. Se fuerza UTF-8.
os.environ.setdefault("PYTHONUTF8", "1")

from prefect import serve  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from src.pipelines.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from src.pipelines.inferencia_flow import inferencia_flow  # noqa: E402

if __name__ == "__main__":
    entrenamiento = entrenamiento_flow.to_deployment(name="entrenamiento-manual")
    inferencia = inferencia_flow.to_deployment(
        name="inferencia-diaria",
        schedules=[Cron("0 6 * * *", timezone="America/Bogota")],  # minuto 0, hora 6, todos los días
    )
    serve(entrenamiento, inferencia)