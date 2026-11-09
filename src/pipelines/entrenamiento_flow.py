"""Pipeline de entrenamiento: ingesta → entrenamiento → evaluación → registro (actividad 2).

Entrena cada candidato de config.yaml con las variables elegidas en features.yaml, registra cada
uno como un run de MLflow con el tag `fase = entrenamiento` y promueve el mejor a `champion` solo
si su AUC supera `registro.umbral_auc` y el AUC del champion vigente. Si no, el alias no cambia.

La lógica del modelo está en src/models/modelo.py; este archivo solo orquesta y registra.

Requisitos: servicios levantados y las variables PREFECT_API_URL y MLFLOW_TRACKING_URI.
Ejecución (desde la raíz del repositorio):
    poetry run python -m src.pipelines.entrenamiento_flow
"""
import os

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from sklearn.model_selection import train_test_split

from src.data.load_data import cargar_datos, leer_config
from src.models.modelo import calcular_metricas, construir_modelo, resolver_features

RUTA_FEATURES = "configs/features.yaml"


#Ingesta
# NO_CACHE: estas tareas reciben DataFrames y modelos grandes; no tiene sentido calcular su hash para cachear.
@task(cache_policy=NO_CACHE)
def ingesta(cfg, features):
    """Carga train.csv, elige las variables y lo divide en entrenamiento y prueba (estratificado)."""
    df = cargar_datos("train", cfg)
    numericas, categoricas = resolver_features(cfg["variables"], features)
    columnas = numericas + categoricas
    objetivo = cfg["variables"]["objetivo"]
    division = cfg["division"]
    train, test = train_test_split(
        df, test_size=division["test_size"], random_state=division["random_state"], stratify=df[objetivo]
    )
    print(f"Ingesta: {len(df)} filas | features={features['seleccion']} ({len(columnas)}): {columnas}")
    print(f"Entrenamiento: {len(train)} filas | prueba: {len(test)} filas | fraude: {df[objetivo].mean():.2%}")
    return {
        "X_train": train[columnas],
        "y_train": train[objetivo],
        "X_test": test[columnas],
        "y_test": test[objetivo],
        "numericas": numericas,
        "categoricas": categoricas,
    }
    
#Entrenamiento y evaluación
@task(cache_policy=NO_CACHE)
def entrenar(candidato, datos):
    """Construye el pipeline (preprocesado + estimador) del candidato y lo entrena."""
    modelo = construir_modelo(candidato["tipo"], candidato["parametros"], datos["numericas"], datos["categoricas"])
    modelo.fit(datos["X_train"], datos["y_train"])
    print(f"{candidato['nombre_run']} entrenado ({candidato['tipo']})")
    return modelo


@task(cache_policy=NO_CACHE)
def evaluar(modelo, datos, cfg):
    """Calcula las métricas sobre el conjunto de prueba."""
    proba = modelo.predict_proba(datos["X_test"])[:, 1]
    metricas = calcular_metricas(datos["y_test"], proba, cfg["metricas"]["umbral"])
    print("Métricas: " + ", ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return metricas


#Registro mlflow
@task(cache_policy=NO_CACHE)
def registrar(modelo, metricas, datos, candidato, features, cfg):
    """Crea el run de MLflow con tag fase=entrenamiento, parámetros, métricas y el modelo. Devuelve el model_uri."""
    columnas = datos["numericas"] + datos["categoricas"]
    with mlflow.start_run(run_name=f"{candidato['nombre_run']}-{features['seleccion'].lower()}"):
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params(
            {
                "modelo": candidato["tipo"],
                **candidato["parametros"],
                "features": features["seleccion"],
                "n_features": len(columnas),
                "semilla": cfg["division"]["random_state"],
                "umbral": cfg["metricas"]["umbral"],
            }
        )
        mlflow.log_metrics(metricas)
        mlflow.log_artifact("configs/config.yaml")
        mlflow.log_artifact(RUTA_FEATURES)
        ejemplo = datos["X_test"].head(3)
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict_proba(ejemplo)[:, 1]),
            input_example=ejemplo,
            # skops (el formato de guardado) bloquea por seguridad los árboles de HistGradientBoosting.
            # Son modelos entrenados aquí mismo, así que se confía solo en ese tipo concreto.
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        )
    print(f"Run registrado en MLflow: {info.model_uri}")
    return info.model_uri


#Champion
@task
def promover_si_mejora(model_uri, auc, cfg):
    """Registra una versión nueva y le asigna `champion` solo si supera el AUC mínimo y al champion vigente."""
    client = MlflowClient()
    nombre, minimo = cfg["registro"]["nombre_modelo"], cfg["registro"]["umbral_auc"]
    if auc < minimo:
        print(f"AUC {auc:.4f} < mínimo {minimo}: no se promueve")
        return None
    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
        auc_champion = client.get_run(champion.run_id).data.metrics["auc"]
    except MlflowException:
        champion, auc_champion = None, None  # todavía no hay champion
    if champion is not None and auc <= auc_champion:
        print(f"AUC {auc:.4f} no supera al champion v{champion.version} ({auc_champion:.4f}): el alias no cambia")
        return None

    version = mlflow.register_model(model_uri, nombre)
    client.set_registered_model_alias(nombre, "champion", version.version)
    client.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    anterior = f"v{champion.version} ({auc_champion:.4f})" if champion else "ninguno"
    print(f"Nuevo champion: {nombre} v{version.version} (AUC {auc:.4f}); anterior: {anterior}")
    return version.version


#flow
@flow(name="entrenamiento-fraude", log_prints=True)
def entrenamiento_flow():
    cfg = leer_config()
    with open(RUTA_FEATURES, encoding="utf-8") as f:
        features = yaml.safe_load(f)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento_modelo"])

    datos = ingesta(cfg, features)
    resultados = []
    for candidato in cfg["candidatos"]:
        modelo = entrenar(candidato, datos)
        metricas = evaluar(modelo, datos, cfg)
        model_uri = registrar(modelo, metricas, datos, candidato, features, cfg)
        resultados.append((model_uri, metricas["auc"], candidato["nombre_run"]))

    # Solo el candidato con mayor AUC compite por el alias champion.
    mejor_uri, mejor_auc, mejor_nombre = max(resultados, key=lambda r: r[1])
    print(f"Mejor candidato: {mejor_nombre} (AUC {mejor_auc:.4f})")
    return promover_si_mejora(mejor_uri, mejor_auc, cfg)


if __name__ == "__main__":
    entrenamiento_flow()
