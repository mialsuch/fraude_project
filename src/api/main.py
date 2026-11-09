"""API de inferencia: sirve el modelo con alias `champion` de MLflow (actividad 2).

Ejecución (desde la raíz del repositorio, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn src.api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs

El modelo se carga al arrancar. Si el entrenamiento promueve un nuevo champion, reinicia la API.
"""
import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.data.load_data import leer_config

cfg = leer_config()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
VERSION = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version

app = FastAPI(title="API de fraude", description="Predicción de fraude bancario con el modelo champion de MLflow.")


class Cliente(BaseModel):
    """Datos de un cliente. Los nombres coinciden con las columnas del CSV de la competencia."""

    Num_Tarjetas: int = Field(examples=[1])
    Balance_Tj: float = Field(examples=[21937])
    Total_Transac: int = Field(examples=[18])
    Trans_Int: int = Field(examples=[0])
    Estado: str = Field(examples=["Maryland"])
    Genero: str = Field(examples=["Female"])
    Tipo_Tarjeta: str = Field(examples=["American Express"])


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}


@app.post("/predict")
def predict(cliente: Cliente):
    # Solo las columnas con las que se entrenó el champion (dependen de features.yaml).
    datos = pd.DataFrame([cliente.model_dump()])[list(modelo.feature_names_in_)]
    proba = float(modelo.predict_proba(datos)[0, 1])
    return {"probabilidad_fraude": round(proba, 4), "prediccion": int(proba >= UMBRAL), "version_modelo": VERSION}