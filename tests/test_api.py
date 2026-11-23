"""Pruebas de la API con un modelo falso (mock): no necesitan MLflow levantado."""
import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient

CLIENTE = {
    "Num_Tarjetas": 1,
    "Balance_Tj": 21937,
    "Total_Transac": 18,
    "Trans_Int": 0,
    "Estado": "Maryland",
    "Genero": "Female",
    "Tipo_Tarjeta": "American Express",
}


class ModeloFalso:
    feature_names_in_ = np.array(list(CLIENTE))

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    # main.py carga el champion desde MLflow al importarse: se reemplazan esas dos llamadas
    # por versiones falsas y después se importa la API desde cero.
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: SimpleNamespace(get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")),
    )
    sys.modules.pop("src.api.main", None)
    import src.api.main as main

    return TestClient(main.app)


def test_health_informa_la_version(client):
    assert client.get("/health").json() == {"estado": "ok", "modelo": "fraude-modelo", "version": "7"}


def test_predict_devuelve_probabilidad_prediccion_y_version(client):
    respuesta = client.post("/predict", json=CLIENTE)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"probabilidad_fraude": 0.8, "prediccion": 1, "version_modelo": "7"}


def test_predict_rechaza_datos_invalidos(client):
    malo = {**CLIENTE, "Balance_Tj": "mucho"}
    assert client.post("/predict", json=malo).status_code == 422


def test_predict_rechaza_campos_faltantes(client):
    incompleto = {k: v for k, v in CLIENTE.items() if k != "Estado"}
    assert client.post("/predict", json=incompleto).status_code == 422