"""Pruebas de src/models/modelo.py. No necesitan MLflow, Prefect ni los datos de Kaggle."""
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import leer_config
from src.models.modelo import calcular_metricas, construir_modelo, resolver_features

VARIABLES = leer_config()["variables"]


@pytest.mark.parametrize(
    "features, esperadas",
    [
        ({"seleccion": "ALL"}, 7),
        ({"seleccion": "NUMERIC"}, 4),
        ({"seleccion": "CUSTOM", "custom": ["Balance_Tj", "Tipo_Tarjeta"]}, 2),
    ],
)
def test_resolver_features(features, esperadas):
    numericas, categoricas = resolver_features(VARIABLES, features)
    assert len(numericas + categoricas) == esperadas


def test_custom_con_variable_desconocida_falla():
    with pytest.raises(ValueError, match="desconocidas"):
        resolver_features(VARIABLES, {"seleccion": "CUSTOM", "custom": ["Edad"]})


def test_seleccion_invalida_falla():
    with pytest.raises(ValueError, match="no válida"):
        resolver_features(VARIABLES, {"seleccion": "TODAS"})


@pytest.mark.parametrize("tipo", ["logistic_regression", "hist_gradient_boosting"])
def test_construir_y_entrenar_modelo(tipo):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"Balance_Tj": rng.integers(0, 20000, 60), "Tipo_Tarjeta": rng.choice(["Visa", "Amex"], 60)})
    y = (X["Balance_Tj"] > 10000).astype(int)
    modelo = construir_modelo(tipo, {}, ["Balance_Tj"], ["Tipo_Tarjeta"]).fit(X, y)
    assert modelo.predict_proba(X).shape == (60, 2)


def test_metricas_de_un_modelo_perfecto():
    metricas = calcular_metricas(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), umbral=0.5)
    assert metricas["auc"] == 1.0 and metricas["recall"] == 1.0