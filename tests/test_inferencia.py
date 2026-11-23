"""Pruebas de la lógica de inferencia: predicción con un modelo falso y lote con drift. Sin MLflow ni Prefect."""
import numpy as np
import pandas as pd
import pytest

from src.models.modelo import predecir
from src.monitoring.monitoreo import aplicar_drift


class ModeloFalso:
    """Imita un modelo de sklearn: probabilidad de fraude = 0.9 si Balance_Tj > 10 000, si no 0.1."""

    feature_names_in_ = np.array(["Balance_Tj", "Trans_Int"])

    def predict_proba(self, X):
        p = np.where(X["Balance_Tj"] > 10000, 0.9, 0.1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def lote():
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Balance_Tj": [0, 5000, 20000],
            "Trans_Int": [0, 3, 10],
            "Estado": ["Ohio", "Texas", "Guam"],
            "Fraude": [0, 0, 1],
        }
    )


def test_predecir_agrega_probabilidad_y_prediccion(lote):
    resultado = predecir(ModeloFalso(), lote, umbral=0.5)
    assert resultado["probabilidad_fraude"].tolist() == [0.1, 0.1, 0.9]
    assert resultado["prediccion"].tolist() == [0, 0, 1]
    assert {"ID", "Estado", "Fraude"} <= set(resultado.columns)  # conserva el lote original


def test_umbral_de_config_cambia_la_prediccion(lote):
    assert predecir(ModeloFalso(), lote, umbral=0.05)["prediccion"].tolist() == [1, 1, 1]


def test_predecir_usa_solo_las_columnas_del_modelo(lote):
    # El lote trae columnas extra (ID, Estado, Fraude) que el modelo no debe recibir.
    class ModeloEstricto(ModeloFalso):
        def predict_proba(self, X):
            assert list(X.columns) == ["Balance_Tj", "Trans_Int"]
            return super().predict_proba(X)

    predecir(ModeloEstricto(), lote, umbral=0.5)


def test_lote_con_drift_sigue_siendo_puntuable(lote):
    con_drift = aplicar_drift(lote, 1.0, ["Balance_Tj", "Trans_Int"], "Fraude", 0.3)
    resultado = predecir(ModeloFalso(), con_drift, umbral=0.5)
    assert len(resultado) == len(lote)
    assert resultado["probabilidad_fraude"].between(0, 1).all()