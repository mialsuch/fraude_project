"""Pruebas de datos: esquema, rangos y nulos de los CSV de la competencia.

Necesitan data/train.csv y data/test.csv; si faltan, se saltan.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import cargar_datos, leer_config

CFG = leer_config()
VARIABLES = CFG["variables"]
pytestmark = pytest.mark.skipif(
    not Path(CFG["datos"]["ruta_train"]).exists(), reason="data/train.csv no está descargado"
)


@pytest.fixture(scope="module")
def train():
    return cargar_datos("train", CFG)


def test_esquema_y_tipos(train):
    for columna in VARIABLES["numericas"] + [VARIABLES["objetivo"]]:
        assert pd.api.types.is_integer_dtype(train[columna]), columna
    for columna in VARIABLES["categoricas"]:
        assert not pd.api.types.is_numeric_dtype(train[columna]), columna


def test_sin_nulos(train):
    assert train.isna().sum().sum() == 0


def test_rangos_de_las_numericas(train):
    assert train["Num_Tarjetas"].between(1, 2).all()
    for columna in ["Balance_Tj", "Total_Transac", "Trans_Int"]:
        assert (train[columna] >= 0).all(), columna
    assert (train["Trans_Int"] <= 60).all()


def test_categorias_conocidas(train):
    assert set(train["Genero"]) == {"Male", "Female"}
    assert train["Tipo_Tarjeta"].nunique() == 4
    assert train["Estado"].nunique() <= 51


def test_objetivo_binario_y_desbalanceado(train):
    tasa = train[VARIABLES["objetivo"]].mean()
    assert set(train[VARIABLES["objetivo"]].unique()) == {0, 1}
    assert 0.03 < tasa < 0.10  # la tasa de fraude histórica es ~6 %


def test_ids_unicos(train):
    assert train[VARIABLES["id"]].is_unique


def test_test_csv_tiene_el_mismo_esquema_sin_objetivo():
    test = cargar_datos("test", CFG)
    assert VARIABLES["objetivo"] not in test.columns
    assert test.isna().sum().sum() == 0