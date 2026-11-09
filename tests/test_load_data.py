"""Pruebas de src/data/load_data.py. Se ejecutan desde la raíz del repositorio: poetry run pytest -q"""
import copy
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import cargar_datos, leer_config

CFG = leer_config()
HAY_DATOS = Path(CFG["datos"]["ruta_train"]).exists()


def cfg_con_csv(tmp_path, df, conjunto="train", separador=";"):
    """Copia de la configuración que apunta a un CSV pequeño escrito en una carpeta temporal."""
    cfg = copy.deepcopy(CFG)
    ruta = tmp_path / f"{conjunto}.csv"
    df.to_csv(ruta, sep=separador, index=False)
    cfg["datos"][f"ruta_{conjunto}"] = str(ruta)
    return cfg

#Dataset ejemplo prueba
@pytest.fixture
def df_valido():
    return pd.DataFrame(
        {
            "ID": [1, 2],
            "Num_Tarjetas": [1, 2],
            "Balance_Tj": [2000, 0],
            "Total_Transac": [31, 25],
            "Trans_Int": [9, 0],
            "Estado": ["Alabama", "Texas"],
            "Genero": ["Male", "Female"],
            "Tipo_Tarjeta": ["American Express", "Visa"],
            "Fraude": [0, 1],
        }
    )
    
#Pruebas   
def test_carga_un_csv_valido(tmp_path, df_valido):
    df = cargar_datos("train", cfg_con_csv(tmp_path, df_valido))
    assert df.shape == (2, 9)


def test_separador_incorrecto_falla(tmp_path, df_valido):
    cfg = cfg_con_csv(tmp_path, df_valido, separador=",")  # el config sigue esperando ";"
    with pytest.raises(ValueError, match="faltan columnas"):
        cargar_datos("train", cfg)


def test_nulos_fallan(tmp_path, df_valido):
    df_valido.loc[0, "Balance_Tj"] = None
    with pytest.raises(ValueError, match="nulos"):
        cargar_datos("train", cfg_con_csv(tmp_path, df_valido))


def test_objetivo_no_binario_falla(tmp_path, df_valido):
    df_valido.loc[0, "Fraude"] = 2
    with pytest.raises(ValueError, match="binaria"):
        cargar_datos("train", cfg_con_csv(tmp_path, df_valido))


def test_archivo_inexistente_indica_la_fuente(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["datos"]["ruta_train"] = str(tmp_path / "no_existe.csv")
    with pytest.raises(FileNotFoundError, match="kaggle"):
        cargar_datos("train", cfg)
        
        
@pytest.mark.skipif(not HAY_DATOS, reason="data/train.csv no está descargado")
def test_datos_reales_de_la_competencia():
    train = cargar_datos("train")
    test = cargar_datos("test")
    assert train.shape == (848548, 9)
    assert test.shape == (200027, 8)
    assert "Fraude" not in test.columns