"""Carga los CSV de la competencia Modelos de Fraude 2026 y valida su estructura.

Los datos no se versionan en Git (reglas de la competencia, cédulas y tamaño).
Descárgalos desde la URL de `datos.fuente` en configs/config.yaml y cópialos en data/.

Ejecución (desde la raíz del repositorio):
    poetry run python -m src.data.load_data

Uso desde otro módulo:
    from src.data.load_data import cargar_datos
"""
from pathlib import Path

import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"


#Leer configuración
def leer_config(ruta=RUTA_CONFIG):
    """Devuelve el contenido de config.yaml como diccionario."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


#Cargar y validar
def cargar_datos(conjunto="train", cfg=None):
    """Lee el CSV de entrenamiento ("train") o de prueba ("test") y valida sus columnas.

    El de prueba no trae la columna objetivo, porque es el que se predice en la competencia.
    """
    cfg = cfg or leer_config()
    datos, variables = cfg["datos"], cfg["variables"]

    ruta = Path(datos[f"ruta_{conjunto}"])
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Descárgalo desde {datos['fuente']} y cópialo en data/.")

    df = pd.read_csv(ruta, sep=datos["separador"])

    esperadas = [variables["id"], *variables["numericas"], *variables["categoricas"]]
    if conjunto == "train":
        esperadas.append(variables["objetivo"])
    faltan = [c for c in esperadas if c not in df.columns]
    if faltan:
        raise ValueError(f"A {ruta} le faltan columnas: {faltan}. ¿El separador es '{datos['separador']}'?")

    if df[esperadas].isna().any().any():
        raise ValueError(f"{ruta} tiene valores nulos en las columnas del modelo")
    if conjunto == "train" and not set(df[variables["objetivo"]].unique()) <= {0, 1}:
        raise ValueError("La columna objetivo debe ser binaria (0/1)")

    return df[esperadas].copy()

#Función
def main():
    cfg = leer_config()
    train = cargar_datos("train", cfg)
    test = cargar_datos("test", cfg)
    print(f"train: {train.shape[0]} filas, {train.shape[1]} columnas")
    print(f"test: {test.shape[0]} filas, {test.shape[1]} columnas")
    print(f"Proporción de fraude en train: {train[cfg['variables']['objetivo']].mean():.4f}")


if __name__ == "__main__":
    main()