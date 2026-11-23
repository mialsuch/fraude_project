"""Pruebas de la lógica de drift (src/monitoring/monitoreo.py). No necesitan MLflow, Prefect ni Evidently."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.monitoreo import (
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    hay_alerta,
    ruta_lote,
    validar_lote,
)

NUMERICAS = ["Num_Tarjetas", "Balance_Tj", "Total_Transac", "Trans_Int"]
OBJETIVO = "Fraude"
SUBIDA = 0.3


@pytest.fixture
def lote():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({columna: rng.normal(100, 10, 1000) for columna in NUMERICAS})
    df["Estado"] = rng.choice(["Texas", "Ohio"], 1000)
    df[OBJETIVO] = (rng.random(1000) < 0.06).astype(int)
    return df


def test_nivel_cero_no_cambia_el_lote(lote):
    pd.testing.assert_frame_equal(aplicar_drift(lote, 0.0, NUMERICAS, OBJETIVO, SUBIDA), lote)


@pytest.mark.parametrize("nivel, columnas", [(0.3, 1), (0.6, 2), (1.0, 4)])
def test_drift_desplaza_la_fraccion_indicada_de_variables(lote, nivel, columnas):
    con_drift = aplicar_drift(lote, nivel, NUMERICAS, OBJETIVO, SUBIDA)
    cambiadas = [c for c in NUMERICAS if not np.allclose(con_drift[c], lote[c])]
    assert len(cambiadas) == columnas
    for c in cambiadas:  # la media sube 2 desviaciones típicas
        assert con_drift[c].mean() == pytest.approx(lote[c].mean() + 2 * lote[c].std())


def test_drift_sube_la_tasa_de_fraude_segun_el_nivel(lote):
    con_drift = aplicar_drift(lote, 0.6, NUMERICAS, OBJETIVO, SUBIDA)
    assert con_drift[OBJETIVO].mean() == pytest.approx(lote[OBJETIVO].mean() + 0.6 * SUBIDA, abs=0.002)


def test_drift_no_modifica_el_lote_original(lote):
    copia = lote.copy()
    aplicar_drift(lote, 1.0, NUMERICAS, OBJETIVO, SUBIDA)
    pd.testing.assert_frame_equal(lote, copia)


@pytest.mark.parametrize("nivel", [-0.1, 1.5])
def test_nivel_fuera_de_rango_falla(lote, nivel):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        aplicar_drift(lote, nivel, NUMERICAS, OBJETIVO, SUBIDA)


def test_ruta_lote_incluye_el_nivel():
    assert ruta_lote("data/lotes", 0.6).name == "lote_nivel_0.6.csv"


def test_fraccion_desde_evidently():
    resultado = {
        "metrics": [
            {"metric_name": "ValueDrift(column=Balance_Tj)", "value": 0.4},
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 2, "share": 0.2857}},
        ]
    }
    assert fraccion_desde_evidently(resultado) == pytest.approx(0.2857)


def test_fraccion_desde_evidently_sin_metrica_falla():
    with pytest.raises(ValueError, match="DriftedColumnsCount"):
        fraccion_desde_evidently({"metrics": []})


@pytest.mark.parametrize("fraccion, alerta", [(0.0, False), (0.14, False), (0.2, False), (0.29, True)])
def test_alerta_solo_por_encima_del_umbral(fraccion, alerta):
    assert hay_alerta(fraccion, umbral=0.2) is alerta


def test_lote_sin_etiquetas_se_rechaza(lote):
    with pytest.raises(ValueError, match="no tiene la columna 'Fraude'"):
        validar_lote(lote.drop(columns=[OBJETIVO]), OBJETIVO)


def test_lote_con_etiquetas_vacias_se_rechaza(lote):
    lote[OBJETIVO] = lote[OBJETIVO].astype(float)
    lote.loc[0, OBJETIVO] = None
    with pytest.raises(ValueError, match="sin valor"):
        validar_lote(lote, OBJETIVO)


def test_drift_de_target_detecta_subida_de_fraude():
    referencia = pd.Series([0] * 940 + [1] * 60)
    lote = pd.Series([0] * 760 + [1] * 240)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is True


def test_drift_de_target_sin_cambio_no_alerta():
    referencia = pd.Series([0] * 9400 + [1] * 600)
    lote = pd.Series([0] * 940 + [1] * 60)
    resultado = drift_de_target(referencia, lote, alfa=0.05)
    assert resultado["drift"] is False
    assert resultado["tasa_lote"] == pytest.approx(0.06)