"""Lógica de monitoreo de drift (actividad 3): emulación de drift, drift de features y drift del target.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
#Importación
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


def dividir(df, division, objetivo):
    """Parte los datos igual que el entrenamiento (misma semilla y estratificación). Devuelve (entrenamiento, prueba)."""
    return train_test_split(
        df, test_size=division["test_size"], random_state=division["random_state"], stratify=df[objetivo]
    )
    
#drift
def aplicar_drift(df, nivel, numericas, objetivo, subida_tasa_maxima, semilla=42):
    """Emula drift en un lote según `nivel` (0.0 a 1.0).

    1. Features: desplaza la media de round(nivel * len(numericas)) variables numéricas en 2 desviaciones típicas.
    2. Target: sube la tasa de fraude en nivel * subida_tasa_maxima puntos, convirtiendo negativos en positivos.
    Con nivel 0.0 devuelve una copia idéntica del lote.
    """
    if not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, no {nivel}")
    out = df.copy()
    rng = np.random.default_rng(semilla)

    n = round(nivel * len(numericas))
    for columna in rng.choice(numericas, size=n, replace=False):
        desviacion = out[columna].std()
        if desviacion > 0:
            out[columna] = out[columna] + 2 * desviacion

    out[objetivo] = _subir_tasa(out[objetivo], nivel * subida_tasa_maxima, rng)
    return out


def _subir_tasa(objetivo, subida, rng):
    """Convierte negativos en positivos, al azar, hasta que la tasa sube `subida` puntos."""
    faltan = round(min(objetivo.mean() + subida, 1.0) * len(objetivo)) - int(objetivo.sum())
    ceros = objetivo.index[objetivo == 0]
    faltan = min(faltan, len(ceros))
    if faltan <= 0:
        return objetivo
    subidos = objetivo.copy()
    subidos.loc[rng.choice(ceros, size=faltan, replace=False)] = 1
    return subidos

#Lote, evidently y alerta
def ruta_lote(carpeta, nivel):
    """Ruta del lote guardado por la inferencia para un nivel de drift, p. ej. data/lotes/lote_nivel_0.6.csv."""
    return Path(carpeta) / f"lote_nivel_{nivel:.1f}.csv"


def fraccion_desde_evidently(resultado):
    """Fracción de variables con drift a partir del resultado de Evidently (`snapshot.dict()`)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return float(metrica["value"]["share"])
    raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")


def hay_alerta(fraccion, umbral):
    """True si la fracción de variables con drift supera el umbral configurado."""
    return fraccion > umbral

#evento drfit
def validar_lote(lote, objetivo):
    """Falla con un mensaje claro si el lote no trae la variable objetivo con valores (etiquetas reales)."""
    if objetivo not in lote.columns:
        raise ValueError(f"El lote no tiene la columna '{objetivo}': no se puede calcular el drift de evento")
    if lote[objetivo].isna().any():
        raise ValueError(f"El lote tiene '{objetivo}' sin valor: no se puede calcular el drift de evento")


def drift_de_target(referencia, lote, alfa):
    """Compara la tasa de fraude del lote con la de referencia con un test chi-cuadrado."""
    pos_ref, pos_lote = int(referencia.sum()), int(lote.sum())
    tabla = [[pos_ref, len(referencia) - pos_ref], [pos_lote, len(lote) - pos_lote]]
    _, p_valor, _, _ = chi2_contingency(tabla)
    return {
        "tasa_referencia": pos_ref / len(referencia),
        "tasa_lote": pos_lote / len(lote),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }