"""Lógica de negocio del modelo de fraude: selección de variables, pipeline y métricas.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ESTIMADORES = {
    "logistic_regression": LogisticRegression,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

#Selección de variables

def resolver_features(variables, features):
    """Devuelve (numericas, categoricas) según `seleccion` de features.yaml: ALL, NUMERIC o CUSTOM."""
    numericas, categoricas = variables["numericas"], variables["categoricas"]
    seleccion = features["seleccion"]
    if seleccion == "ALL":
        return numericas, categoricas
    if seleccion == "NUMERIC":
        return numericas, []
    if seleccion == "CUSTOM":
        elegidas = features.get("custom") or []
        desconocidas = [v for v in elegidas if v not in numericas + categoricas]
        if desconocidas:
            raise ValueError(f"variables desconocidas en 'custom': {desconocidas}")
        if not elegidas:
            raise ValueError("CUSTOM necesita al menos una variable en 'custom'")
        return [v for v in numericas if v in elegidas], [v for v in categoricas if v in elegidas]
    raise ValueError(f"seleccion '{seleccion}' no válida: usa ALL, NUMERIC o CUSTOM")

#Pipeline
def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar numéricas, one-hot categóricas) más el estimador elegido."""
    if tipo not in ESTIMADORES:
        raise ValueError(f"modelo '{tipo}' no soportado: usa {list(ESTIMADORES)}")
    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        # Salida densa: HistGradientBoosting no acepta matrices dispersas.
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return Pipeline([("preprocesado", ColumnTransformer(transformadores)), ("modelo", ESTIMADORES[tipo](**parametros))])


#Metricas
def calcular_metricas(y_true, proba, umbral):
    """Métricas de clasificación binaria a partir de la probabilidad de fraude."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }