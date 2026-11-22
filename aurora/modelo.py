"""Modelo de score (gradient boosting), equidad, explicabilidad y latencia."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from .pipeline import FEATURES_PROXY

COLS_EXCLUIR = ["rezago_telco_h", "score_con_rezago", "optin_social"]
GRUPOS = {"rural": "Rural", "adulto_mayor": "Adulto mayor (60+)", "informal": "Trabajo informal"}


def columnas_modelo(df, quitar_proxys=False):
    cols = [c for c in df.columns if c not in COLS_EXCLUIR]
    if quitar_proxys:
        cols = [c for c in cols if c not in FEATURES_PROXY]
    return cols


def pesos_reweighting(demo):
    """Pesos inversos a la sobre/sub-representación de cada grupo en el histórico (reweighting)."""
    ref = {"rural": 0.30, "adulto_mayor": 0.10, "informal": None}
    w = np.ones(len(demo))
    for g, pref in [("rural", 0.30), ("adulto_mayor", 0.10)]:
        obs = demo[g].mean()
        w *= np.where(demo[g] == 1, pref / obs, (1 - pref) / (1 - obs))
    return w


def entrenar(X, y, w=None, seed=7):
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=5,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(X, y, sample_weight=w)
    return m


def ks(y, p):
    d = pd.DataFrame(dict(y=y, p=p)).sort_values("p")
    c1 = (d.y == 1).cumsum() / max((d.y == 1).sum(), 1)
    c0 = (d.y == 0).cumsum() / max((d.y == 0).sum(), 1)
    return float((c1 - c0).abs().max())


def aprobacion(p, tasa=0.60):
    """Aprueba al `tasa` de menor riesgo (corte común a todos los grupos)."""
    corte = np.quantile(p, tasa)
    return (p <= corte).astype(int), corte


def equidad(aprobado, demo, y=None):
    filas = []
    for g, nombre in GRUPOS.items():
        a1, a0 = aprobado[demo[g] == 1].mean(), aprobado[demo[g] == 0].mean()
        fila = dict(grupo=nombre, aprob_grupo=round(a1, 3), aprob_resto=round(a0, 3), razon_aprobacion=round(a1 / a0, 3))
        if y is not None:  # tasa de buenos pagadores rechazados (falsos rechazos)
            b1 = (y[(demo[g] == 1)] == 0); b0 = (y[(demo[g] == 0)] == 0)
            fr1 = 1 - aprobado[(demo[g] == 1) & (y == 0)].mean(); fr0 = 1 - aprobado[(demo[g] == 0) & (y == 0)].mean()
            fila.update(falso_rechazo_grupo=round(fr1, 3), falso_rechazo_resto=round(fr0, 3))
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar(modelo, X, y, demo, tasa=0.60):
    p = modelo.predict_proba(X)[:, 1]
    ap, corte = aprobacion(p, tasa)
    eq = equidad(ap, demo, y)
    return dict(auc=roc_auc_score(y, p), ks=ks(y, p), mora_aprobados=float(y[ap == 1].mean()),
                mora_total=float(y.mean()), equidad=eq, p=p, aprobado=ap)


def importancia(modelo, X, y, n=4000, seed=3):
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    r = permutation_importance(modelo, X.iloc[idx], y[idx], scoring="roc_auc", n_repeats=5, random_state=seed)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def explicar_local(modelo, X_ref, fila, top=5):
    """Explicación local aproximada por oclusión: cuánto cambia el score al reemplazar
    cada variable por la mediana de la población (aprox. a SHAP/LIME, sin dependencias)."""
    base = modelo.predict_proba(fila.to_frame().T.astype(float))[:, 1][0]
    med = X_ref.median()
    contrib = {}
    for c in X_ref.columns:
        f2 = fila.copy(); f2[c] = med[c]
        contrib[c] = base - modelo.predict_proba(f2.to_frame().T.astype(float))[:, 1][0]
    s = pd.Series(contrib).sort_values(key=np.abs, ascending=False).head(top)
    return base, s


def latencia_ms(modelo, X, n=1000, seed=5):
    """Mide la inferencia puntual (una solicitud por llamada) dentro del proceso."""
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    t = []
    for i in idx:
        t0 = time.perf_counter(); modelo.predict_proba(X.iloc[[i]]); t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)), max=float(t.max()))
