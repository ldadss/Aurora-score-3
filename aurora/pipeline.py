"""Ingesta (batch + streaming simulado), calidad de datos y construcción de variables."""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BATCH = ["telco", "utilities", "social"]          # llegan por lotes (archivos)
STREAM = ["billeteras", "app", "biometria"]       # llegan por eventos
SLA_REZAGO_H = {"telco": 48, "utilities": 24 * 31, "billeteras": 1, "app": 1, "social": 48, "biometria": 1}  # horas máx. por fuente (RF03)
FEATURES_PROXY = ["antiguedad_telefono_meses", "plan_postpago", "densidad_antenas"]


# ---------------- ingesta batch ----------------
def ingesta_batch(fuentes, carpeta):
    """Escribe y vuelve a leer cada fuente por lotes (simula la carga nocturna al lakehouse)."""
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); filas = 0; out = {}
    for nombre in BATCH:
        ruta = carpeta / f"{nombre}.csv"
        fuentes[nombre].to_csv(ruta, index=False)
        out[nombre] = pd.read_csv(ruta); filas += len(out[nombre])
    return out, dict(filas=filas, segundos=time.perf_counter() - t0)


# ---------------- ingesta streaming (simulada) ----------------
def generar_eventos(fuentes, n_eventos, rng):
    """Genera un flujo de eventos (billeteras / app) a partir de las tablas base."""
    ids = rng.integers(0, len(fuentes["app"]), n_eventos)
    tipo = rng.choice(["wallet_tx", "app_sesion"], n_eventos, p=[.7, .3])
    monto = np.round(rng.lognormal(2.2, .6, n_eventos), 2)
    for i, t, m in zip(ids, tipo, monto):
        yield dict(id=int(i), tipo=t, monto=float(m))


def ingesta_streaming(eventos, tam_micro_lote=1000):
    """Consume el flujo en micro-lotes y mantiene contadores por cliente (estado)."""
    estado = defaultdict(lambda: [0, 0.0, 0])  # tx, monto, sesiones
    n = 0; t0 = time.perf_counter(); lote = []
    for ev in eventos:
        lote.append(ev); n += 1
        if len(lote) == tam_micro_lote:
            _procesar(lote, estado); lote = []
    if lote:
        _procesar(lote, estado)
    seg = time.perf_counter() - t0
    return estado, dict(eventos=n, segundos=seg, eventos_por_seg=n / seg)


def _procesar(lote, estado):
    for ev in lote:
        s = estado[ev["id"]]
        if ev["tipo"] == "wallet_tx":
            s[0] += 1; s[1] += ev["monto"]
        else:
            s[2] += 1


# ---------------- calidad de datos (RF03) ----------------
def reporte_calidad(fuentes):
    filas = []
    for nombre, df in fuentes.items():
        datos = df.drop(columns=[c for c in ["id", "rezago_h"] if c in df.columns])
        compl = 1 - datos.isna().mean().mean()
        rez = df["rezago_h"].max() if "rezago_h" in df else 0
        pct_rez = (df["rezago_h"] > SLA_REZAGO_H[nombre]).mean() if "rezago_h" in df else 0
        filas.append(dict(fuente=nombre, modo="batch" if nombre in BATCH else "streaming",
                          filas=len(df), completitud=round(compl, 3),
                          duplicados_id=int(df["id"].duplicated().sum()),
                          rezago_max_h=int(rez), pct_fuera_SLA=round(float(pct_rez), 3)))
    return pd.DataFrame(filas)


def completitud_por_grupo(fuentes, pop, col_fuente="telco", cols=("regularidad_recarga",)):
    df = fuentes[col_fuente].merge(pop[["id", "rural", "edad", "informal"]], on="id")
    df["adulto_mayor"] = (df.edad >= 60).astype(int)
    cols = list(cols)
    res = {}
    for g in ["rural", "adulto_mayor", "informal"]:
        res[g] = df.groupby(g)[cols].apply(lambda x: 1 - x.isna().mean().mean()).round(3).to_dict()
    return res


# ---------------- variables (feature store mínimo) ----------------
def construir_features(fuentes):
    """Une las fuentes del score. La biometría NO entra (solo antifraude)."""
    df = fuentes["telco"].drop(columns="rezago_h").rename(columns={})
    for n in ["utilities", "billeteras", "app", "social"]:
        df = df.merge(fuentes[n].drop(columns="rezago_h"), on="id", how="left")
    df["rezago_telco_h"] = fuentes["telco"]["rezago_h"].values
    df["score_con_rezago"] = (df["rezago_telco_h"] > SLA_REZAGO_H["telco"]).astype(int)
    df["recarga_por_tx"] = df["recarga_mensual_usd"] / (df["tx_mensuales"].fillna(0) + 1)
    return df.set_index("id")
