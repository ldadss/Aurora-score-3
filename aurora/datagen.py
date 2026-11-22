"""Generador de datos sintéticos de las 6 fuentes del caso AURORA SCORE.

NO son datos reales de VFG: se simulan para poder probar la arquitectura y el
modelo sin exponer datos personales. Los parámetros reproducen los sesgos que el
Informe N°1 identificó (representatividad, proxys, rezago por fuente).
"""
import numpy as np
import pandas as pd

SEED = 42


def _sig(x):
    return 1 / (1 + np.exp(-x))


def generar_poblacion(n=40000, seed=SEED):
    rng = np.random.default_rng(seed)
    rural = rng.random(n) < 0.30
    edad = np.where(rng.random(n) < 0.10, rng.integers(60, 82, n), rng.integers(21, 60, n))
    informal = rng.random(n) < np.where(rural, 0.42, 0.22)
    mujer = rng.random(n) < 0.5
    ingreso = np.exp(rng.normal(0, 0.45, n) - 0.35 * informal - 0.30 * rural)  # índice de ingreso
    z = rng.normal(0, 1, n)  # riesgo latente (independiente del grupo)
    p_mora = _sig(-2.35 + 0.95 * z)
    mora_real = (rng.random(n) < p_mora).astype(int)
    # sesgo histórico: la mora observada en la base de entrenamiento está inflada
    # para grupos con menor acceso a canales de pago/cobranza (se aplica solo al histórico)
    pop = pd.DataFrame(dict(id=np.arange(n), rural=rural.astype(int), edad=edad,
                            informal=informal.astype(int), mujer=mujer.astype(int),
                            ingreso=ingreso, z=z, p_mora=p_mora, mora_real=mora_real))
    return pop, rng


def generar_fuentes(pop, rng):
    n = len(pop)
    rural, inf, ing, z, ed = pop.rural.values, pop.informal.values, pop.ingreso.values, pop.z.values, pop.edad.values
    # --- telecomunicaciones (batch, rezago 24-72 h; geolocalización por antena) ---
    telco = pd.DataFrame(dict(id=pop.id))
    telco["recarga_mensual_usd"] = np.round(8 * ing * np.exp(rng.normal(0, .25, n)), 2)
    telco["regularidad_recarga"] = np.clip(0.62 - 0.14 * z + rng.normal(0, .12, n), 0, 1)
    telco["plan_postpago"] = (rng.random(n) < _sig(-1.0 + 0.9 * np.log(ing) - 0.25 * z - 0.6 * rural)).astype(int)
    telco["antiguedad_telefono_meses"] = np.clip(rng.gamma(4, 5, n) * (0.5 + 0.7 * ing) * (1 - 0.25 * rural) * (1 - 0.003 * (ed - 40).clip(0)) - 2 * z, 1, 120)
    telco["densidad_antenas"] = np.clip(np.where(rural, rng.normal(0.25, .10, n), rng.normal(0.80, .12, n)), 0.02, 1)
    telco["rezago_h"] = rng.choice([24, 48, 72], n, p=[.5, .3, .2])
    # calidad de la geolocalización peor en zonas rurales -> datos faltantes
    miss = rng.random(n) < np.where(rural, 0.22, 0.04)
    telco.loc[miss, ["regularidad_recarga", "antiguedad_telefono_meses"]] = np.nan
    # --- utilities (batch mensual) ---
    util = pd.DataFrame(dict(id=pop.id))
    util["pagos_a_tiempo_12m"] = np.clip(0.80 - 0.17 * z + rng.normal(0, .10, n), 0, 1)
    util["meses_servicio_activo"] = np.clip(rng.normal(30, 14, n) + 4 * ing - 3 * z, 1, 120).round()
    util["rezago_h"] = 24 * 30  # actualización mensual
    miss = rng.random(n) < 0.06
    util.loc[miss, "pagos_a_tiempo_12m"] = np.nan
    # --- billeteras y e-commerce (streaming) ---
    wal = pd.DataFrame(dict(id=pop.id))
    wal["tx_mensuales"] = np.clip(rng.poisson(18 * ing * (1 + 0.1 * (ed < 40)), n) - (0.8 * z).round(), 0, None)
    wal["ticket_promedio_usd"] = np.round(12 * ing * np.exp(rng.normal(0, .3, n)), 2)
    wal["ratio_ingreso_egreso"] = np.clip(1.05 - 0.10 * z + rng.normal(0, .15, n), 0.2, 3)
    wal["rezago_h"] = 0
    miss = rng.random(n) < np.where(ed >= 60, 0.40, 0.12)  # poca huella digital en mayores
    wal.loc[miss, ["tx_mensuales", "ticket_promedio_usd", "ratio_ingreso_egreso"]] = np.nan
    # --- app VFG (streaming) ---
    app = pd.DataFrame(dict(id=pop.id))
    app["sesiones_30d"] = np.clip(rng.poisson(9, n) - (0.8 * z).round(), 0, None)
    app["seg_llenado_formulario"] = np.clip(rng.normal(210, 55, n) + 14 * z + 40 * (ed >= 60), 40, None)
    app["cambios_dispositivo_6m"] = rng.poisson(np.clip(0.4 + 0.25 * z, 0.05, None), n)
    app["rezago_h"] = 0
    # --- redes sociales (solo opt-in ~40%; NO se analiza contenido) ---
    soc = pd.DataFrame(dict(id=pop.id))
    optin = rng.random(n) < 0.40
    soc["optin_social"] = optin.astype(int)
    soc["antiguedad_cuenta_meses"] = np.where(optin, np.clip(rng.normal(70, 30, n) - 4 * z - 25 * (ed >= 60), 1, None), np.nan)
    soc["rezago_h"] = 24
    # --- biometría: SOLO antifraude, fuera del score ---
    bio = pd.DataFrame(dict(id=pop.id))
    bio["alerta_antifraude"] = (rng.random(n) < 0.012).astype(int)
    bio["rezago_h"] = 0
    return dict(telco=telco, utilities=util, billeteras=wal, app=app, social=soc, biometria=bio)


def generar_historico_sesgado(pop, rng):
    """Base histórica de entrenamiento: sub-representa grupos y la mora observada está inflada."""
    w = np.ones(len(pop))
    w *= np.where(pop.rural == 1, 0.22, 1)
    w *= np.where(pop.edad >= 60, 0.30, 1)
    w *= np.where(pop.informal == 1, 0.40, 1)
    sel = rng.random(len(pop)) < w * 0.9
    hist = pop.copy()
    infl = 0.55 * hist.rural + 0.35 * (hist.edad >= 60) + 0.45 * hist.informal  # menor acceso a canales de pago
    p = _sig(np.log(hist.p_mora / (1 - hist.p_mora)) + infl)
    hist["mora_obs"] = (rng.random(len(pop)) < p).astype(int)
    hist["en_historico"] = sel
    return hist
