"""Ejecuta el pipeline completo y guarda métricas y figuras. Uso: python run_all.py"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
RAIZ = Path(__file__).parent; sys.path.insert(0, str(RAIZ))
from aurora import datagen, pipeline as pl, modelo as md

NAVY, BLUE, ORG, GRY = "#1F3864", "#2F5597", "#F2B632", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
(RAIZ / "figuras").mkdir(exist_ok=True)

# 1) datos y ingesta -----------------------------------------------------------
pop, rng = datagen.generar_poblacion()
fuentes = datagen.generar_fuentes(pop, rng)
hist = datagen.generar_historico_sesgado(pop, rng)
lote, m_batch = pl.ingesta_batch(fuentes, RAIZ / "data")
estado, m_stream = pl.ingesta_streaming(pl.generar_eventos(fuentes, 200_000, rng))
calidad = pl.reporte_calidad(fuentes)
calidad.to_csv(RAIZ / "data" / "reporte_calidad.csv", index=False)
X = pl.construir_features(fuentes)
demo = pop.set_index("id")[["rural", "edad", "informal"]].copy(); demo["adulto_mayor"] = (demo.edad >= 60).astype(int)

# 2) entrenamiento y evaluación ------------------------------------------------
en_hist = hist.set_index("id")["en_historico"].values
y_obs = hist.set_index("id")["mora_obs"].values; y_real = pop.set_index("id")["mora_real"].values
test = np.random.default_rng(11).random(len(X)) < 0.30        # población representativa (no sesgada)
tr = en_hist & ~test
res = {}
for nombre, qp, rw in [("M0 base", False, False), ("M1 reweighting", False, True), ("M2 reweighting\n+ sin proxys", True, True)]:
    cols = md.columnas_modelo(X, qp)
    m = md.entrenar(X[cols][tr], y_obs[tr], md.pesos_reweighting(demo[tr]) if rw else None)
    res[nombre] = (m, cols, md.evaluar(m, X[cols][test], y_real[test], demo[test]))
tabla = pd.DataFrame([dict(modelo=k.replace("\n", " "), AUC=round(v[2]["auc"], 3), KS=round(v[2]["ks"], 3),
        razon_min=float(v[2]["equidad"].razon_aprobacion.min()), mora_aprobados=round(v[2]["mora_aprobados"], 3)) for k, v in res.items()])
tabla.to_csv(RAIZ / "data" / "comparacion_modelos.csv", index=False)
eq = {k.replace("\n", " "): v[2]["equidad"] for k, v in res.items()}
for k, v in eq.items(): v.to_csv(RAIZ / "data" / f"equidad_{k.split()[0]}.csv", index=False)

# 3) explicabilidad y latencia (modelo final M2) ------------------------------
mF, colsF, rF = res["M2 reweighting\n+ sin proxys"]
imp = md.importancia(mF, X[colsF][test], y_real[test])
Xt = X[colsF][test]; pt = rF["p"]; cand = np.where((rF["aprobado"] == 0) & (demo[test].rural.values == 1))[0][0]
base, contrib = md.explicar_local(mF, Xt, Xt.iloc[cand])
lat = md.latencia_ms(mF, Xt)

# 4) figuras -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.8)); ks_ = list(res); xs = np.arange(3); wd = .26
cols_g = [NAVY, BLUE, ORG]
for i, g in enumerate(md.GRUPOS.values()):
    vals = [res[k][2]["equidad"].set_index("grupo").loc[g, "razon_aprobacion"] for k in ks_]
    ax.bar(xs + (i - 1) * wd, vals, wd, label=g, color=cols_g[i])
ax.axhline(0.80, color="#C00000", ls="--", lw=1, label="Umbral 0,80")
ax.set_xticks(xs); ax.set_xticklabels(ks_); ax.set_ylim(0, 1.15); ax.set_ylabel("Razón de aprobación\n(grupo / resto)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.13)); fig.tight_layout(); fig.savefig(RAIZ / "figuras" / "fig_equidad.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 2.8)); top = imp.head(10)[::-1]
ax.barh(top.index, top.values, color=BLUE); ax.set_xlabel("Caída de AUC al permutar la variable (importancia global)")
fig.tight_layout(); fig.savefig(RAIZ / "figuras" / "fig_importancia.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 2.4)); c = contrib[::-1]
ax.barh(c.index, c.values, color=[("#C00000" if v > 0 else "#548235") for v in c.values])
ax.axvline(0, color=GRY, lw=.8); ax.set_xlabel(f"Cambio en prob. de mora (score = {base:.3f}); rojo sube el riesgo, verde lo baja")
fig.tight_layout(); fig.savefig(RAIZ / "figuras" / "fig_explicacion_local.png", dpi=200); plt.close()

# 5) métricas a JSON -----------------------------------------------------------
out = dict(batch=m_batch, streaming=m_stream, latencia_ms=lat, tabla=tabla.to_dict("records"),
           importancia=imp.head(10).round(4).to_dict(), explicacion_local=dict(score=round(float(base), 4), contrib=contrib.round(4).to_dict()),
           calidad=calidad.to_dict("records"), n_train=int(tr.sum()), n_test=int(test.sum()),
           mora_hist_obs=float(y_obs[tr].mean()), mora_real=float(y_real.mean()),
           composicion_train={g: round(float(demo[tr][g].mean()), 3) for g in ["rural", "adulto_mayor", "informal"]},
           composicion_pob={g: round(float(demo[g].mean()), 3) for g in ["rural", "adulto_mayor", "informal"]},
           equidad={k: v.to_dict("records") for k, v in eq.items()},
           n_features=dict(base=len(res["M0 base"][1]), final=len(colsF)))
json.dump(out, open(RAIZ / "data" / "metricas.json", "w"), indent=1, default=float, ensure_ascii=False)
if __name__ == "__main__":
    print(tabla.to_string(index=False)); print(calidad.to_string(index=False)); print(json.dumps({k: out[k] for k in ["batch", "streaming", "latencia_ms", "composicion_train", "composicion_pob", "n_train", "n_test", "explicacion_local", "n_features", "mora_hist_obs", "mora_real"]}, indent=1, default=float))
    for k, v in eq.items(): print(k); print(v.to_string(index=False))
