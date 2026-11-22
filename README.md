# AURORA SCORE — Prototipo técnico (EP5)
Gestión de Proyectos de Datos (SCY1102) · Equipo: Ignacio Cerda, Sebastián González, Jesús Morán

Implementa el tramo F3–F5 del Plan de Proyecto: ingesta batch/streaming, calidad de datos, variables,
modelo de score (gradient boosting), auditoría de equidad con mitigación y explicabilidad.

> **Datos 100% sintéticos.** No son datos de Vertex Financial Group; reproducen los sesgos descritos en el
> Informe de Factibilidad N°1 para probar el pipeline sin exponer datos personales.

## Estructura
| Ruta | Contenido |
|---|---|
| `aurora/datagen.py` | Generador de las 6 fuentes y del histórico sesgado |
| `aurora/pipeline.py` | Ingesta batch y streaming, calidad/frescura, feature store |
| `aurora/modelo.py` | Modelos, equidad, explicabilidad, latencia |
| `run_all.py` | Ejecuta todo y escribe `data/metricas.json` y `figuras/` |
| `notebooks/AURORA_SCORE_prototipo.ipynb` | Recorrido documentado y ejecutado |
| `gestion/Tablero_Seguimiento_AURORA.xlsx` | Tareas planificadas vs. ejecutadas, avance, desviaciones, riesgos |

## Cómo reproducir
```bash
pip install -r requirements.txt
python run_all.py                                   # métricas y figuras
jupyter nbconvert --to notebook --execute --inplace notebooks/AURORA_SCORE_prototipo.ipynb
```
Semilla fija (42 / 11 / 7): los resultados son reproducibles.

## Resultados (población de evaluación, n = 12.170)
| Modelo | AUC | KS | Razón mínima de aprobación | Cumple ≥ 0,80 |
|---|---|---|---|---|
| M0 base | 0,708 | 0,303 | 0,665 | No |
| M1 reweighting | 0,704 | 0,297 | 0,722 | No |
| M2 reweighting + sin proxys | 0,707 | 0,301 | 0,935 | Sí |

## Limitaciones
Datos sintéticos; la latencia (≈1 ms p95) es solo de inferencia en proceso, no extremo a extremo;
la explicabilidad local es aproximada (oclusión), no SHAP.
