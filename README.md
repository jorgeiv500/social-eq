# Bogotá EV Charging Social Equity Reproduction

Reproducible local pipeline for:

`Public electric-vehicle charging and accessibility inequality in Bogotá: A socio-spatial allocation framework`
(enviado a *Journal of Transport Geography*)

The repository stages a minimal input bundle, solves scenarios `S0`-`S3` with `Pyomo + NEOS/CPLEX`, runs the rapid robustness battery, and exports the curated paper figures/tables under:

`outputs/social_equity_followup_outputs/paper_v6_experiments`

## Versión enviada a la revista

La carpeta `submission/jtg_2026-04/` es el **registro canónico** de lo que se envió:
el fuente LaTeX, manuscrito y suplemento en PDF, las nueve figuras finales, las tablas
de respaldo, el código que las generó y los manifiestos de la corrida.
Su `README.md` incluye el mapeo figura↔archivo, las cifras clave y las sumas de verificación.

- Número de manuscrito: **JTRG-D-26-00692**
- PDF compilado: 2026-04-17 (pdfTeX / TeX Live 2025)
- Enviado a la revista: 2026-04-17
- Figuras y tablas: 2026-04-17
- Corrida de escenarios S0–S3: 2026-04-15 (Pyomo + NEOS/CPLEX)
- Fuente LaTeX: incluido en `submission/jtg_2026-04/00_latex_fuente/` (exportado de Overleaf)

## Revisión en curso

El 2026-08-20 la revista devolvió el artículo con **revisión mayor** y plazo hasta el
**2026-10-11**. Los tres informes de los revisores, la nota del editor y una copia de trabajo
del fuente LaTeX están en `revision/r1_2026-08/`.

Esa carpeta **no se versiona**: los informes son confidenciales y el repositorio es público.
Ver `revision/LEEME_CONFIDENCIAL.md`.

El material intermedio y las copias superadas quedaron en
`outputs/social_equity_followup_outputs/paper_v6_experiments/_archivo_superado/`
(fuera del control de versiones).

## Corrección y paquete de replicación (septiembre de 2026)

Durante la revisión se detectó que el cuaderno con el que se entrenó el ST-GNN para el envío
de abril emparejaba códigos de zona con ceros a la izquierda (`UTAM043`) contra códigos sin
ellos (`UTAM43`): la etiqueta de entrenamiento quedó en cero en 98 de las 142 zonas. El modelo
se reentrenó con el emparejamiento corregido y **todos** los resultados se recalcularon
(41 corridas NEOS/CPLEX, todas óptimas). Los insumos superados se conservan en
`data/inputs/superseded_2026-04/`.

- `REPRODUCIBILITY.md` (en inglés): mapa de cada tabla y figura del manuscrito al archivo que
  contiene sus valores y al comando que lo regenera, por niveles (sin solver, con NEOS, con
  reentrenamiento).
- `scripts/reproduce_all.sh [NIVEL]`: ejecuta los niveles en orden; los niveles con NEOS o con
  microdatos no arrancan sin `NEOS_EMAIL` o `EM2023_DIR`.
- `experiments/revision_2026/`: los doce experimentos de la revisión (barrido de tolerancia,
  benchmark de equidad directa, crecimiento 2028/2030, topes de subestación, baselines
  estáticos, ablación de pesos, tenencia, contrafactual de prioridad, pisos vinculantes,
  mapas, duraciones, figuras de impresión), cada uno con README, código, insumos pequeños y
  salidas. Los archivos crudos que no se redistribuyen (EM-2021, EM-2023) se documentan con su
  fuente.
- `scripts/hoja_de_cifras.py`: extrae todos los números que cita el manuscrito con su archivo
  de origen (`outputs/.../hoja_de_cifras.md`).
- `requirements-lock.txt`: entorno exacto (Python 3.9.6).

## Repository layout

- `data/inputs`: versioned inputs required for reproduction.
- `data/reference`: optional reference outputs for comparison.
- `scripts/stage_inputs.py`: stages local inputs from known `Downloads` and synchronized Google Drive paths.
- `scripts/reproduce_followup.py`: runs the full reproduction pipeline. **It has no command-line
  options and submits the NEOS runs as soon as it starts**; prefer `scripts/run_pipeline_resiliente.py`
  (socket timeout and retries) for a deliberate re-solve.
- `scripts/retrain_stgnn.py`: retrains the ST-GNN from the EM-2023 microdata (`EM2023_DIR`).
- `scripts/make_vulnerable_quartile_table.py`, `scripts/hoja_de_cifras.py`: supplementary Table E1 and the sheet of cited numbers.
- `scripts/reproduce_all.sh`: level-based driver (see `REPRODUCIBILITY.md`).
- `experiments/revision_2026/`: revision experiments A1–A4, B1–B4, C1, E3, E6, E7 (shared helpers in `comun.py`).
- `outputs/stgnn_retraining/peso_fexp_miles/`: archived retraining outputs and run protocol.
- `src/social_eq/followup.py`: model logic, figures, tables, cleanup, and paper-package export.
- `docs/jtg_figure_style_notes.md`: visual rules transferred from the reference article.
- `docs/paper_submission_recommendations.md`: recommended main-text vs supplementary figure/table plan.
- `submission/jtg_2026-04/`: paquete definitivo del envío a *Journal of Transport Geography*.

## Quick start

1. Install the project into the virtual environment:

```bash
.venv/bin/pip install -e .
```

2. Stage local inputs:

```bash
.venv/bin/python scripts/stage_inputs.py
```

3. Run the full reproduction:

```bash
.venv/bin/python scripts/reproduce_followup.py
```

If you want to override the NEOS account used by the solver:

```bash
export NEOS_EMAIL="your_email@domain.com"
```

## Solver and reproducibility notes

- Optimization runs through `NEOS` with `CPLEX` by default.
- The staged manifest is written to `data/inputs/input_manifest.json`.
- The run manifest is written to `outputs/social_equity_followup_outputs/run_manifest.json`.
- The current reproduction closes the OD matrix with `survey_graph_completed` when the alternative full matrix is unavailable locally.

## Curated paper outputs

Los artefactos se exportan a `outputs/social_equity_followup_outputs/paper_v6_experiments/`
y se consolidan en `submission/jtg_2026-04/`.

Figuras del texto principal del manuscrito revisado (la Fig. 1 es un diagrama TikZ dentro del
LaTeX y la Fig. 2, el área de estudio, la genera `experiments/revision_2026/E3_mapas/`):

- `fig_official_ev_growth.png` (Fig. 3)
- `fig_social_context_panels.png` (Fig. 4)
- `fig_attention_profile.png` (Fig. 5)
- `fig_allocation_composite.png` (Fig. 6)
- `fig_net_change_S2_minus_S1.png` (Fig. 7)
- `fig_lorenz_curves.png` (Fig. 8)
- `fig_policy_tradeoff.png` (Fig. 9, cuatro paneles; superpone el barrido A1 y el benchmark A2)

Figuras del suplemento:

- `fig_growth_progression.png` (Fig. C1)
- `fig_robustness.png` (Fig. C2)

Figuras generadas como respaldo, no incluidas en el envío:

- `fig_gnn_learning_history.png`, `fig_priority_map.png`, `fig_S3_protected_set.png`,
  `fig_allocation_S0.png`–`fig_allocation_S3.png`

Tablas del texto principal:

- `table_scenario_design.csv` (Tabla 3)
- `table_equity_metrics_final.csv` (Tabla 4, con las filas per cápita de `hoja_de_cifras.json`)
- `table_policy_tradeoff.csv` (Tabla 5)
- `table_vulnerable_quartile_provision.csv` (Tabla E1 del suplemento)
- `table_official_ev_growth.csv` (respaldo de la Fig. 3)

Las Tablas 6, 7 y 8 salen de los experimentos A1/A2, B1 y A3 en `experiments/revision_2026/`;
el mapa completo está en `REPRODUCIBILITY.md`.

## Important result note

In the current reproducible run, `S3` is identical to `S2`.
Therefore `S3-S2` net-change maps are structurally empty and are intentionally omitted from the curated package.
