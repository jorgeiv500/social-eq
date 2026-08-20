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

El material intermedio y las copias superadas quedaron en
`outputs/social_equity_followup_outputs/paper_v6_experiments/_archivo_superado/`
(fuera del control de versiones).

## Repository layout

- `data/inputs`: versioned inputs required for reproduction.
- `data/reference`: optional reference outputs for comparison.
- `scripts/stage_inputs.py`: stages local inputs from known `Downloads` and synchronized Google Drive paths.
- `scripts/reproduce_followup.py`: runs the full reproduction pipeline.
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

Figuras del texto principal (7 de las 8; la figura 2 es un diagrama TikZ dentro del LaTeX):

- `fig_official_ev_growth.png` (Fig. 1)
- `fig_social_context_panels.png` (Fig. 3)
- `fig_attention_profile.png` (Fig. 4)
- `fig_allocation_composite.png` (Fig. 5)
- `fig_net_change_S2_minus_S1.png` (Fig. 6)
- `fig_lorenz_curves.png` (Fig. 7)
- `fig_policy_tradeoff.png` (Fig. 8, cuatro paneles)

Figuras del suplemento:

- `fig_growth_progression.png` (Fig. S1)
- `fig_robustness.png` (Fig. S2)

Figuras generadas como respaldo, no incluidas en el envío:

- `fig_gnn_learning_history.png`, `fig_priority_map.png`, `fig_S3_protected_set.png`,
  `fig_allocation_S0.png`–`fig_allocation_S3.png`

Tablas del texto principal:

- `table_scenario_design.csv` (Tabla 2)
- `table_equity_metrics_final.csv` (Tabla 3)
- `table_policy_tradeoff.csv` (Tabla 4)
- `table_official_ev_growth.csv` (respaldo de la Fig. 1)

## Important result note

In the current reproducible run, `S3` is identical to `S2`.
Therefore `S3-S2` net-change maps are structurally empty and are intentionally omitted from the curated package.
