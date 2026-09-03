# Reproducibility guide

Companion to the revised manuscript *Public electric-vehicle charging and accessibility
inequality in Bogotá: A socio-spatial allocation framework* (Journal of Transport Geography,
manuscript JTRG-D-26-00692, revision submitted September 2026).

Every number, table and figure in the manuscript and its supplement is produced by code in this
repository from the versioned inputs in `data/inputs/`. This guide maps each reported item to
the file that holds its values and to the command that regenerates it, and states which steps
need resources that cannot be redistributed (the NEOS optimization server and the
mobility-survey microdata). Nothing in the manuscript is typed by hand: the sheet produced by
`scripts/hoja_de_cifras.py` lists every cited figure next to its source file.

## 1. Correction notice

During the revision we found that the notebook used to train the spatio-temporal graph neural
network for the April 2026 submission matched zone codes with leading zeros (`UTAM043`) against
codes without them (`UTAM43`). The training label was therefore zero in 98 of the 142 zones, the
survey expansion factor was not applied, and comma-decimal weights were coerced to zero. The
model was retrained with the matching corrected (`scripts/retrain_stgnn.py`, variant
`fexp_miles`), and every downstream result was recomputed: the allocation pipeline
(16 NEOS runs) and the seven solver-based experiments (30 NEOS runs), 46 runs in total, all
terminating at optimality. The superseded inputs are kept for the record in
`data/inputs/superseded_2026-04/` with a README, and `data/inputs/input_manifest.json` records
the replacement under the key `superseded`. The retraining outputs (priority scores, temporal
attention, loss history, split metrics, seed stability and the run protocol) are archived in
`outputs/stgnn_retraining/peso_fexp_miles/`.

## 2. Environment

All runs used Python 3.9.6 on macOS. The exact package set is frozen in `requirements-lock.txt`
(`torch 2.8.0` and `torch-geometric 2.6.1` are needed only for retraining).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .
```

The optimization model is solved with CPLEX on the NEOS server through Pyomo. NEOS requires an
e-mail address, read from the `NEOS_EMAIL` environment variable. Every NEOS run writes a
manifest with termination status, objective value, MIP gap and time limit next to its
allocation table.

## 3. Levels of reproduction

| Level | What it regenerates | External resources | Typical duration |
| --- | --- | --- | --- |
| 0 | All figures and derived tables from the archived allocations and experiment outputs | none | about 20 min (map rendering dominates) |
| 1 | Experiments that need no solver: static baselines (B1), grid-capacity diagnostic (A4), rental-tenure check (B3), trip-duration evidence (E6) | B3 needs the EM-2021 raw files; E6 the EM-2023 trip table (see Section 6) | minutes |
| 2 | The allocation pipeline (S0 to S3 plus the robustness battery, 16 NEOS runs) and the solver experiments A1, A2, A3, B2, B4, C1 (30 NEOS runs) | NEOS account and network access | pipeline: see `outputs/social_equity_followup_outputs/run_manifest.json`; experiments: 6 min (A1, three runs; the five frontier runs took a further 12 min), 1.5 min (A2), 8 min (A3), 3.5 min (B2), 4 min (C1), 10 min (B4) in our runs, dominated by the NEOS queue |
| 3 | Retraining of the ST-GNN | EM-2023 microdata (Section 6) | minutes on CPU (best epoch 119 of at most 1,200, early stopping) |

`scripts/reproduce_all.sh LEVEL` runs the levels in order up to `LEVEL` (default 0). Levels 2
and 3 refuse to start unless `NEOS_EMAIL` or `EM2023_DIR` are set, so nothing is submitted to
NEOS by accident. Note that `scripts/reproduce_followup.py` has no command-line options and
launches the NEOS runs as soon as it is invoked; use `scripts/run_pipeline_resiliente.py`,
which adds a socket timeout and retries, when re-solving deliberately.

## 4. Manuscript items and their sources

Paths under `outputs/` are relative to
`outputs/social_equity_followup_outputs/paper_v6_experiments/`; paths under `experiments/` are
relative to `experiments/revision_2026/`.

### Main text

| Item | File with the values | Produced by |
| --- | --- | --- |
| Fig. 1 (methodology) | TikZ diagram in the LaTeX source; no data | — |
| Fig. 2 (study area, existing chargers) | `experiments/E3_mapas/salidas/fig_study_area_context.png` (caption text in `..._caption.txt`) | `E3_mapas/regenerar_figuras.py`, charger points from `E3_mapas/datos/` |
| Fig. 3 (EV registrations) | `outputs/tables/table_official_ev_growth.csv` from `data/inputs/bogota_official_ev_growth.csv` | `E7_figuras_impresion/regenerar_todas.py` |
| Fig. 4 (priority, vulnerability, protected set) | `data/inputs/priority_utam_scores.csv`, `data/inputs/utam_social_vulnerability.csv`, `outputs/tables/scenario_S3.csv` | E7 (chart variant) and E3 (map variant) |
| Fig. 5 (temporal attention) | `data/inputs/temporal_attention_by_utam.csv` | E7 |
| Fig. 6 (allocation maps S0 to S3) | `outputs/tables/scenario_S0.csv` to `scenario_S3.csv` | E7 and E3 |
| Fig. 7 (net reallocation S2 minus S1) | `outputs/tables/scenario_S1.csv`, `scenario_S2.csv`; summary in `table_allocation_changes_S2_vs_S1.csv` | E7 and E3 |
| Fig. 8 (Lorenz curves) | `outputs/tables/scenario_S0.csv` to `scenario_S3.csv` | E7 |
| Fig. 9 (policy trade-off synthesis) | `outputs/tables/table_policy_tradeoff.csv`, `table_robustness.csv`, `table_equity_metrics_final.csv`; overlays from `experiments/A1_barrido_epsilon/salidas/tablas/resumen_barrido_epsilon.csv` and `experiments/A2_benchmark_equidad/salidas/tablas/resumen_benchmark.csv` | E7 |
| Table 3 (scenario design) | `outputs/tables/table_scenario_design.csv` | pipeline |
| Table 4 (accessibility and equity metrics; ports per 10,000 residents) | `outputs/tables/table_equity_metrics_final.csv`; per-capita rows in `outputs/tables/table_provision_intensity.csv`; coverage at 5, 10, 15, 20 and 30 minutes in `outputs/tables/table_coverage_thresholds.csv` | pipeline, `scripts/intervention_size.py`, `scripts/coverage_thresholds.py` |
| Table 5 (compact trade-off) | `outputs/tables/table_policy_tradeoff.csv` | pipeline |
| Table 6 (tolerance sweep, frontier runs and direct-equity benchmark) | `experiments/A1_barrido_epsilon/salidas/tablas/resumen_barrido_epsilon.csv` (ε = 0.15–100), `percapita_por_epsilon.csv` (provision per capita along the frontier, cited in Section 5.5 and Fig. 9b), `experiments/A2_benchmark_equidad/salidas/tablas/resumen_benchmark.csv` | `run_a1.py --lanzar` (published sweep) and `run_a1.py --lanzar --epsilons 1.00,3.00,10.00,30.00,100.00` (frontier), `run_a2.py --lanzar` |
| Table 7 (static baselines vs the ST-GNN) | `experiments/B1_baselines/resultados/b1_metricas_cv.csv`, `b1_spearman_prioridad.csv`, `b1_diagnostico_insumos.csv` | `run_baselines.py` |
| Table 8 (projected growth) | `experiments/A3_escenarios_crecimiento/salidas/tablas/resumen_crecimiento.csv`, `derivacion_cuotas.csv` | `run_a3.py --lanzar` |
| Grid-capacity caps (text) | `experiments/A4_topes_capacidad/salidas/tablas/resumen_diagnostico.json`, `diagnostico_topes.csv` | `preparar_topes.py`, then `run_a4.py` |
| Venue-weight ablation (text) | `experiments/B2_ablacion_omega/salidas/tablas/resumen_ablacion_omega.csv` | `run_b2.py --lanzar` |
| Rental tenure (text) | `experiments/B3_tenencia/salidas/tabla_correlaciones.csv` | `codigo/analisis_tenencia.py` |
| Priority counterfactual (text) | `experiments/B4_contrafactual_prioridad/salidas/tablas/resumen_contrafactual.csv` | `run_b4.py --lanzar` |
| Binding minimum-service floors (text) | `experiments/C1_piso_vinculante/salidas/tablas/resumen_piso_vinculante.csv` | `run_c1.py --lanzar` |
| Size of the intervention (Section 5.3: 947 and 1,902 ports, 8.7\% and 17.5\% of the quota) | `outputs/tables/table_provision_intensity.csv` (one row per scenario) and `outputs/tables/table_vulnerable_quartile_provision.csv` (zone by zone) | `scripts/intervention_size.py`, `scripts/make_vulnerable_quartile_table.py` |

Tables 1 and 2 are narrative (literature streams and dataset roles) and have no computed
values.

### Supplement

| Item | File with the values | Produced by |
| --- | --- | --- |
| Fig. C1 (growth progression) | `outputs/tables/table_official_ev_growth.csv` | E7 |
| Fig. C2 (robustness battery) | `outputs/tables/table_robustness.csv` | E7 |
| Appendix D (trip durations by purpose) | `experiments/E6_lambdas/salidas/tablas/duracion_por_proposito.csv` | `run_e6.py` (needs the EM-2023 trip table) |
| Table E1 (vulnerable-quartile provision by zone) | `outputs/tables/table_vulnerable_quartile_provision.csv` | `scripts/make_vulnerable_quartile_table.py` |

## 5. Commands by level

Level 0, figures and derived tables (no solver):

```bash
.venv/bin/python experiments/revision_2026/E7_figuras_impresion/regenerar_todas.py
.venv/bin/python experiments/revision_2026/E3_mapas/regenerar_figuras.py
.venv/bin/python scripts/make_vulnerable_quartile_table.py
.venv/bin/python scripts/intervention_size.py
.venv/bin/python scripts/coverage_thresholds.py
.venv/bin/python scripts/hoja_de_cifras.py
```

Level 1, experiments without a solver:

```bash
(cd experiments/revision_2026/B1_baselines && ../../../.venv/bin/python run_baselines.py)
(cd experiments/revision_2026/A4_topes_capacidad && ../../../.venv/bin/python preparar_topes.py && ../../../.venv/bin/python run_a4.py)
(cd experiments/revision_2026/B3_tenencia && ../../../.venv/bin/python codigo/analisis_tenencia.py)   # needs datos_fuente/, see Section 6
EM2023_VIAJES=/path/to/viajes.csv .venv/bin/python experiments/revision_2026/E6_lambdas/run_e6.py
```

Level 2, re-solving with NEOS (submits jobs under your e-mail):

```bash
export NEOS_EMAIL="you@example.org"
.venv/bin/python scripts/run_pipeline_resiliente.py          # S0 to S3 and the robustness battery
for e in A1_barrido_epsilon A2_benchmark_equidad A3_escenarios_crecimiento B2_ablacion_omega B4_contrafactual_prioridad C1_piso_vinculante; do
  (cd experiments/revision_2026/$e && ../../../.venv/bin/python run_*.py --lanzar)
done
```

Level 3, retraining the ST-GNN (writes to `outputs/stgnn_retraining/peso_fexp_miles/`):

```bash
EM2023_DIR=/path/to/em2023 .venv/bin/python scripts/retrain_stgnn.py --peso-viajes fexp_miles
```

After retraining, copy `priority_utam_scores.csv`, `temporal_attention_by_utam.csv` and
`learning_loss_history.csv` from that folder into `data/inputs/` before re-solving.

## 6. Data that is not redistributed

- **Encuesta de Movilidad 2023 (EM-2023) microdata**, Secretaría Distrital de Movilidad de
  Bogotá: needed for retraining (`EM2023_DIR`) and for Appendix D (`EM2023_VIAJES`, the trip
  module as a semicolon-separated CSV). The derived inputs that the allocation model actually
  consumes are versioned in `data/inputs/`.
- **Encuesta Multipropósito 2021 raw files** for the rental-tenure check (B3): the download
  links are in `experiments/revision_2026/B3_tenencia/README.md`; place the files in
  `experiments/revision_2026/B3_tenencia/datos_fuente/`. The derived zone table
  (`salidas/utam_tenencia.csv`) is included, so the correlations can be verified without the raw
  files.
- **Electrical capacity layer** (A4) and **existing charger points** (Fig. 2): small extracts
  are included under `A4_topes_capacidad/datos/` and `E3_mapas/datos/` with a manifest that
  records the source URL, the query and the download date.

## 7. Checking that a reproduction matches the manuscript

1. Run `scripts/intervention_size.py`, `scripts/coverage_thresholds.py` and
   `scripts/hoja_de_cifras.py` in that order, then compare `outputs/.../hoja_de_cifras.md`
   with the values in the text; the JSON keeps full precision and names the source of each
   entry, including the per-capita block and the coverage thresholds.
2. Compare `outputs/.../tables/table_equity_metrics_final.csv` with the archived copy in git
   history; allocations from NEOS/CPLEX are deterministic at the reported MIP gap (0.01), but a
   different CPLEX version can return an alternative optimum with the same objective. The
   experiment READMEs state which conclusions depend on ties.
3. Each experiment folder has a README (in Spanish) with the question it answers, the exact
   settings, the outputs and the interpretation used in the manuscript.
