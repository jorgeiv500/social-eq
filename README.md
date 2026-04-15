# Bogotá EV Charging Social Equity Reproduction

Reproducible local pipeline for:

`Socially equitable EV charging allocation in Bogotá: Linking mobility learning and lexicographic optimization`

The repository stages a minimal input bundle, solves scenarios `S0`-`S3` with `Pyomo + NEOS/CPLEX`, runs the rapid robustness battery, and exports the curated paper figures/tables under:

`outputs/social_equity_followup_outputs/paper_v6_experiments`

## Repository layout

- `data/inputs`: versioned inputs required for reproduction.
- `data/reference`: optional reference outputs for comparison.
- `scripts/stage_inputs.py`: stages local inputs from known `Downloads` and synchronized Google Drive paths.
- `scripts/reproduce_followup.py`: runs the full reproduction pipeline.
- `src/social_eq/followup.py`: model logic, figures, tables, cleanup, and paper-package export.
- `docs/jtg_figure_style_notes.md`: visual rules transferred from the reference article.
- `docs/paper_submission_recommendations.md`: recommended main-text vs supplementary figure/table plan.

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

Main paper figures:

- `fig_official_ev_growth.png`
- `fig_social_context_panels.png`
- `fig_attention_profile.png`
- `fig_allocation_composite.png`
- `fig_net_change_S2_minus_S1.png`
- `fig_lorenz_curves.png`
- `fig_policy_tradeoff.png`

Supplementary figures:

- `fig_growth_progression.png`
- `fig_gnn_learning_history.png`
- `fig_robustness.png`

Main paper tables:

- `table_scenario_design.csv`
- `table_equity_metrics_final.csv`
- `table_official_ev_growth.csv`
- `table_policy_tradeoff.csv`

The pipeline also builds:

- `outputs/social_equity_followup_outputs/paper_v6_experiments/paper_submission_package/`
- `outputs/social_equity_followup_outputs/paper_v6_experiments/paper_submission_package.zip`

These contain the curated figures, tables, documentation, and the minimal reproducibility files for submission support.

## Important result note

In the current reproducible run, `S3` is identical to `S2`.
Therefore `S3-S2` net-change maps are structurally empty and are intentionally omitted from the curated package.
