# Bogotá EV Charging Social Equity Reproduction

Repositorio para replicar en local el experimento del paper:

`Socially equitable EV charging allocation in Bogotá: Linking mobility learning and lexicographic optimization`

El pipeline toma un conjunto mínimo de insumos versionados en `data/inputs`, reproduce los escenarios `S0`-`S3`, ejecuta la batería rápida de robustez y exporta las figuras/tablas del manuscrito en `outputs/social_equity_followup_outputs/paper_v6_experiments`.

La optimización se ejecuta por defecto en `NEOS` usando `CPLEX`, siguiendo la lógica del cuaderno original.

## Estructura

- `data/inputs`: insumos mínimos para la reproducción principal.
- `data/reference`: opcional; salidas de referencia para comparación.
- `scripts/stage_inputs.py`: copia insumos desde rutas locales conocidas (`Downloads` y Google Drive sincronizado).
- `scripts/reproduce_followup.py`: ejecuta la reproducción completa.
- `src/social_eq/followup.py`: lógica del modelo, tablas y figuras.

## Uso rápido

1. Crear o activar el entorno virtual.
2. Instalar el proyecto:

```bash
.venv/bin/pip install -e .
```

3. Preparar insumos locales:

```bash
.venv/bin/python scripts/stage_inputs.py
```

4. Ejecutar la reproducción:

```bash
.venv/bin/python scripts/reproduce_followup.py
```

Si quieres sobrescribir el correo para `NEOS`, exporta antes:

```bash
export NEOS_EMAIL="tu_correo@dominio.com"
```

## Salidas principales

El script genera, entre otros:

- `fig_priority_map.png`
- `fig_gnn_learning_history.png`
- `fig_attention_profile.png`
- `fig_allocation_S0.png`
- `fig_allocation_S1.png`
- `fig_allocation_S2.png`
- `fig_allocation_S3.png`
- `fig_net_change_S2_minus_S1.png`
- `fig_net_change_S3_minus_S2.png`
- `fig_S3_protected_set.png`
- `fig_lorenz_curves.png`
- `fig_robustness.png`

y las tablas:

- `table_gnn_performance_available.csv`
- `table_priority_summary.csv`
- `table_allocation_changes_S2_vs_S1.csv`
- `table_equity_metrics.csv`
- `table_equity_metrics_final.csv`
- `table_robustness_fast.csv`
- `table_robustness.csv`

## Notas

- La optimización se envía a `NEOS/CPLEX` vía `Pyomo`.
- `scripts/stage_inputs.py` además puede copiar una carpeta de referencia desde Google Drive sincronizado para contrastar resultados.
- Si faltan insumos, revisa `data/inputs/input_manifest.json` después del staging.
- La carpeta `outputs/social_equity_followup_outputs/reference_comparison.json` resume la distancia entre las tablas reproducidas y la carpeta de referencia.
- En este entorno no fue posible reconstruir la matriz alternativa vía OSRM por un problema de SSL del cliente local; por eso la corrida actual usa `survey_graph_completed` como cierre de la matriz OD y no reproduce exactamente todas las métricas de equidad de la referencia.
