# A1 — Barrido amplio de epsilon (0.15, 0.20, 0.30)

**Responde:** R1.4 (ampliar la tolerancia de fidelidad) y alimenta R2.1 con la curva
"tolerancia → nº de zonas cambiadas → ΔGini / Δcobertura".
**Estado: COMPLETADO (2026-09-01)** — 3 corridas NEOS (ε = 0.15, 0.20, 0.30), las tres `optimal`; resultados en `salidas/tablas/resumen_barrido_epsilon.csv` e integrados en la Tabla 6 y §5.5 del manuscrito (ver `../../TABLERO.md`).

## Qué hace

Resuelve la especificación **S2 del paper** (pesos de vulnerabilidad ν, κ = 4.0, sin piso,
cuotas 7,352 / 2,739 / 779 — valores de `ModelConfig` en `src/social_eq/followup.py`) con la
restricción lexicográfica relajada a ε ∈ {0.15, 0.20, 0.30}. El barrido publicado estaba capado
a 0.10 por la propiedad `robustness_epsilon_fast` (líneas 147–154 de `followup.py`); la función
`solve_equity_refined(..., epsilon=...)` acepta valores arbitrarios, así que no hay impedimento
de código.

Referencias que **no** se recalculan (salen de la corrida oficial archivada):

| Referencia | Fuente |
|---|---|
| u\* = 62.8284379878768 (óptimo de fidelidad S0) | `outputs/social_equity_followup_outputs/paper_v6_experiments/tables/table_priority_summary.csv`, fila S0, columna `objective_abs_dev` (en `summarize_scenario` esa métrica es exactamente el óptimo lexicográfico) |
| Asignación S1 (ε = 0.03) para "zonas cambiadas vs S1" | `.../tables/scenario_S1.csv` |
| Asignación S2 (ε = 0.08) como punto existente de la curva | `.../tables/scenario_S2.csv` |
| Métricas archivadas S1/S2 (filas de contexto del resumen) | `.../tables/table_equity_metrics_final.csv` |

Si se prefiere re-derivar u\* con una corrida S0 nueva: `--recalcular-s0` (1 corrida NEOS extra).

## Comando exacto

```bash
cd /Users/jorgeiv500/Documents/social-eq
export NEOS_EMAIL="correo_del_autor@dominio"   # ver nota abajo

# 1) Corrida en seco (sin NEOS): valida insumos e imprime el plan
.venv/bin/python experiments/revision_2026/A1_barrido_epsilon/run_a1.py

# 2) Lanzamiento real (3 corridas NEOS)
.venv/bin/python experiments/revision_2026/A1_barrido_epsilon/run_a1.py --lanzar
```

Sin `--lanzar` el script **nunca** envía trabajos a NEOS.

## Corridas que lanza

| # | Corrida | ε | Especificación |
|---|---|---|---|
| 1 | `S2_eps_0.150` | 0.15 | S2 (ν, κ=4, sin piso), timelimit 3600 s, mipgap 0.01 |
| 2 | `S2_eps_0.200` | 0.20 | ídem |
| 3 | `S2_eps_0.300` | 0.30 | ídem |

Total: **3 corridas NEOS** (4 con `--recalcular-s0`). Los parámetros de solver son los de la
especificación principal del paper (`solver_time_limit = 3600`, `solver_mip_gap = 0.01` en
`ModelConfig`), no los rebajados de la batería de robustez (180 s / 0.02).

## Salidas (en `salidas/tablas/`)

- `asignacion_eps_<ε>.csv` — asignación por UTAM (puertos por sede y total, accesibilidad,
  vulnerabilidad), una por corrida.
- `resumen_barrido_epsilon.csv` — una fila por corrida **más** dos filas de contexto (S1 y S2
  archivadas), con: `objective_abs_dev`, pérdida de fidelidad vs S0 (%), Gini no ponderado y
  ponderado **a 6 decimales** (R1.13/R1.14), cobertura a 15 min de UTAM vulnerables
  (cuartil superior de vulnerabilidad, umbral `coverage_threshold_min = 15.0`), nº de zonas
  cambiadas vs S1 (por total y por sede), puertos movidos, y lo mismo vs S2. Se escribe
  incrementalmente tras cada corrida.
- `manifiesto_eps_<ε>.json` — métricas a precisión completa, condición de terminación,
  parámetros y fuentes de referencia de cada corrida.

## Tiempo estimado por corrida NEOS

El repositorio **no registra tiempos de pared por corrida**, así que solo puede acotarse con lo
que hay:

- Cota superior de solver: `solver_time_limit = 3600 s` por corrida (`ModelConfig`).
- Evidencia previa: la batería de robustez resolvió esta misma especificación S2 con
  ε ∈ {0.056, 0.080, 0.100} bajo un límite de **180 s** y mipgap 0.02, y las tres corridas
  terminaron con solución (`outputs/.../tables/table_robustness.csv`, filas "Alternative
  epsilon tolerance"). El modelo es de la misma escala aquí: 142 UTAM → 10,011 pares de
  equidad, 426 variables enteras.
- A eso se suma la cola de NEOS, variable y no documentada en el repo.

Estimación honesta: **minutos por corrida** en condiciones normales, con techo duro de 1 h de
solver + cola NEOS. Presupuesto prudente del experimento completo: ≤ 3 h. Con ε grandes el
espacio factible crece y el MIP puede tardar más que en el barrido capado; si una corrida
agota el límite, `solve_safe(..., accept_feasible=True)` acepta la incumbente y la condición
de terminación queda registrada en el manifiesto JSON.

## Notas

- `NEOS_EMAIL`: `solve_model` (líneas 514–516 de `followup.py`) usa la variable de entorno y,
  si no está definida, cae al valor por defecto `ModelConfig.neos_email`
  (`jorgei.romerog@utadeo.edu.co`). Exportarla explícitamente antes de lanzar.
- La comparación "vs S1" usa la asignación archivada del paper (misma semilla de datos,
  entradas deterministas); no se relanza S1.
- La corrida a ε = 0.08 no se repite: ya existe como S2 oficial y entra al resumen como fila
  de contexto.
