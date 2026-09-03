#!/usr/bin/env bash
# Reproduce por niveles (ver REPRODUCIBILITY.md):
#   0 = figuras y tablas derivadas desde las asignaciones archivadas (sin solver)
#   1 = + experimentos sin solver (B1, A4, B3 y E6 cuando sus datos están disponibles)
#   2 = + pipeline de asignación y experimentos con NEOS/CPLEX (requiere NEOS_EMAIL)
#   3 = + reentrenamiento del ST-GNN (requiere EM2023_DIR)
# Uso: scripts/reproduce_all.sh [NIVEL]
set -euo pipefail
NIVEL="${1:-0}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PY="$RAIZ/.venv/bin/python"
EXP="$RAIZ/experiments/revision_2026"
cd "$RAIZ"

if [ "$NIVEL" -ge 3 ]; then
  : "${EM2023_DIR:?Nivel 3: defina EM2023_DIR con la carpeta de la EM2023 (no se redistribuye)}"
  echo "== Nivel 3: reentrenamiento del ST-GNN (variante fexp_miles)"
  "$PY" scripts/retrain_stgnn.py --peso-viajes fexp_miles
  cp outputs/stgnn_retraining/peso_fexp_miles/{priority_utam_scores,temporal_attention_by_utam,learning_loss_history}.csv data/inputs/
fi

if [ "$NIVEL" -ge 2 ]; then
  : "${NEOS_EMAIL:?Nivel 2: defina NEOS_EMAIL; las corridas se envían al servidor NEOS bajo ese correo}"
  echo "== Nivel 2: pipeline de asignación (16 corridas NEOS)"
  "$PY" scripts/run_pipeline_resiliente.py
  for e in A1_barrido_epsilon A2_benchmark_equidad A3_escenarios_crecimiento B2_ablacion_omega B4_contrafactual_prioridad C1_piso_vinculante; do
    echo "== Nivel 2: experimento $e"
    (cd "$EXP/$e" && "$PY" run_*.py --lanzar)
  done
fi

if [ "$NIVEL" -ge 1 ]; then
  echo "== Nivel 1: baselines estáticos (B1)"
  (cd "$EXP/B1_baselines" && "$PY" run_baselines.py)
  echo "== Nivel 1: topes de capacidad (A4)"
  (cd "$EXP/A4_topes_capacidad" && "$PY" preparar_topes.py && "$PY" run_a4.py)
  if [ -d "$EXP/B3_tenencia/datos_fuente" ]; then
    echo "== Nivel 1: tenencia en arriendo (B3)"
    (cd "$EXP/B3_tenencia" && "$PY" codigo/analisis_tenencia.py)
  else
    echo "-- B3 omitido: falta $EXP/B3_tenencia/datos_fuente (ver README de B3)"
  fi
  if [ -n "${EM2023_VIAJES:-}" ] && [ -f "$EM2023_VIAJES" ]; then
    echo "== Nivel 1: duraciones por propósito (E6)"
    "$PY" "$EXP/E6_lambdas/run_e6.py"
  else
    echo "-- E6 omitido: defina EM2023_VIAJES con la tabla de viajes de la EM2023"
  fi
fi

echo "== Nivel 0: figuras y tablas derivadas"
"$PY" "$EXP/E7_figuras_impresion/regenerar_todas.py"
"$PY" "$EXP/E3_mapas/regenerar_figuras.py"
"$PY" scripts/make_vulnerable_quartile_table.py
"$PY" scripts/intervention_size.py
"$PY" scripts/coverage_thresholds.py
"$PY" scripts/hoja_de_cifras.py
echo "Listo. Figuras: $EXP/E7_figuras_impresion/salidas y $EXP/E3_mapas/salidas"
