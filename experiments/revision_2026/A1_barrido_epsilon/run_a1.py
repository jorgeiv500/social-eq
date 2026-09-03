#!/usr/bin/env python
"""A1 — Barrido amplio de epsilon sobre el escenario S2 (revisión JTRG-D-26-00692).

Responde R1.4 y alimenta R2.1: extiende la tolerancia de fidelidad epsilon a
{0.15, 0.20, 0.30} (el barrido publicado estaba capado a 0.10 en
`ModelConfig.robustness_epsilon_fast`) y reporta, por corrida, la curva
"tolerancia -> zonas cambiadas -> ΔGini / Δcobertura".

Reutiliza sin duplicar lógica: `solve_equity_refined` (la especificación S2 del
paper: pesos de vulnerabilidad nu, kappa=4, sin piso), `summarize_scenario`,
`extract_solution_refined` y los insumos cargados con `load_minimal_inputs`
(todo desde src/social_eq/followup.py).

SEGURIDAD: por defecto el script hace una corrida en seco (prepara insumos e
imprime el plan SIN enviar nada a NEOS). Solo `--lanzar` envía trabajos.

Uso:
    python run_a1.py                # corrida en seco (sin NEOS)
    python run_a1.py --lanzar       # 3 corridas NEOS: eps 0.15, 0.20, 0.30
    python run_a1.py --lanzar --epsilons 0.15,0.20  # subconjunto
    python run_a1.py --lanzar --recalcular-s0       # +1 corrida NEOS para re-derivar u*

Referencias fijas (corrida oficial del paper, no se recalculan):
- u* (óptimo de fidelidad S0) desde table_priority_summary.csv.
- Asignaciones S1 y S2 desde scenario_S1.csv / scenario_S2.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comun  # noqa: E402
from social_eq.followup import (  # noqa: E402
    extract_solution,
    extract_solution_refined,
    solve_equity_refined,
    solve_s0,
    solve_safe,
    summarize_scenario,
)

AQUI = Path(__file__).resolve().parent
DIR_SALIDAS = AQUI / "salidas"
EPSILONS_PLAN = (0.15, 0.20, 0.30)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A1: barrido amplio de epsilon sobre S2.")
    parser.add_argument(
        "--lanzar",
        action="store_true",
        help="Enviar realmente los trabajos a NEOS. Sin esta bandera solo se imprime el plan.",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(f"{e:.2f}" for e in EPSILONS_PLAN),
        help="Lista de epsilon separada por comas (por defecto la aprobada: 0.15,0.20,0.30).",
    )
    parser.add_argument(
        "--recalcular-s0",
        action="store_true",
        help="Re-derivar u* resolviendo S0 en NEOS (1 corrida extra) en lugar de leer la tabla archivada.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Límite de tiempo del solver en segundos (por defecto ModelConfig.solver_time_limit = 3600).",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Gap MIP relativo (por defecto ModelConfig.solver_mip_gap = 0.01).",
    )
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]
    if any(e <= 0 for e in epsilons):
        raise SystemExit("Todos los epsilon deben ser positivos.")

    print("Preparando insumos (idéntico camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config = ctx.config
    n_zonas = len(ctx.df_inputs)
    n_pares = n_zonas * (n_zonas - 1) // 2

    asignacion_s1 = comun.leer_asignacion_paper("S1")
    asignacion_s2 = comun.leer_asignacion_paper("S2")
    metricas_paper = comun.leer_metricas_paper()

    if args.recalcular_s0 and args.lanzar:
        print("Resolviendo S0 en NEOS para re-derivar u* (corrida extra pedida con --recalcular-s0)...", flush=True)
        modelo_s0, u_star = solve_s0(ctx.target, config=config)
        if not solve_safe(modelo_s0):
            raise SystemExit(f"S0 no resolvió: {getattr(modelo_s0, '_termination_condition', None)}")
        # Se conserva también la asignación S0 por trazabilidad.
        n0, _u0, _ = extract_solution(modelo_s0, n_zonas, len(config.venues))
        pd.DataFrame({"UTAM": ctx.df_inputs["UTAM"], "ports_total": n0.sum(axis=1)}).to_csv(
            DIR_SALIDAS / "tablas" / "asignacion_S0_recalculada.csv", index=False
        )
    else:
        u_star = comun.leer_u_star()

    time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap

    print()
    print("=== Plan del barrido A1 ===")
    print(f"Zonas (UTAM): {n_zonas} | pares de equidad: {n_pares} | variables enteras: {n_zonas * len(config.venues)}")
    print(f"u* de referencia: {u_star!r} (fuente: {'S0 recalculado' if args.recalcular_s0 else comun.TABLAS_PAPER / 'table_priority_summary.csv'})")
    print(f"Especificación por corrida: S2 (nu=vulnerabilidad, kappa={config.s2_kappa}, sin piso), "
          f"cuotas {config.quotas}, timelimit={time_limit}s, mipgap={mip_gap}")
    print(f"Corridas NEOS previstas: {len(epsilons)}" + (" + 1 (S0)" if args.recalcular_s0 else ""))
    for eps in epsilons:
        print(f"  - S2 con epsilon = {eps:.3f}  (restricción lex: suma|n - target| <= {u_star * (1 + eps):.4f})")
    print(f"Correo NEOS que se usará: {comun.email_neos(config)}")
    print()

    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS. Ejecute con --lanzar cuando el runner esté aprobado.")
        return 0

    (DIR_SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    filas_resumen: list[dict] = []
    ruta_resumen = DIR_SALIDAS / "tablas" / "resumen_barrido_epsilon.csv"

    # Filas de contexto desde la corrida oficial (sin costo NEOS): S1 (eps=0.03) y S2 (eps=0.08).
    cambios_s2_vs_s1 = comun.comparar_asignaciones(asignacion_s2, asignacion_s1)
    for escenario, ref_propia, cambios_vs_s1, cambios_vs_s2 in (
        ("S1", asignacion_s1, {"zonas_cambiadas_total": 0, "zonas_cambiadas_sede": 0, "puertos_movidos": 0.0}, comun.comparar_asignaciones(asignacion_s1, asignacion_s2)),
        ("S2", asignacion_s2, cambios_s2_vs_s1, {"zonas_cambiadas_total": 0, "zonas_cambiadas_sede": 0, "puertos_movidos": 0.0}),
    ):
        fila_paper = metricas_paper.loc[metricas_paper["scenario"].astype(str).eq(escenario)].iloc[0]
        filas_resumen.append(
            {
                "fuente": f"archivo_paper_{escenario}",
                "epsilon": float(fila_paper["epsilon_used"]),
                "termination": "archivada",
                "objective_abs_dev": float(fila_paper["objective_abs_dev"]),
                "perdida_fidelidad_vs_S0_pct": 100.0 * (float(fila_paper["objective_abs_dev"]) - u_star) / u_star,
                "gini_no_ponderado_6d": comun.redondear6(fila_paper["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(fila_paper["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(fila_paper["coverage_within_threshold_vulnerable"]),
                "zonas_cambiadas_total_vs_S1": cambios_vs_s1["zonas_cambiadas_total"],
                "zonas_cambiadas_sede_vs_S1": cambios_vs_s1["zonas_cambiadas_sede"],
                "puertos_movidos_vs_S1": cambios_vs_s1["puertos_movidos"],
                "zonas_cambiadas_total_vs_S2": cambios_vs_s2["zonas_cambiadas_total"],
                "puertos_movidos_vs_S2": cambios_vs_s2["puertos_movidos"],
                "rho_prioridad_puertos": float(fila_paper["rho_priority_ports"]),
            }
        )
    pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

    for eps in epsilons:
        etiqueta = f"S2_eps_{eps:.3f}"
        print(f"Lanzando {etiqueta} vía NEOS/CPLEX...", flush=True)
        modelo, _ = solve_equity_refined(
            target=ctx.target,
            coeffs=ctx.coeffs,
            u_star=u_star,
            config=config,
            nu=ctx.vector_vulnerabilidad,
            epsilon=eps,
            kappa=config.s2_kappa,
            floor_mask=None,
            floor_reference=None,
            use_hard_floor=False,
            caps=config.max_ports_matrix,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        terminacion = str(getattr(modelo, "_termination_condition", "desconocida"))
        if not solve_safe(modelo, accept_feasible=True):
            print(f"  ADVERTENCIA: {etiqueta} no entregó solución utilizable (termination={terminacion}). Se registra y se sigue.")
            filas_resumen.append({"fuente": "corrida_nueva", "epsilon": eps, "termination": terminacion})
            pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
            continue

        n_sol, u_sol, a_sol, _ = extract_solution_refined(modelo, n_zonas, len(config.venues))
        assert a_sol is not None
        df_asig, metricas = summarize_scenario(
            etiqueta, ctx.df_inputs, n_sol, u_sol, a_sol, ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config
        )
        ruta_asig = DIR_SALIDAS / "tablas" / f"asignacion_eps_{eps:.3f}.csv"
        df_asig.to_csv(ruta_asig, index=False)

        cambios_s1 = comun.comparar_asignaciones(df_asig, asignacion_s1)
        cambios_s2 = comun.comparar_asignaciones(df_asig, asignacion_s2)
        fila = {
            "fuente": "corrida_nueva",
            "epsilon": eps,
            "termination": terminacion,
            "objective_abs_dev": float(metricas["objective_abs_dev"]),
            "perdida_fidelidad_vs_S0_pct": 100.0 * (float(metricas["objective_abs_dev"]) - u_star) / u_star,
            "gini_no_ponderado_6d": comun.redondear6(metricas["accessibility_gini"]),
            "gini_ponderado_6d": comun.redondear6(metricas["accessibility_weighted_gini"]),
            "cobertura_15min_vulnerables": float(metricas["coverage_within_threshold_vulnerable"]),
            "zonas_cambiadas_total_vs_S1": cambios_s1["zonas_cambiadas_total"],
            "zonas_cambiadas_sede_vs_S1": cambios_s1["zonas_cambiadas_sede"],
            "puertos_movidos_vs_S1": cambios_s1["puertos_movidos"],
            "zonas_cambiadas_total_vs_S2": cambios_s2["zonas_cambiadas_total"],
            "puertos_movidos_vs_S2": cambios_s2["puertos_movidos"],
            "rho_prioridad_puertos": float(metricas["rho_priority_ports"]),
        }
        filas_resumen.append(fila)
        # Escritura incremental: si una corrida NEOS posterior falla, lo anterior sobrevive.
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

        manifiesto = {
            "corrida": etiqueta,
            "epsilon": eps,
            "kappa": config.s2_kappa,
            "u_star": u_star,
            "fuente_u_star": "S0 recalculado" if args.recalcular_s0 else str(comun.TABLAS_PAPER / "table_priority_summary.csv"),
            "time_limit_s": time_limit,
            "mip_gap": mip_gap,
            "termination": terminacion,
            "metricas_precision_completa": {k: v for k, v in metricas.items() if k != "scenario"},
            "comparacion_vs_S1": cambios_s1,
            "comparacion_vs_S2": cambios_s2,
        }
        (DIR_SALIDAS / "tablas" / f"manifiesto_eps_{eps:.3f}.json").write_text(
            json.dumps(manifiesto, indent=2, default=float), encoding="utf-8"
        )
        print(f"  {etiqueta}: termination={terminacion}; tablas en {ruta_asig.name} y {ruta_resumen.name}")

    print(f"\nBarrido terminado. Resumen consolidado: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
