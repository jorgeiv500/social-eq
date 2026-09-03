#!/usr/bin/env python
"""A2 — Benchmark de equidad directa (formulación CERRADA 2026-09-01, ver README).

Responde R1.6: ¿cuánta equidad adicional se obtiene si se optimiza la equidad
directamente, sujeta SOLO a las cuotas, y cuánta fidelidad a la prioridad del
ST-GNN se sacrifica a cambio? El modelo minimiza la suma ponderada de
diferencias absolutas por pares de la accesibilidad A_i — la misma forma de
objetivo de equidad de S1/S2 del paper — sin restricción lexicográfica, sin
target de prioridad y sin ε.

Decisiones de formulación (cerradas; razones en README.md de esta carpeta):
1. Objetivo: linealización L1 por pares (min Σ_p w_p·z_p). El QP queda
   descartado y declarado: la ruta Pyomo→NEOS de este repo solo se ha
   ejercitado con LP/MILP.
2. Variantes: DOS corridas con datos completos —
   (a) no_ponderado: w_p = 1 (comparable con el Gini no ponderado publicado);
   (b) ponderado_k4: w_p = 1 + κ(ν_i+ν_j)/2 con κ = ModelConfig.s2_kappa = 4
       y ν = vulnerability_index (los mismos pesos de par de S2).
   La variante per cápita NO se corre: 22/142 UTAM sin `dane_population`
   utilizable y no se imputa (limitación declarada).
3. Topes: los mismos `caps = config.max_ports_matrix` del pipeline (hoy None
   → sin topes, igual que la corrida publicada). Sin topes nuevos sin dato.
4. Enteros: n_iv ∈ Z⁺, mismo dominio que el modelo publicado.

Reutiliza de src/social_eq/followup.py: carga de insumos, coeficientes de
accesibilidad, envío a NEOS (`solve_model`), chequeo de terminación
(`solve_safe`) y métricas (`summarize_scenario`). El modelo Pyomo de este
archivo es la única pieza nueva, porque A2 ES una formulación nueva.

SEGURIDAD: sin `--lanzar` no se envía nada a NEOS.

Uso:
    python run_a2.py                 # corrida en seco (sin NEOS)
    python run_a2.py --lanzar        # 2 corridas NEOS (no_ponderado y ponderado_k4)
    python run_a2.py --lanzar --variantes ponderado_k4   # subconjunto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    minimize,
    value,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comun  # noqa: E402
from social_eq.followup import solve_model, solve_safe, summarize_scenario  # noqa: E402

AQUI = Path(__file__).resolve().parent
DIR_SALIDAS = AQUI / "salidas"
VARIANTES_PLAN = ("no_ponderado", "ponderado_k4")


def pesos_de_pares(
    variante: str,
    pares: list[tuple[int, int]],
    nu: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Pesos w_p por par según la variante (mismas fórmulas que solve_equity*).

    - no_ponderado: w_p = 1 (como solve_equity con nu=None, la forma de S1).
    - ponderado_k4: w_p = 1 + κ(ν_i+ν_j)/2 (idéntico a solve_equity_refined,
      la forma de S2, con κ = ModelConfig.s2_kappa).
    """
    if variante == "no_ponderado":
        return np.ones(len(pares), dtype=float)
    if variante == "ponderado_k4":
        return np.array(
            [1.0 + float(kappa) * (float(nu[i]) + float(nu[j])) / 2.0 for i, j in pares],
            dtype=float,
        )
    raise SystemExit(f"Variante desconocida: {variante!r} (válidas: {', '.join(VARIANTES_PLAN)}).")


def construir_benchmark(
    coeffs: dict,
    config,
    n_zonas: int,
    pesos: np.ndarray,
    caps: np.ndarray | None,
) -> tuple[ConcreteModel, list[tuple[int, int]]]:
    """Modelo L1 por pares sujeto SOLO a cuotas (+ caps del pipeline si existen)."""
    pares = [(i, j) for i in range(n_zonas) for j in range(i + 1, n_zonas)]

    modelo = ConcreteModel()
    modelo.I = RangeSet(0, n_zonas - 1)
    modelo.V = RangeSet(0, len(config.venues) - 1)
    modelo.P = RangeSet(0, len(pares) - 1)
    modelo.n = Var(modelo.I, modelo.V, domain=NonNegativeIntegers)
    modelo.A = Var(modelo.I, domain=NonNegativeReals)
    modelo.z = Var(modelo.P, domain=NonNegativeReals)

    # Única restricción de recursos: las cuotas por sede (sin acople a prioridad).
    modelo.quota = Constraint(
        modelo.V,
        rule=lambda mdl, v: sum(mdl.n[i, v] for i in mdl.I) == int(config.quotas[config.venues[v]]),
    )
    # Decisión 3: mismos topes del pipeline (config.max_ports_matrix; hoy None → sin topes).
    if caps is not None:
        modelo.cap = Constraint(
            modelo.I, modelo.V, rule=lambda mdl, i, v: mdl.n[i, v] <= float(caps[i, v])
        )

    # Misma definición de accesibilidad del paper (coeficientes precalculados).
    def regla_accesibilidad(mdl: ConcreteModel, i: int):
        expr = 0.0
        for indice_sede, sede in enumerate(config.venues):
            expr += sum(coef * mdl.n[j, indice_sede] for j, coef in coeffs[sede][i])
        return mdl.A[i] == expr

    modelo.Adef = Constraint(modelo.I, rule=regla_accesibilidad)

    # Linealización estándar de |A_i − A_j| (idéntica a solve_equity/solve_equity_refined).
    modelo.g1 = Constraint(
        modelo.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pares[p][0]] - mdl.A[pares[p][1]]
    )
    modelo.g2 = Constraint(
        modelo.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pares[p][1]] - mdl.A[pares[p][0]]
    )

    modelo.obj = Objective(
        expr=sum(float(pesos[p]) * modelo.z[p] for p in modelo.P), sense=minimize
    )
    return modelo, pares


def fila_contexto_paper(
    escenario: str,
    metricas_paper: pd.DataFrame,
    u_star: float,
    cambios_vs_s1: dict,
    cambios_vs_s2: dict,
) -> dict:
    """Fila de contexto con las métricas archivadas del paper (sin costo NEOS)."""
    fila = metricas_paper.loc[metricas_paper["scenario"].astype(str).eq(escenario)].iloc[0]
    return {
        "fuente": f"archivo_paper_{escenario}",
        "variante": escenario,
        "termination": "archivada",
        "objetivo_L1_pares": float("nan"),
        "objective_abs_dev": float(fila["objective_abs_dev"]),
        "perdida_fidelidad_vs_S0_pct": 100.0 * (float(fila["objective_abs_dev"]) - u_star) / u_star,
        "gini_no_ponderado_6d": comun.redondear6(fila["accessibility_gini"]),
        "gini_ponderado_6d": comun.redondear6(fila["accessibility_weighted_gini"]),
        "cobertura_15min_vulnerables": float(fila["coverage_within_threshold_vulnerable"]),
        "rho_prioridad_puertos": float(fila["rho_priority_ports"]),
        "zonas_cambiadas_total_vs_S1": cambios_vs_s1["zonas_cambiadas_total"],
        "zonas_cambiadas_sede_vs_S1": cambios_vs_s1["zonas_cambiadas_sede"],
        "puertos_movidos_vs_S1": cambios_vs_s1["puertos_movidos"],
        "zonas_cambiadas_total_vs_S2": cambios_vs_s2["zonas_cambiadas_total"],
        "puertos_movidos_vs_S2": cambios_vs_s2["puertos_movidos"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A2: benchmark de equidad directa (formulación cerrada).")
    parser.add_argument("--lanzar", action="store_true", help="Enviar realmente los trabajos a NEOS.")
    parser.add_argument(
        "--variantes",
        default=",".join(VARIANTES_PLAN),
        help="Variantes separadas por comas (por defecto las dos aprobadas: no_ponderado,ponderado_k4).",
    )
    parser.add_argument("--time-limit", type=int, default=None, help="Segundos de solver (defecto 3600 del config).")
    parser.add_argument("--mip-gap", type=float, default=None, help="Gap MIP (defecto 0.01 del config).")
    args = parser.parse_args()

    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()]
    for v in variantes:
        if v not in VARIANTES_PLAN:
            raise SystemExit(f"Variante desconocida: {v!r} (válidas: {', '.join(VARIANTES_PLAN)}).")

    print("Preparando insumos (mismo camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config = ctx.config
    n_zonas = len(ctx.df_inputs)
    n_pares = n_zonas * (n_zonas - 1) // 2

    u_star = comun.leer_u_star()
    asignacion_s1 = comun.leer_asignacion_paper("S1")
    asignacion_s2 = comun.leer_asignacion_paper("S2")
    metricas_paper = comun.leer_metricas_paper()

    time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap
    caps = config.max_ports_matrix  # Decisión 3: hoy None → sin topes, como el pipeline.

    print()
    print("=== Plan del benchmark A2 (formulación cerrada) ===")
    print(f"Zonas: {n_zonas} | pares: {n_pares} | variables enteras: {n_zonas * len(config.venues)}")
    print(f"Cuotas: {config.quotas} (única restricción de recursos; sin lex, sin target, sin ε)")
    print(f"Topes (caps): {'None → sin topes, como el pipeline publicado' if caps is None else 'matriz del config'}")
    print(f"Variantes: {', '.join(variantes)} (κ de la ponderada: {config.s2_kappa}, ν = vulnerability_index)")
    print(f"Solver: NEOS/CPLEX, timelimit={time_limit}s, mipgap={mip_gap} | correo: {comun.email_neos(config)}")
    print(f"Fidelidad ex post contra u* = {u_star!r} (tabla archivada del paper)")
    print(f"Corridas NEOS previstas: {len(variantes)}")
    print()

    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS. Ejecute con --lanzar cuando el runner esté aprobado.")
        return 0

    (DIR_SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    ruta_resumen = DIR_SALIDAS / "tablas" / "resumen_benchmark.csv"

    # Filas de contexto desde la corrida oficial archivada (sin costo NEOS).
    filas_resumen: list[dict] = [
        fila_contexto_paper(
            "S1",
            metricas_paper,
            u_star,
            {"zonas_cambiadas_total": 0, "zonas_cambiadas_sede": 0, "puertos_movidos": 0.0},
            comun.comparar_asignaciones(asignacion_s1, asignacion_s2),
        ),
        fila_contexto_paper(
            "S2",
            metricas_paper,
            u_star,
            comun.comparar_asignaciones(asignacion_s2, asignacion_s1),
            {"zonas_cambiadas_total": 0, "zonas_cambiadas_sede": 0, "puertos_movidos": 0.0},
        ),
    ]
    pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

    pares_indices = [(i, j) for i in range(n_zonas) for j in range(i + 1, n_zonas)]
    for variante in variantes:
        etiqueta = f"benchmark_{variante}"
        pesos = pesos_de_pares(variante, pares_indices, ctx.vector_vulnerabilidad, config.s2_kappa)
        print(f"Lanzando {etiqueta} vía NEOS/CPLEX...", flush=True)
        modelo, _pares = construir_benchmark(ctx.coeffs, config, n_zonas, pesos, caps)
        solve_model(modelo, config=config, tee=False, time_limit=time_limit, mip_gap=mip_gap)
        terminacion = str(getattr(modelo, "_termination_condition", "desconocida"))
        if not solve_safe(modelo, accept_feasible=True):
            print(f"  ADVERTENCIA: {etiqueta} no entregó solución utilizable (termination={terminacion}). Se registra y se sigue.")
            filas_resumen.append({"fuente": "corrida_nueva", "variante": variante, "termination": terminacion})
            pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
            continue

        n_sol = np.zeros((n_zonas, len(config.venues)), dtype=int)
        for i in range(n_zonas):
            for v in range(len(config.venues)):
                n_sol[i, v] = int(round(value(modelo.n[i, v])))
        a_sol = np.array([float(value(modelo.A[i])) for i in range(n_zonas)], dtype=float)

        # Fidelidad ex post: |n − target| con el make_targets del paper (NO entra al
        # modelo). Se pasa como u_sol a summarize_scenario para que la métrica
        # objective_abs_dev sea exactamente esa fidelidad ex post — la cifra del
        # trade-off que pide R1.6.
        desviaciones = np.abs(n_sol - ctx.target)
        df_asig, metricas = summarize_scenario(
            etiqueta, ctx.df_inputs, n_sol, desviaciones, a_sol, ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config
        )
        ruta_asig = DIR_SALIDAS / "tablas" / f"asignacion_{etiqueta}.csv"
        df_asig.to_csv(ruta_asig, index=False)

        cambios_s1 = comun.comparar_asignaciones(df_asig, asignacion_s1)
        cambios_s2 = comun.comparar_asignaciones(df_asig, asignacion_s2)
        fila = {
            "fuente": "corrida_nueva",
            "variante": variante,
            "termination": terminacion,
            "objetivo_L1_pares": float(value(modelo.obj)),
            "objective_abs_dev": float(metricas["objective_abs_dev"]),
            "perdida_fidelidad_vs_S0_pct": 100.0 * (float(metricas["objective_abs_dev"]) - u_star) / u_star,
            "gini_no_ponderado_6d": comun.redondear6(metricas["accessibility_gini"]),
            "gini_ponderado_6d": comun.redondear6(metricas["accessibility_weighted_gini"]),
            "cobertura_15min_vulnerables": float(metricas["coverage_within_threshold_vulnerable"]),
            "rho_prioridad_puertos": float(metricas["rho_priority_ports"]),
            "zonas_cambiadas_total_vs_S1": cambios_s1["zonas_cambiadas_total"],
            "zonas_cambiadas_sede_vs_S1": cambios_s1["zonas_cambiadas_sede"],
            "puertos_movidos_vs_S1": cambios_s1["puertos_movidos"],
            "zonas_cambiadas_total_vs_S2": cambios_s2["zonas_cambiadas_total"],
            "puertos_movidos_vs_S2": cambios_s2["puertos_movidos"],
        }
        filas_resumen.append(fila)
        # Escritura incremental: si la corrida NEOS siguiente falla, lo anterior sobrevive.
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

        manifiesto = {
            "corrida": etiqueta,
            "variante": variante,
            "formulacion": "min sum_p w_p*|A_i - A_j| s.a. cuotas (sin lex, sin target, sin eps)",
            "pesos_pares": (
                "w_p = 1 (nu=1)"
                if variante == "no_ponderado"
                else f"w_p = 1 + {config.s2_kappa}*(nu_i+nu_j)/2 (nu = vulnerability_index, como S2)"
            ),
            "caps": "None (config.max_ports_matrix, como el pipeline publicado)" if caps is None else "matriz del config",
            "dominio_n": "enteros no negativos (mismo dominio que el modelo publicado)",
            "u_star": u_star,
            "fuente_u_star": str(comun.TABLAS_PAPER / "table_priority_summary.csv"),
            "time_limit_s": time_limit,
            "mip_gap": mip_gap,
            "termination": terminacion,
            "objetivo_L1_pares": float(value(modelo.obj)),
            "fidelidad_expost_abs_dev": float(metricas["objective_abs_dev"]),
            "metricas_precision_completa": {k: v for k, v in metricas.items() if k != "scenario"},
            "comparacion_vs_S1": cambios_s1,
            "comparacion_vs_S2": cambios_s2,
        }
        (DIR_SALIDAS / "tablas" / f"manifiesto_{etiqueta}.json").write_text(
            json.dumps(manifiesto, indent=2, default=float), encoding="utf-8"
        )
        print(f"  {etiqueta}: termination={terminacion}; tablas en {ruta_asig.name} y {ruta_resumen.name}")

    print(f"\nBenchmark terminado. Resumen consolidado: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
