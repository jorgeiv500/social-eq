#!/usr/bin/env python
"""A3 — Escenarios de crecimiento 2028/2030 con cuotas escaladas (JTRG-D-26-00692).

Responde R2.2 y refuerza R2.1: escala las cuotas Q_v del paper con la
trayectoria oficial de flota eléctrica de Bogotá (media +1,715 VE/año,
derivada en tiempo de ejecución de data/inputs/bogota_official_ev_growth.csv)
y re-corre S0 -> S1 -> S2 por horizonte con el modelo EXISTENTE
(`solve_s0`, `solve_equity`, `solve_equity_refined` de src/social_eq/followup.py).

Los supuestos de escenario (extrapolación lineal, proporcionalidad
puertos<->flota, mezcla por sede constante, redondeo al entero más cercano)
están declarados en el README.md de esta carpeta; el runner los implementa
tal cual y NO introduce cifras propias.

SEGURIDAD: sin `--lanzar` solo se imprime la derivación de cuotas y el plan.

Uso:
    python run_a3.py                       # corrida en seco (sin NEOS)
    python run_a3.py --lanzar              # 6 corridas NEOS (S0/S1/S2 x 2028/2030)
    python run_a3.py --lanzar --horizontes 2028   # 3 corridas
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comun  # noqa: E402
from social_eq.followup import (  # noqa: E402
    ModelConfig,
    extract_solution,
    extract_solution_refined,
    make_targets,
    solve_equity,
    solve_equity_refined,
    solve_s0,
    solve_safe,
    summarize_scenario,
)

AQUI = Path(__file__).resolve().parent
DIR_SALIDAS = AQUI / "salidas"
CSV_FLOTA = comun.RAIZ / "data" / "inputs" / "bogota_official_ev_growth.csv"
ANIO_BASE = 2024  # último corte oficial del CSV; ancla del supuesto de proporcionalidad (S2 del README)
ANIO_INICIO = 2020


def derivar_cuotas(config: ModelConfig, horizontes: list[int]) -> pd.DataFrame:
    """Deriva multiplicadores y cuotas escaladas desde el CSV oficial (sin cifras fijas)."""
    flota = pd.read_csv(CSV_FLOTA)
    try:
        ve_inicio = int(flota.loc[flota["year"] == ANIO_INICIO, "electric_vehicles"].iloc[0])
        ve_base = int(flota.loc[flota["year"] == ANIO_BASE, "electric_vehicles"].iloc[0])
    except IndexError as exc:
        raise SystemExit(f"{CSV_FLOTA} no contiene los años {ANIO_INICIO} y {ANIO_BASE}: {exc}")
    media_anual = (ve_base - ve_inicio) / (ANIO_BASE - ANIO_INICIO)

    filas = []
    for anio in horizontes:
        flota_proyectada = ve_base + (anio - ANIO_BASE) * media_anual
        multiplicador = flota_proyectada / ve_base
        cuotas = {sede: int(round(valor * multiplicador)) for sede, valor in config.quotas.items()}
        filas.append(
            {
                "horizonte": anio,
                "ve_2020": ve_inicio,
                "ve_2024": ve_base,
                "media_anual_ve": media_anual,
                "flota_proyectada": flota_proyectada,
                "multiplicador": multiplicador,
                **{f"cuota_{sede}": cuota for sede, cuota in cuotas.items()},
                "cuota_total": sum(cuotas.values()),
            }
        )
    return pd.DataFrame(filas)


def spearman_vs_paper(df_asig: pd.DataFrame, asignacion_paper: pd.DataFrame) -> float:
    """Estabilidad de rangos frente al S2 del paper (cuotas distintas: no se cuentan puertos)."""
    junto = df_asig[["UTAM", "ports_total"]].merge(
        asignacion_paper[["UTAM", "ports_total"]], on="UTAM", suffixes=("_h", "_paper")
    )
    return float(junto["ports_total_h"].corr(junto["ports_total_paper"], method="spearman"))


def main() -> int:
    parser = argparse.ArgumentParser(description="A3: escenarios de crecimiento con cuotas escaladas.")
    parser.add_argument("--lanzar", action="store_true", help="Enviar realmente los trabajos a NEOS.")
    parser.add_argument("--horizontes", default="2028,2030", help="Años horizonte separados por comas.")
    parser.add_argument("--time-limit", type=int, default=None, help="Segundos de solver (defecto 3600 del config).")
    parser.add_argument("--mip-gap", type=float, default=None, help="Gap MIP (defecto 0.01 del config).")
    args = parser.parse_args()
    horizontes = [int(x) for x in args.horizontes.split(",") if x.strip()]

    print("Preparando insumos (mismo camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config_base = ctx.config
    n_zonas = len(ctx.df_inputs)
    asignacion_s2_paper = comun.leer_asignacion_paper("S2")

    derivacion = derivar_cuotas(config_base, horizontes)
    (DIR_SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    derivacion.to_csv(DIR_SALIDAS / "tablas" / "derivacion_cuotas.csv", index=False)

    time_limit = args.time_limit if args.time_limit is not None else config_base.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config_base.solver_mip_gap

    print()
    print("=== Plan de escenarios de crecimiento A3 ===")
    print(f"Fuente de flota: {CSV_FLOTA}")
    print(derivacion.to_string(index=False))
    print(f"\nPor horizonte: S0 (deriva u* nuevo, los targets escalan con las cuotas), "
          f"S1 (eps={max(config_base.epsilon_relax, config_base.epsilon_s1)}), "
          f"S2 (eps={max(config_base.epsilon_relax, config_base.epsilon_s2)}, kappa={config_base.s2_kappa}).")
    print(f"Corridas NEOS previstas: {3 * len(horizontes)} | timelimit={time_limit}s, mipgap={mip_gap}")
    print(f"Correo NEOS que se usará: {comun.email_neos(config_base)}")
    print()

    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS. Confirmar los supuestos S1-S4 del README antes de --lanzar.")
        return 0

    filas_resumen: list[dict] = []
    ruta_resumen = DIR_SALIDAS / "tablas" / "resumen_crecimiento.csv"

    for _, fila_cuotas in derivacion.iterrows():
        anio = int(fila_cuotas["horizonte"])
        cuotas = {sede: int(fila_cuotas[f"cuota_{sede}"]) for sede in config_base.venues}
        config_h = replace(config_base, quotas=cuotas)
        _pi, target_h = make_targets(ctx.df_inputs, config_h)
        asignaciones_h: dict[str, pd.DataFrame] = {}

        def registrar(nombre: str, n_sol, u_sol, a_sol, metricas_extra: dict, terminacion: str) -> pd.DataFrame:
            df_asig, metricas = summarize_scenario(
                f"{nombre}_{anio}", ctx.df_inputs, n_sol, u_sol, a_sol, ctx.tmat,
                ctx.mascara_alta_vulnerabilidad, config=config_h,
            )
            df_asig.to_csv(DIR_SALIDAS / "tablas" / f"asignacion_{nombre}_{anio}.csv", index=False)
            fila = {
                "horizonte": anio,
                "escenario": nombre,
                **{f"cuota_{sede}": cuotas[sede] for sede in config_h.venues},
                "termination": terminacion,
                "objective_abs_dev": float(metricas["objective_abs_dev"]),
                "gini_no_ponderado_6d": comun.redondear6(metricas["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(metricas["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(metricas["coverage_within_threshold_vulnerable"]),
                "rho_prioridad_puertos": float(metricas["rho_priority_ports"]),
                "spearman_vs_S2_paper": spearman_vs_paper(df_asig, asignacion_s2_paper),
                **metricas_extra,
            }
            filas_resumen.append(fila)
            pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)  # escritura incremental
            manifiesto = {
                **fila,
                "time_limit_s": time_limit,
                "mip_gap": mip_gap,
                "metricas_precision_completa": {k: v for k, v in metricas.items() if k != "scenario"},
            }
            (DIR_SALIDAS / "tablas" / f"manifiesto_{nombre}_{anio}.json").write_text(
                json.dumps(manifiesto, indent=2, default=float), encoding="utf-8"
            )
            return df_asig

        print(f"[{anio}] Lanzando S0 (cuotas {cuotas}) vía NEOS/CPLEX...", flush=True)
        modelo_s0, u_star_h = solve_s0(target_h, config=config_h)
        if not solve_safe(modelo_s0):
            raise SystemExit(f"S0_{anio} no resolvió: {getattr(modelo_s0, '_termination_condition', None)}")
        n0, u0, _ = extract_solution(modelo_s0, n_zonas, len(config_h.venues))
        a0 = comun.calcular_accesibilidad(n0, ctx.coeffs, config_h)
        registrar("S0", n0, u0, a0, {"u_star_horizonte": u_star_h, "perdida_fidelidad_pct": 0.0},
                  str(getattr(modelo_s0, "_termination_condition", "desconocida")))

        print(f"[{anio}] Lanzando S1 vía NEOS/CPLEX...", flush=True)
        eps_s1 = max(config_h.epsilon_relax, config_h.epsilon_s1)
        modelo_s1 = solve_equity(target_h, ctx.coeffs, u_star_h, config=config_h, nu=None, epsilon=eps_s1)
        if not solve_safe(modelo_s1, accept_feasible=True):
            raise SystemExit(f"S1_{anio} no resolvió: {getattr(modelo_s1, '_termination_condition', None)}")
        n1, u1, a1 = extract_solution(modelo_s1, n_zonas, len(config_h.venues))
        assert a1 is not None
        asignaciones_h["S1"] = registrar(
            "S1", n1, u1, a1,
            {"u_star_horizonte": u_star_h,
             "perdida_fidelidad_pct": 100.0 * (float(u1.sum()) - u_star_h) / u_star_h,
             "epsilon": eps_s1},
            str(getattr(modelo_s1, "_termination_condition", "desconocida")),
        )

        print(f"[{anio}] Lanzando S2 vía NEOS/CPLEX...", flush=True)
        eps_s2 = max(config_h.epsilon_relax, config_h.epsilon_s2)
        modelo_s2, _ = solve_equity_refined(
            target=target_h, coeffs=ctx.coeffs, u_star=u_star_h, config=config_h,
            nu=ctx.vector_vulnerabilidad, epsilon=eps_s2, kappa=config_h.s2_kappa,
            floor_mask=None, floor_reference=None, use_hard_floor=False,
            caps=config_h.max_ports_matrix, time_limit=time_limit, mip_gap=mip_gap,
        )
        if not solve_safe(modelo_s2, accept_feasible=True):
            raise SystemExit(f"S2_{anio} no resolvió: {getattr(modelo_s2, '_termination_condition', None)}")
        n2, u2, a2, _ = extract_solution_refined(modelo_s2, n_zonas, len(config_h.venues))
        assert a2 is not None
        df_s2 = registrar(
            "S2", n2, u2, a2,
            {"u_star_horizonte": u_star_h,
             "perdida_fidelidad_pct": 100.0 * (float(u2.sum()) - u_star_h) / u_star_h,
             "epsilon": eps_s2},
            str(getattr(modelo_s2, "_termination_condition", "desconocida")),
        )

        # La cifra que responde a R2.1: reasignación S2 vs S1 dentro del mismo horizonte
        # (mismas cuotas totales, la comparación de puertos movidos es válida).
        cambios = comun.comparar_asignaciones(df_s2, asignaciones_h["S1"])
        for fila in filas_resumen:
            if fila["horizonte"] == anio and fila["escenario"] == "S2":
                fila.update({f"{k}_S2_vs_S1": v for k, v in cambios.items()})
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
        print(f"[{anio}] S2 vs S1: {cambios}")

    print(f"\nEscenarios terminados. Resumen consolidado: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
