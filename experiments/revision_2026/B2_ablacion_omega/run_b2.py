#!/usr/bin/env python
"""B2 — Ablación de ω_iv: cadena S0→S1→S2 con pesos de sede uniformes (revisión JTRG-D-26-00692).

Responde R1.3. ω_iv (Eq. 10) entra en la optimización solo por dos puntos, verificados en
`src/social_eq/followup.py`: los targets (`make_targets`, vía π_iv ∝ s_i·ω_iv) y los
coeficientes de accesibilidad (`build_access_coeffs`, ω_iv·exp(−t_ij/λ_v)). La variante
"uniforme" fija ω_iv = 1/3 en las tres sedes (la salvaguarda que el propio `build_inputs`
aplica cuando la atención es nula) y recorre S0u→S1u→S2u con TODO lo demás idéntico al
pipeline oficial (`run_refined_scenarios`). u* se re-deriva con S0u porque los targets
uniformes cambian el óptimo de fidelidad.

Cada asignación uniforme se evalúa con dos varas (post-proceso, sin costo NEOS):
- métrica propia: A_i con coeficientes uniformes (lo que reportaría el pipeline ablacionado);
- métrica aprendida: A_i con los coeficientes ω aprendidos del paper (comparable con las
  tablas archivadas de `outputs/.../paper_v6_experiments/tables/`).

SEGURIDAD: por defecto corrida en seco (deriva el diagnóstico de ω e imprime el plan SIN
enviar nada a NEOS). Solo `--lanzar` envía trabajos (3 corridas: S0u, S1u, S2u).

Uso:
    python run_b2.py                 # corrida en seco (sin NEOS)
    python run_b2.py --lanzar        # 3 corridas NEOS
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comun  # noqa: E402
from social_eq.followup import (  # noqa: E402
    build_access_coeffs,
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
DIR_TABLAS = DIR_SALIDAS / "tablas"


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2: ablación de omega (uniforme vs aprendida) sobre S0-S2.")
    parser.add_argument(
        "--lanzar",
        action="store_true",
        help="Enviar realmente los trabajos a NEOS. Sin esta bandera solo se imprime el plan.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Límite de tiempo del solver en segundos, solo para S2u (única etapa cuya firma lo acepta).",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Gap MIP relativo, solo para S2u.",
    )
    return parser.parse_args()


def spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    valor = pd.Series(np.asarray(a, dtype=float)).corr(pd.Series(np.asarray(b, dtype=float)), method="spearman")
    return float(valor) if pd.notna(valor) else float("nan")


def spearman_asignaciones(df_nueva: pd.DataFrame, df_referencia: pd.DataFrame) -> dict[str, float]:
    """rho de Spearman de la asignación entre variantes, alineada por UTAM."""
    junto = df_nueva[["UTAM", "ports_total", *comun.COLUMNAS_PUERTOS]].merge(
        df_referencia[["UTAM", "ports_total", *comun.COLUMNAS_PUERTOS]],
        on="UTAM",
        how="inner",
        suffixes=("_nueva", "_ref"),
    )
    if len(junto) != len(df_nueva):
        raise RuntimeError("Las asignaciones no cubren el mismo conjunto de UTAM.")
    salida = {"rho_ports_total": spearman(junto["ports_total_nueva"], junto["ports_total_ref"])}
    for col in comun.COLUMNAS_PUERTOS:
        salida[f"rho_{col}"] = spearman(junto[f"{col}_nueva"], junto[f"{col}_ref"])
    return salida


def main() -> int:
    args = parsear_argumentos()

    print("Preparando insumos (idéntico camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config = ctx.config
    n_zonas = len(ctx.df_inputs)
    n_sedes = len(config.venues)

    # --- Variante uniforme: omega_iv = 1/3, mismas funciones públicas del pipeline ---
    df_unif = ctx.df_inputs.copy()
    for sede in config.venues:
        df_unif[sede] = 1.0 / n_sedes
    _, target_unif = make_targets(df_unif, config)
    coeffs_unif = build_access_coeffs(df_unif, ctx.tmat, config.lambda_v, config)

    # --- Diagnóstico sin NEOS: cuán lejos de uniforme está la omega aprendida ---
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    omega_aprendida = ctx.df_inputs[list(config.venues)].to_numpy(dtype=float)
    target_aprendido = ctx.target
    dif_target_por_sede = {
        sede: float(np.abs(target_unif[:, k] - target_aprendido[:, k]).sum() / 2.0)
        for k, sede in enumerate(config.venues)
    }
    diagnostico = {
        "desviacion_media_abs_omega_vs_uniforme": float(np.abs(omega_aprendida - 1.0 / n_sedes).mean()),
        **{f"omega_media_{sede}": float(omega_aprendida[:, k].mean()) for k, sede in enumerate(config.venues)},
        "rho_target_total_unif_vs_aprendido": spearman(target_unif.sum(axis=1), target_aprendido.sum(axis=1)),
        **{
            f"rho_target_{sede}_unif_vs_aprendido": spearman(target_unif[:, k], target_aprendido[:, k])
            for k, sede in enumerate(config.venues)
        },
        **{f"desplazamiento_target_puertos_{sede}": v for sede, v in dif_target_por_sede.items()},
        "desplazamiento_target_puertos_total": float(sum(dif_target_por_sede.values())),
    }
    pd.DataFrame([diagnostico]).to_csv(DIR_TABLAS / "diagnostico_omega.csv", index=False)

    asignaciones_paper = {esc: comun.leer_asignacion_paper(esc) for esc in ("S0", "S1", "S2")}
    metricas_paper = comun.leer_metricas_paper()
    u_star_paper = comun.leer_u_star()

    eps_s1 = max(config.epsilon_relax, config.epsilon_s1)
    eps_s2 = max(config.epsilon_relax, config.epsilon_s2)
    time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap

    print()
    print("=== Plan de la ablación B2 ===")
    print(f"Zonas (UTAM): {n_zonas} | pares de equidad: {n_zonas * (n_zonas - 1) // 2} | variables enteras: {n_zonas * n_sedes}")
    print(f"omega aprendida: desviación media |omega - 1/3| = {diagnostico['desviacion_media_abs_omega_vs_uniforme']:.4f}; "
          f"medias por sede = " + ", ".join(f"{s}={diagnostico[f'omega_media_{s}']:.3f}" for s in config.venues))
    print(f"Targets: rho(total unif vs aprendido) = {diagnostico['rho_target_total_unif_vs_aprendido']:.4f}; "
          f"desplazamiento de target = {diagnostico['desplazamiento_target_puertos_total']:.1f} puertos-equivalentes")
    print("Corridas NEOS previstas: 3")
    print(f"  1. S0u  (solve_s0, targets uniformes) -> deriva u*_unif (u* aprendido, solo contexto: {u_star_paper:.4f})")
    print(f"  2. S1u  (solve_equity, nu=None, eps={eps_s1}, solver del config: {config.solver_time_limit}s/{config.solver_mip_gap})")
    print(f"  3. S2u  (solve_equity_refined, nu=vulnerabilidad, kappa={config.s2_kappa}, eps={eps_s2}, sin piso, "
          f"timelimit={time_limit}s, mipgap={mip_gap})")
    print(f"Diagnóstico sin NEOS escrito en {DIR_TABLAS / 'diagnostico_omega.csv'}")
    print(f"Correo NEOS que se usará: {comun.email_neos(config)}")
    print()

    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS. Ejecute con --lanzar cuando el runner esté aprobado.")
        return 0

    ruta_resumen = DIR_TABLAS / "resumen_ablacion_omega.csv"
    filas_resumen: list[dict] = []

    # Filas de contexto: métricas archivadas del paper (vara aprendida por construcción).
    for esc in ("S0", "S1", "S2"):
        fila_paper = metricas_paper.loc[metricas_paper["scenario"].astype(str).eq(esc)].iloc[0]
        filas_resumen.append(
            {
                "fuente": f"archivo_paper_{esc}",
                "escenario": esc,
                "omega": "aprendida",
                "termination": "archivada",
                "u_star_variante": u_star_paper,
                "objective_abs_dev": float(fila_paper["objective_abs_dev"]),
                "perdida_fidelidad_vs_S0_pct": 100.0 * (float(fila_paper["objective_abs_dev"]) - u_star_paper) / u_star_paper,
                "gini_no_ponderado_6d_metrica_aprendida": comun.redondear6(fila_paper["accessibility_gini"]),
                "gini_ponderado_6d_metrica_aprendida": comun.redondear6(fila_paper["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(fila_paper["coverage_within_threshold_vulnerable"]),
                "rho_prioridad_puertos": float(fila_paper["rho_priority_ports"]),
            }
        )
    pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

    u_star_unif: float | None = None

    def registrar(etiqueta: str, escenario_ref: str, n_sol, u_sol, terminacion: str,
                  duracion_s: float, parametros: dict) -> None:
        nonlocal filas_resumen
        # Vara propia (coeficientes uniformes) y vara aprendida (coeficientes del paper).
        a_propia = comun.calcular_accesibilidad(n_sol, coeffs_unif, config)
        a_aprendida = comun.calcular_accesibilidad(n_sol, ctx.coeffs, config)
        df_asig, met_propia = summarize_scenario(
            etiqueta, ctx.df_inputs, n_sol, u_sol, a_propia, ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config
        )
        _, met_aprendida = summarize_scenario(
            etiqueta + "_metrica_aprendida", ctx.df_inputs, n_sol, u_sol, a_aprendida, ctx.tmat,
            ctx.mascara_alta_vulnerabilidad, config=config
        )
        ruta_asig = DIR_TABLAS / f"asignacion_B2_{etiqueta}.csv"
        df_asig.to_csv(ruta_asig, index=False)

        referencia = asignaciones_paper[escenario_ref]
        cambios = comun.comparar_asignaciones(df_asig, referencia)
        rhos = spearman_asignaciones(df_asig, referencia)

        assert u_star_unif is not None
        fila = {
            "fuente": "corrida_nueva",
            "escenario": escenario_ref,
            "omega": "uniforme",
            "termination": terminacion,
            "u_star_variante": u_star_unif,
            "objective_abs_dev": float(met_propia["objective_abs_dev"]),
            "perdida_fidelidad_vs_S0_pct": 100.0 * (float(met_propia["objective_abs_dev"]) - u_star_unif) / u_star_unif,
            "gini_no_ponderado_6d_metrica_aprendida": comun.redondear6(met_aprendida["accessibility_gini"]),
            "gini_ponderado_6d_metrica_aprendida": comun.redondear6(met_aprendida["accessibility_weighted_gini"]),
            "gini_no_ponderado_6d_metrica_propia": comun.redondear6(met_propia["accessibility_gini"]),
            "gini_ponderado_6d_metrica_propia": comun.redondear6(met_propia["accessibility_weighted_gini"]),
            "cobertura_15min_vulnerables": float(met_propia["coverage_within_threshold_vulnerable"]),
            "rho_prioridad_puertos": float(met_propia["rho_priority_ports"]),
            "rho_asignacion_total_vs_paper": rhos["rho_ports_total"],
            "zonas_cambiadas_total_vs_paper": cambios["zonas_cambiadas_total"],
            "zonas_cambiadas_sede_vs_paper": cambios["zonas_cambiadas_sede"],
            "puertos_movidos_vs_paper": cambios["puertos_movidos"],
            "duracion_s": round(duracion_s, 1),
        }
        filas_resumen.append(fila)
        # Escritura incremental: si una corrida NEOS posterior falla, lo anterior sobrevive.
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

        manifiesto = {
            "corrida": etiqueta,
            "escenario_referencia": escenario_ref,
            "construccion_omega_uniforme": "df_inputs oficial con columnas de sede fijadas a 1/3; targets y coeficientes recomputados con make_targets/build_access_coeffs",
            "u_star_unif": u_star_unif,
            "u_star_paper_solo_contexto": u_star_paper,
            "parametros": parametros,
            "termination": terminacion,
            "duracion_s": duracion_s,
            "metricas_precision_completa_metrica_propia": {k: v for k, v in met_propia.items() if k != "scenario"},
            "metricas_precision_completa_metrica_aprendida": {k: v for k, v in met_aprendida.items() if k != "scenario"},
            "comparacion_vs_paper": cambios,
            "rho_spearman_vs_paper": rhos,
        }
        (DIR_TABLAS / f"manifiesto_B2_{etiqueta}.json").write_text(
            json.dumps(manifiesto, indent=2, default=float), encoding="utf-8"
        )
        print(f"  {etiqueta}: termination={terminacion} ({duracion_s:.0f}s); tablas en {ruta_asig.name} y {ruta_resumen.name}")

    # --- Corrida 1: S0u (deriva u*_unif) ---
    print("Lanzando S0u (targets uniformes) vía NEOS/CPLEX...", flush=True)
    t0 = time.time()
    modelo_s0, u_star_unif = solve_s0(target_unif, config=config)
    dur = time.time() - t0
    term = str(getattr(modelo_s0, "_termination_condition", "desconocida"))
    if not solve_safe(modelo_s0):
        raise SystemExit(f"S0u no resolvió (termination={term}); se aborta: sin u*_unif no hay cadena.")
    n0, u0, _ = extract_solution(modelo_s0, n_zonas, n_sedes)
    print(f"  u*_unif = {u_star_unif:.6f} (u* aprendido, contexto: {u_star_paper:.6f})")
    registrar("S0u", "S0", n0, u0, term, dur, {"etapa": "solve_s0", "targets": "uniformes"})

    # --- Corrida 2: S1u ---
    print("Lanzando S1u vía NEOS/CPLEX...", flush=True)
    t0 = time.time()
    modelo_s1 = solve_equity(target_unif, coeffs_unif, u_star_unif, config=config, nu=None, epsilon=eps_s1)
    dur = time.time() - t0
    term = str(getattr(modelo_s1, "_termination_condition", "desconocida"))
    if not solve_safe(modelo_s1):
        print(f"  ADVERTENCIA: S1u no entregó solución utilizable (termination={term}). Se registra y se sigue con S2u.")
        filas_resumen.append({"fuente": "corrida_nueva", "escenario": "S1", "omega": "uniforme", "termination": term})
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
    else:
        n1, u1, _ = extract_solution(modelo_s1, n_zonas, n_sedes)
        registrar("S1u", "S1", n1, u1, term, dur, {"etapa": "solve_equity", "epsilon": eps_s1, "nu": None})

    # --- Corrida 3: S2u ---
    print("Lanzando S2u vía NEOS/CPLEX...", flush=True)
    t0 = time.time()
    modelo_s2, _ = solve_equity_refined(
        target=target_unif,
        coeffs=coeffs_unif,
        u_star=u_star_unif,
        config=config,
        nu=ctx.vector_vulnerabilidad,
        epsilon=eps_s2,
        kappa=config.s2_kappa,
        floor_mask=None,
        floor_reference=None,
        use_hard_floor=False,
        caps=config.max_ports_matrix,
        time_limit=time_limit,
        mip_gap=mip_gap,
    )
    dur = time.time() - t0
    term = str(getattr(modelo_s2, "_termination_condition", "desconocida"))
    if not solve_safe(modelo_s2, accept_feasible=True):
        print(f"  ADVERTENCIA: S2u no entregó solución utilizable (termination={term}). Se registra.")
        filas_resumen.append({"fuente": "corrida_nueva", "escenario": "S2", "omega": "uniforme", "termination": term})
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
    else:
        n2, u2, _, _ = extract_solution_refined(modelo_s2, n_zonas, n_sedes)
        registrar(
            "S2u", "S2", n2, u2, term, dur,
            {"etapa": "solve_equity_refined", "epsilon": eps_s2, "kappa": config.s2_kappa,
             "time_limit_s": time_limit, "mip_gap": mip_gap, "piso": None},
        )

    print(f"\nAblación terminada. Resumen consolidado: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
