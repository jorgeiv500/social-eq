#!/usr/bin/env python
"""C1 — Variantes de piso vinculante sobre S3 (revisión JTRG-D-26-00692).

Responde R1.5: el S3 del paper (γ=0.95 sobre el percentil 10 de la accesibilidad S2 del
conjunto protegido) no muerde — 95.45 % de las zonas ya superan ese piso y S3 coincide con
S2. Este experimento corre el MISMO mecanismo de piso blando (`solve_equity_refined` con
`floor_mask`/`floor_reference`/`shortfall_penalty`, idéntico al S3 oficial, líneas 1019–1034
de followup.py) con pisos derivados de la propia distribución archivada de accesibilidad S2
del conjunto protegido — sin umbrales inventados:

  - C1a: piso = mediana del conjunto protegido      (floor_quantile=0.50, gamma=1.0)
  - C1b: piso = percentil 75 del conjunto protegido (floor_quantile=0.75, gamma=1.0)

Por construcción, ~50 % y ~75 % de las zonas protegidas quedan por debajo del piso en el
punto de partida: el piso muerde y se observa qué paga el modelo (fidelidad, Gini, déficits
residuales d_k) por levantarlas.

Referencias fijas (corrida oficial del paper, no se recalculan):
- u* desde table_priority_summary.csv (fila S0), vía comun.leer_u_star().
- floor_reference = columna `accessibility` de scenario_S2.csv (la a2 archivada), igual que
  el S3 oficial usa la a2 de su propia corrida.
- Conjunto protegido: el mismo del S3 oficial (línea 961): vulnerabilidad >=
  cuantil(1 - s3_protected_share).

SEGURIDAD: por defecto corrida en seco. Solo --lanzar envía trabajos (2 corridas NEOS).

Uso:
    python run_c1.py            # corrida en seco: diagnóstico de pisos, sin NEOS
    python run_c1.py --lanzar   # 2 corridas NEOS (C1a, C1b)
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
    extract_solution_refined,
    solve_equity_refined,
    solve_safe,
    summarize_scenario,
)

AQUI = Path(__file__).resolve().parent
DIR_SALIDAS = AQUI / "salidas"
DIR_TABLAS = DIR_SALIDAS / "tablas"

VARIANTES = (
    {"etiqueta": "C1a_piso_mediana", "floor_quantile": 0.50, "floor_gamma": 1.0},
    {"etiqueta": "C1b_piso_p75", "floor_quantile": 0.75, "floor_gamma": 1.0},
)


def main() -> int:
    parser = argparse.ArgumentParser(description="C1: variantes de piso vinculante sobre el mecanismo S3.")
    parser.add_argument("--lanzar", action="store_true", help="Enviar realmente los trabajos a NEOS.")
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    args = parser.parse_args()

    print("Preparando insumos (idéntico camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config = ctx.config
    n_zonas = len(ctx.df_inputs)

    # Conjunto protegido: mismo criterio que el S3 oficial (followup.py línea 961).
    vuln = ctx.vector_vulnerabilidad
    mascara_protegida = vuln >= np.quantile(vuln, 1.0 - config.s3_protected_share)

    # floor_reference: accesibilidad S2 archivada, alineada al orden de df_inputs.
    s2_paper = comun.leer_asignacion_paper("S2")
    alineada = ctx.df_inputs[["UTAM"]].merge(s2_paper[["UTAM", "accessibility"]], on="UTAM", how="left")
    if alineada["accessibility"].isna().any():
        raise SystemExit("scenario_S2.csv no cubre todas las UTAM del contexto; no hay referencia de piso.")
    a2_archivada = alineada["accessibility"].to_numpy(dtype=float)

    u_star = comun.leer_u_star()
    asignacion_s1 = comun.leer_asignacion_paper("S1")
    asignacion_s2 = comun.leer_asignacion_paper("S2")
    eps = max(config.epsilon_relax, config.epsilon_s3)
    time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap

    protegidas = a2_archivada[mascara_protegida]
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)

    # Diagnóstico sin NEOS: dónde caería cada piso sobre la distribución archivada.
    filas_diag = []
    for esquema in ({"etiqueta": "S3_oficial", "floor_quantile": config.s3_floor_quantile,
                     "floor_gamma": config.s3_floor_gamma}, *VARIANTES):
        base = float(np.quantile(protegidas, esquema["floor_quantile"]))
        piso = esquema["floor_gamma"] * base
        filas_diag.append(
            {
                **{k: v for k, v in esquema.items()},
                "base_floor": base,
                "floor_value": piso,
                "protegidas_total": int(mascara_protegida.sum()),
                "protegidas_bajo_piso_en_S2": int((protegidas < piso).sum()),
                "pct_protegidas_bajo_piso_en_S2": 100.0 * float((protegidas < piso).mean()),
            }
        )
    diag = pd.DataFrame(filas_diag)
    diag.to_csv(DIR_TABLAS / "diagnostico_pisos.csv", index=False)

    print()
    print("=== Plan C1: pisos derivados de la distribución archivada (conjunto protegido, a2 de S2) ===")
    print(diag.to_string(index=False))
    print(f"\nEspecificación por corrida: mecanismo S3 oficial (piso blando, penalización "
          f"{config.s3_shortfall_penalty}), nu=vulnerabilidad, kappa={config.s2_kappa}, eps={eps}, "
          f"timelimit={time_limit}s, mipgap={mip_gap}")
    print(f"Corridas NEOS previstas: {len(VARIANTES)}")
    print(f"Correo NEOS que se usará: {comun.email_neos(config)}")
    print()

    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS. Ejecute con --lanzar cuando el runner esté aprobado.")
        return 0

    ruta_resumen = DIR_TABLAS / "resumen_piso_vinculante.csv"
    filas_resumen: list[dict] = []

    # Contexto: métricas archivadas de S2 y S3 (el S3 oficial no muerde).
    metricas_paper = comun.leer_metricas_paper()
    for esc in ("S2", "S3"):
        fila = metricas_paper.loc[metricas_paper["scenario"].astype(str).eq(esc)].iloc[0]
        filas_resumen.append(
            {
                "fuente": f"archivo_paper_{esc}",
                "variante": esc,
                "termination": "archivada",
                "gini_no_ponderado_6d": comun.redondear6(fila["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(fila["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(fila["coverage_within_threshold_vulnerable"]),
                "objective_abs_dev": float(fila["objective_abs_dev"]),
                "perdida_fidelidad_vs_S0_pct": 100.0 * (float(fila["objective_abs_dev"]) - u_star) / u_star,
            }
        )
    pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

    for esquema in VARIANTES:
        etiqueta = esquema["etiqueta"]
        print(f"Lanzando {etiqueta} vía NEOS/CPLEX...", flush=True)
        t0 = time.time()
        modelo, info_piso = solve_equity_refined(
            target=ctx.target,
            coeffs=ctx.coeffs,
            u_star=u_star,
            config=config,
            nu=vuln,
            epsilon=eps,
            kappa=config.s2_kappa,
            floor_mask=mascara_protegida,
            floor_reference=a2_archivada,
            floor_gamma=esquema["floor_gamma"],
            floor_quantile=esquema["floor_quantile"],
            shortfall_penalty=config.s3_shortfall_penalty,
            use_hard_floor=False,
            caps=config.max_ports_matrix,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        dur = time.time() - t0
        term = str(getattr(modelo, "_termination_condition", "desconocida"))
        if not solve_safe(modelo, accept_feasible=True):
            print(f"  ADVERTENCIA: {etiqueta} no entregó solución utilizable (termination={term}). Se registra y se sigue.")
            filas_resumen.append({"fuente": "corrida_nueva", "variante": etiqueta, "termination": term})
            pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)
            continue

        n_sol, u_sol, a_sol, d_sol = extract_solution_refined(modelo, n_zonas, len(config.venues))
        assert a_sol is not None
        df_asig, metricas = summarize_scenario(
            etiqueta, ctx.df_inputs, n_sol, u_sol, a_sol, ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config
        )
        df_asig.to_csv(DIR_TABLAS / f"asignacion_{etiqueta}.csv", index=False)

        piso = float(info_piso["floor_value"]) if info_piso else float("nan")
        protegidas_post = a_sol[mascara_protegida]
        cambios_s1 = comun.comparar_asignaciones(df_asig, asignacion_s1)
        cambios_s2 = comun.comparar_asignaciones(df_asig, asignacion_s2)
        fila = {
            "fuente": "corrida_nueva",
            "variante": etiqueta,
            "termination": term,
            "floor_quantile": esquema["floor_quantile"],
            "floor_gamma": esquema["floor_gamma"],
            "floor_value": piso,
            "protegidas_bajo_piso_pre_S2": int((protegidas < piso).sum()),
            "protegidas_bajo_piso_post": int((protegidas_post < piso - 1e-9).sum()),
            "deficit_total": float(np.sum(d_sol)) if d_sol is not None else 0.0,
            "deficit_max": float(np.max(d_sol)) if d_sol is not None and len(d_sol) else 0.0,
            "gini_no_ponderado_6d": comun.redondear6(metricas["accessibility_gini"]),
            "gini_ponderado_6d": comun.redondear6(metricas["accessibility_weighted_gini"]),
            "cobertura_15min_vulnerables": float(metricas["coverage_within_threshold_vulnerable"]),
            "objective_abs_dev": float(metricas["objective_abs_dev"]),
            "perdida_fidelidad_vs_S0_pct": 100.0 * (float(metricas["objective_abs_dev"]) - u_star) / u_star,
            "rho_prioridad_puertos": float(metricas["rho_priority_ports"]),
            "zonas_cambiadas_total_vs_S1": cambios_s1["zonas_cambiadas_total"],
            "puertos_movidos_vs_S1": cambios_s1["puertos_movidos"],
            "zonas_cambiadas_total_vs_S2": cambios_s2["zonas_cambiadas_total"],
            "puertos_movidos_vs_S2": cambios_s2["puertos_movidos"],
            "duracion_s": round(dur, 1),
        }
        filas_resumen.append(fila)
        # Escritura incremental: si la corrida siguiente falla, lo anterior sobrevive.
        pd.DataFrame(filas_resumen).to_csv(ruta_resumen, index=False)

        manifiesto = {
            **fila,
            "mecanismo": "piso blando S3 oficial (solve_equity_refined), floor_reference = accessibility de scenario_S2.csv archivado",
            "conjunto_protegido": "vulnerabilidad >= cuantil(1 - s3_protected_share), mismo criterio que followup.py línea 961",
            "info_piso": info_piso,
            "metricas_precision_completa": {k: v for k, v in metricas.items() if k != "scenario"},
        }
        (DIR_TABLAS / f"manifiesto_{etiqueta}.json").write_text(
            json.dumps(manifiesto, indent=2, default=float), encoding="utf-8"
        )
        print(f"  {etiqueta}: termination={term} ({dur:.0f}s); piso={piso:.2f}; "
              f"bajo piso pre/post = {fila['protegidas_bajo_piso_pre_S2']}/{fila['protegidas_bajo_piso_post']}; "
              f"déficit total = {fila['deficit_total']:.3f}")

    print(f"\nC1 terminado. Resumen consolidado: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
