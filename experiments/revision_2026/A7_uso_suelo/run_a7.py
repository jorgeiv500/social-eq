#!/usr/bin/env python
"""A7 — S2 con elegibilidad por uso del suelo (POT, Decreto 555 de 2021) por (zona, sede).

Regla (declarada en el README): la sede v es elegible en la UTAM i solo si la fracción del
suelo zonificado de i cuyo área de actividad es compatible con v alcanza el umbral tau;
si no, n[i, v] <= 0. Las UTAM que la capa no zonifica (cobertura < 5 %: suelo rural y
borde) quedan sin tope, porque no hay dato con qué acotarlas.

Pasos:
1. Diagnóstico determinista (siempre, sin NEOS): por umbral, pares excluidos, masa de target
   expulsada D_v, cota de conservación de masa 2·D frente al presupuesto u*·(1+0.08) del S2
   archivado, y u*_cap = óptimo EXACTO de la etapa S0 con los topes. S0 es separable por sede
   (min sum|n - t| con sum n = cuota, 0 <= n <= tope, n entero) y su función es convexa por
   zona, así que el voraz por costo marginal es óptimo; se valida reproduciendo el u*
   archivado sin topes antes de usarlo.
2. Con --lanzar: S2 (nu = vulnerabilidad, kappa = 4, epsilon = 0.08 sobre u*_cap, sin piso)
   con caps por (zona, sede) vía `solve_equity_refined(..., caps=...)` en NEOS/CPLEX
   (3600 s, gap 0.01). No modifica src/.

Uso:
    python run_a7.py                                   # diagnóstico en seco (sin NEOS)
    python run_a7.py --lanzar --umbrales 0.05,0.10     # corridas NEOS (una por umbral)
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

import comun  # noqa: E402
from social_eq.followup import (  # noqa: E402
    extract_solution_refined,
    ports_per_capita_by_quartile,
    solve_equity_refined,
    solve_safe,
    summarize_scenario,
)

DIR_SALIDAS = AQUI / "salidas"
TABLAS = DIR_SALIDAS / "tablas"
USO_SUELO = TABLAS / "uso_suelo_utam.csv"
VULN_CSV = comun.RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv"
COBERTURA_MIN = 0.05  # por debajo, la UTAM no está zonificada por la capa: sin tope
EPSILON_S2 = 0.08
UMBRALES_DIAGNOSTICO = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
UMBRAL_PRINCIPAL = 0.10


def s0_exacto(target: np.ndarray, cuotas: list[int], caps: np.ndarray | None) -> tuple[float, np.ndarray]:
    """Óptimo exacto de S0 (con topes opcionales) por sede, voraz sobre costos marginales convexos."""
    n_z, n_v = target.shape
    sol = np.zeros((n_z, n_v), dtype=int)
    for v in range(n_v):
        t = target[:, v]
        tope = np.full(n_z, np.iinfo(np.int64).max) if caps is None else np.floor(caps[:, v] + 1e-9).astype(np.int64)
        n = np.minimum(np.floor(t).astype(np.int64), tope)
        falta = int(cuotas[v] - n.sum())
        if falta > 0:
            heap = [(abs(n[i] + 1 - t[i]) - abs(n[i] - t[i]), i) for i in range(n_z) if n[i] < tope[i]]
            heapq.heapify(heap)
            for _ in range(falta):
                if not heap:
                    raise RuntimeError(f"Sede {v}: los topes no alcanzan la cuota.")
                _c, i = heapq.heappop(heap)
                n[i] += 1
                if n[i] < tope[i]:
                    heapq.heappush(heap, (abs(n[i] + 1 - t[i]) - abs(n[i] - t[i]), i))
        elif falta < 0:
            heap = [(abs(n[i] - 1 - t[i]) - abs(n[i] - t[i]), i) for i in range(n_z) if n[i] > 0]
            heapq.heapify(heap)
            for _ in range(-falta):
                _c, i = heapq.heappop(heap)
                n[i] -= 1
                if n[i] > 0:
                    heapq.heappush(heap, (abs(n[i] - 1 - t[i]) - abs(n[i] - t[i]), i))
        sol[:, v] = n
    return float(np.abs(sol - target).sum()), sol


def construir_caps(uso: pd.DataFrame, tau: float, cuotas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """caps[i, v] = 0 si la sede no es elegible; cuota_v (no restrictivo) en otro caso."""
    zonificada = uso["cobertura_frac"].to_numpy() >= COBERTURA_MIN
    excluida = np.zeros((len(uso), len(comun.SEDES)), dtype=bool)
    for k, sede in enumerate(comun.SEDES):
        frac = uso[f"frac_compatible_{sede}"].fillna(1.0).to_numpy()
        excluida[:, k] = zonificada & (frac < tau)
    caps = np.where(excluida, 0.0, np.array(cuotas, dtype=float)[None, :])
    return caps, excluida


def parsear() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A7: S2 con elegibilidad por uso del suelo (POT 555).")
    p.add_argument("--lanzar", action="store_true", help="Enviar las corridas a NEOS (sin esto, solo diagnóstico).")
    p.add_argument("--umbrales", default=f"{UMBRAL_PRINCIPAL:.2f}", help="Umbrales tau a resolver en NEOS, separados por comas.")
    p.add_argument("--holgura-absoluta", action="store_true",
                   help="Control: presupuesto u*_cap + 0.08·u* (misma holgura absoluta que el S2 archivado) en vez de (1+0.08)·u*_cap.")
    p.add_argument("--time-limit", type=int, default=None)
    p.add_argument("--mip-gap", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parsear()
    TABLAS.mkdir(parents=True, exist_ok=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS)
    config = ctx.config
    cuotas = [int(config.quotas[s]) for s in config.venues]
    u_star = comun.leer_u_star()
    s2 = comun.leer_asignacion_paper("S2")
    uso = ctx.df_inputs[["UTAM"]].merge(pd.read_csv(USO_SUELO), on="UTAM", how="left")
    if uso["area_utam_m2"].isna().any():
        raise SystemExit(f"UTAM sin cruce de uso del suelo: {uso.loc[uso['area_utam_m2'].isna(), 'UTAM'].tolist()}")
    s2_al = ctx.df_inputs[["UTAM"]].merge(s2, on="UTAM", how="left")
    n_s2 = s2_al[list(comun.COLUMNAS_PUERTOS)].to_numpy()

    # Validación del S0 exacto: sin topes debe reproducir el u* archivado.
    u_libre, _ = s0_exacto(ctx.target, cuotas, None)
    if abs(u_libre - u_star) > 1e-6:
        raise SystemExit(f"El S0 exacto sin topes da {u_libre!r} y el archivado es {u_star!r}; no se sigue.")
    print(f"S0 exacto sin topes = {u_libre:.10f} (archivado {u_star:.10f}): validado")

    filas = []
    for tau in UMBRALES_DIAGNOSTICO:
        caps, excl = construir_caps(uso, tau, cuotas)
        d_v = [float(ctx.target[excl[:, k], k].sum()) for k in range(len(cuotas))]
        u_cap, n0 = s0_exacto(ctx.target, cuotas, caps)
        fila = {
            "tau": tau,
            "utam_zonificadas": int((uso["cobertura_frac"] >= COBERTURA_MIN).sum()),
            "pares_excluidos": int(excl.sum()),
            "utam_con_alguna_sede_excluida": int(excl.any(axis=1).sum()),
            "puertos_S2_en_pares_excluidos": int(n_s2[excl].sum()),
            "puertos_S2_vulnerables_en_pares_excluidos": int(n_s2[excl & ctx.mascara_alta_vulnerabilidad[:, None]].sum()),
            "cota_2D": 2.0 * sum(d_v),
            "presupuesto_S2_archivado": u_star * (1 + EPSILON_S2),
            "infactible_vs_u_star_archivado": bool(2.0 * sum(d_v) > u_star * (1 + EPSILON_S2)),
            "u_star_cap": u_cap,
            "presupuesto_S2_cap": u_cap * (1 + EPSILON_S2),
        }
        for k, sede in enumerate(comun.SEDES):
            fila[f"pares_excluidos_{sede}"] = int(excl[:, k].sum())
            fila[f"D_{sede}"] = d_v[k]
        filas.append(fila)
        pd.DataFrame(
            {"UTAM": uso["UTAM"], **{f"excluida_{s}": excl[:, k] for k, s in enumerate(comun.SEDES)}}
        ).to_csv(TABLAS / f"elegibilidad_tau_{tau:.2f}.csv", index=False)
    diag = pd.DataFrame(filas)
    diag.to_csv(TABLAS / "diagnostico_elegibilidad.csv", index=False)
    print(diag[["tau", "pares_excluidos", "pares_excluidos_residential", "pares_excluidos_workplace",
                "pares_excluidos_public", "puertos_S2_en_pares_excluidos", "cota_2D", "u_star_cap",
                "infactible_vs_u_star_archivado"]].to_string(index=False))

    umbrales = [float(x) for x in args.umbrales.split(",") if x.strip()]
    time_limit = args.time_limit or config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap
    print(f"\nPlan NEOS: {len(umbrales)} corrida(s) S2 con topes de uso del suelo, tau = {umbrales}, "
          f"epsilon = {EPSILON_S2} sobre u*_cap, timelimit = {time_limit} s, gap = {mip_gap}; correo {comun.email_neos(config)}")
    if not args.lanzar:
        print("Corrida en seco: no se envió nada a NEOS.")
        return 0

    for tau in umbrales:
        sufijo = f"tau_{tau:.2f}" + ("_holgura_abs" if args.holgura_absoluta else "")
        etiqueta = f"S2_uso_suelo_{sufijo}"
        caps, excl = construir_caps(uso, tau, cuotas)
        u_cap, _ = s0_exacto(ctx.target, cuotas, caps)
        eps_corrida = (u_cap + EPSILON_S2 * u_star) / u_cap - 1.0 if args.holgura_absoluta else EPSILON_S2
        print(f"Lanzando {etiqueta} (u*_cap = {u_cap:.6f}) vía NEOS/CPLEX...", flush=True)
        modelo, _ = solve_equity_refined(
            target=ctx.target, coeffs=ctx.coeffs, u_star=u_cap, config=config,
            nu=ctx.vector_vulnerabilidad, epsilon=eps_corrida, kappa=config.s2_kappa,
            floor_mask=None, floor_reference=None, use_hard_floor=False,
            caps=caps, time_limit=time_limit, mip_gap=mip_gap,
        )
        term = str(getattr(modelo, "_termination_condition", "desconocida"))
        manifiesto = {"corrida": etiqueta, "tau": tau, "epsilon": eps_corrida, "presupuesto_desviacion": u_cap * (1 + eps_corrida), "kappa": config.s2_kappa,
                      "u_star_archivado": u_star, "u_star_cap": u_cap, "time_limit_s": time_limit,
                      "mip_gap": mip_gap, "termination": term, "pares_excluidos": int(excl.sum())}
        if not solve_safe(modelo, accept_feasible=True):
            print(f"  {etiqueta}: sin solución utilizable (termination={term}).")
            (TABLAS / f"manifiesto_{sufijo}.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
            continue
        n_sol, u_sol, a_sol, _ = extract_solution_refined(modelo, len(ctx.df_inputs), len(config.venues))
        try:
            gap_rel = float(modelo._solver_results.problem.upper_bound - modelo._solver_results.problem.lower_bound) / max(
                1e-9, abs(float(modelo._solver_results.problem.upper_bound)))
        except Exception:  # noqa: BLE001  (NEOS no siempre devuelve cotas)
            gap_rel = None
        df_asig, met = summarize_scenario(etiqueta, ctx.df_inputs, n_sol, u_sol, a_sol, ctx.tmat,
                                          ctx.mascara_alta_vulnerabilidad, config=config)
        df_asig.to_csv(TABLAS / f"asignacion_{sufijo}.csv", index=False)
        pc = ports_per_capita_by_quartile(df_asig, VULN_CSV, config.vulnerability_top_share)
        manifiesto.update({
            "gap_relativo_reportado": gap_rel,
            "violaciones_de_tope": int((n_sol > caps + 1e-9).sum()),
            "topes_activos": int((excl & (n_s2 > 0)).sum()),
            "metricas_precision_completa": {k: v for k, v in met.items() if k != "scenario"},
            "puertos_10k_cuartil_vulnerable": pc[0] if pc else None,
            "puertos_10k_resto": pc[1] if pc else None,
            "comparacion_vs_S2": comun.comparar_asignaciones(df_asig, s2),
        })
        (TABLAS / f"manifiesto_{sufijo}.json").write_text(json.dumps(manifiesto, indent=2, default=float), encoding="utf-8")
        print(f"  {etiqueta}: termination={term}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
