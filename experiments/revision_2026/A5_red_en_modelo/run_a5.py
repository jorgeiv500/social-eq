#!/usr/bin/env python
"""A5 — Restricción de red eléctrica DENTRO del modelo S2 (R1.8, R2.1).

A4 trató la capacidad de subestación como cribado ex post. Aquí la holgura por UTAM entra
como restricción del modelo, en kW y de forma conjunta para las tres sedes:

    sum_v kW_v · n[i,v] <= 1000 · cap_i_MVA      para cada UTAM i con cobertura >= 50 %

(supuestos de A4: factor de potencia ~1, potencia nominal sin simultaneidad, regla de
cobertura del 50 %; las UTAM sin dato quedan sin tope).

Cómo se construye sin tocar src/:
    `solve_s0` y `solve_equity_refined` (src/social_eq/followup.py) arman el modelo y llaman a
    `solve_model` por nombre global del módulo. Durante cada corrida se sustituye
    temporalmente `followup.solve_model` por un envoltorio que añade UNA familia de
    restricciones (`model.red_kw`) al modelo ya construido y después llama al `solve_model`
    original. Todo lo demás (cuotas, desviación absoluta, presupuesto lexicográfico,
    accesibilidad, pares ponderados con kappa=4, parámetros del solver) es el código publicado.

Escenarios de potencia por puerto (clasificación AFIR, Reglamento (UE) 2023/1804, vía EAFO,
citada en A4):
    AC7.4  : 7.4 kW en las tres sedes (confirmación; ningún tope muerde)
    MIXTO  : residencial 7.4, trabajo 22, público 50 kW (tecnología mixta realista)
    DC50   : 50 kW en las tres sedes (cota de estrés)

Regla para el presupuesto de fidelidad (elegida y documentada en el README):
    u*_red = óptimo de S0 con la misma restricción de red (etapa 1 lexicográfica con red);
    S2_red se resuelve con epsilon = 0.08 sobre u*_red. Si el S0 archivado ya cumple la
    restricción, u*_red = u* archivado (la restricción solo quita puntos factibles y el óptimo
    archivado sigue siendo factible), y no se gasta una corrida S0.

SEGURIDAD: por defecto corrida en seco (diagnóstico determinista, sin NEOS).
Uso:
    python run_a5.py                          # diagnóstico, sin NEOS
    python run_a5.py --lanzar --escenario AC7.4
    python run_a5.py --lanzar --escenario MIXTO
    python run_a5.py --lanzar --escenario DC50
    python run_a5.py --consolidar             # tabla resumen con lo que haya corrido
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.environ import Constraint

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

import comun  # noqa: E402
from social_eq import followup  # noqa: E402
from social_eq.followup import (  # noqa: E402
    extract_solution,
    extract_solution_refined,
    ports_per_capita_by_quartile,
    solve_equity_refined,
    solve_s0,
    solve_safe,
    summarize_scenario,
)

DIR_SALIDAS = AQUI / "salidas"
DIR_TABLAS = DIR_SALIDAS / "tablas"
CAPACIDAD = AQUI.parent / "A4_topes_capacidad" / "salidas" / "tablas" / "capacidad_utam.csv"
VULNERABILIDAD_CSV = comun.RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv"

UMBRAL_COBERTURA = 0.50  # misma regla que A4
EPSILON_S2 = 0.08
# kW nominales por puerto, en el orden de config.venues = (residential, workplace, public).
ESCENARIOS = {
    "AC7.4": (7.4, 7.4, 7.4),
    "MIXTO": (7.4, 22.0, 50.0),
    "DC50": (50.0, 50.0, 50.0),
}
TOL = 1e-6


# ---------------------------------------------------------------------------------------------
# Restricción de red inyectada en el modelo publicado
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def restriccion_red(kw: tuple[float, ...], cap_kw: np.ndarray, topada: np.ndarray):
    """Sustituye temporalmente followup.solve_model para añadir sum_v kW_v n[i,v] <= cap_kW_i."""
    original = followup.solve_model
    indices = [int(i) for i in np.flatnonzero(topada)]

    def envoltorio(model, *args, **kwargs):
        if hasattr(model, "red_kw"):
            raise RuntimeError("El modelo ya tiene restricción de red.")
        model.red_kw = Constraint(
            indices,
            rule=lambda mdl, i: sum(float(kw[v]) * mdl.n[i, v] for v in mdl.V) <= float(cap_kw[i]),
        )
        print(f"  restricción de red añadida: {len(indices)} UTAM, kW por sede = {kw}", flush=True)
        return original(model, *args, **kwargs)

    followup.solve_model = envoltorio
    try:
        yield
    finally:
        followup.solve_model = original


# ---------------------------------------------------------------------------------------------
# Insumos y diagnóstico determinista
# ---------------------------------------------------------------------------------------------
def alinear(ctx, df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    out = ctx.df_inputs[["UTAM"]].merge(df[["UTAM", *columnas]], on="UTAM", how="left")
    if out[columnas].isna().any().any():
        raise RuntimeError(f"Faltan UTAM al alinear columnas {columnas}.")
    return out


def matriz_puertos(ctx, df: pd.DataFrame) -> np.ndarray:
    return alinear(ctx, df, list(comun.COLUMNAS_PUERTOS))[list(comun.COLUMNAS_PUERTOS)].to_numpy(dtype=float)


def cargar_capacidad(ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = alinear(ctx, pd.read_csv(CAPACIDAD), ["cobertura_frac", "cap_i_MVA"])
    topada = (cap["cobertura_frac"] >= UMBRAL_COBERTURA).to_numpy()
    cap_kw = 1000.0 * cap["cap_i_MVA"].to_numpy(dtype=float)
    return topada, cap_kw, cap["cobertura_frac"].to_numpy(dtype=float)


def carga_kw(n: np.ndarray, kw: tuple[float, ...]) -> np.ndarray:
    return n @ np.asarray(kw, dtype=float)


def s0_local_highs(target: np.ndarray, quotas: list[int], kw, cap_kw, topada) -> tuple[str, float | None]:
    """S0 con red resuelto localmente con HiGHS (scipy.milp): solo verificación cruzada del diagnóstico."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n, m = target.shape
    nv = n * m  # variables n[i,v] y luego u[i,v]
    c = np.concatenate([np.zeros(nv), np.ones(nv)])
    filas, lo, hi = [], [], []
    for v in range(m):  # cuotas
        r = np.zeros(2 * nv); r[[i * m + v for i in range(n)]] = 1.0
        filas.append(r); lo.append(quotas[v]); hi.append(quotas[v])
    for i in range(n):
        for v in range(m):
            k = i * m + v
            r = np.zeros(2 * nv); r[nv + k] = 1.0; r[k] = -1.0  # u - n >= -t
            filas.append(r); lo.append(-target[i, v]); hi.append(np.inf)
            r = np.zeros(2 * nv); r[nv + k] = 1.0; r[k] = 1.0  # u + n >= t
            filas.append(r); lo.append(target[i, v]); hi.append(np.inf)
    for i in np.flatnonzero(topada):
        r = np.zeros(2 * nv)
        for v in range(m):
            r[i * m + v] = kw[v]
        filas.append(r); lo.append(-np.inf); hi.append(cap_kw[i])
    integralidad = np.concatenate([np.ones(nv), np.zeros(nv)])
    res = milp(c, constraints=LinearConstraint(np.array(filas), lo, hi), integrality=integralidad,
               bounds=Bounds(0, np.inf), options={"mip_rel_gap": 1e-9, "time_limit": 300})
    return ("optimal" if res.status == 0 else f"status_{res.status}:{res.message}"), (float(res.fun) if res.x is not None else None)


def diagnostico(ctx, u_star: float) -> dict:
    topada, cap_kw, cobertura = cargar_capacidad(ctx)
    s0 = matriz_puertos(ctx, comun.leer_asignacion_paper("S0"))
    s2 = matriz_puertos(ctx, comun.leer_asignacion_paper("S2"))
    quotas = [int(ctx.config.quotas[v]) for v in ctx.config.venues]
    out = {}
    for nombre, kw in ESCENARIOS.items():
        carga_t, carga_s0, carga_s2 = carga_kw(ctx.target, kw), carga_kw(s0, kw), carga_kw(s2, kw)
        exceso_t = np.where(topada, np.maximum(0.0, carga_t - cap_kw), 0.0)
        viola_s0 = topada & (carga_s0 > cap_kw + TOL)
        viola_s2 = topada & (carga_s2 > cap_kw + TOL)
        # Cota inferior: cada zona con exceso E_i debe perder >= E_i / max(kW) puertos bajo target,
        # y por conservación de masa en cada sede esos puertos reaparecen sobre target: sum u >= 2 D.
        d_lb = float(exceso_t.sum() / max(kw))
        estado_local, u_local = s0_local_highs(ctx.target, quotas, kw, cap_kw, topada)
        out[nombre] = {
            "kw_por_sede": dict(zip(ctx.config.venues, kw)),
            "utam_con_tope": int(topada.sum()),
            "utam_carga_target_sobre_tope": int((exceso_t > 0).sum()),
            "utam_S0_archivado_viola": int(viola_s0.sum()),
            "utam_S2_archivado_viola": int(viola_s2.sum()),
            "utam_S2_archivado_viola_lista": ctx.df_inputs.loc[viola_s2, "UTAM"].tolist(),
            "utilizacion_max_S2_archivado": float(np.max(carga_s2[topada] / cap_kw[topada])),
            "utam_utilizacion_max_S2_archivado": str(ctx.df_inputs["UTAM"].to_numpy()[topada][np.argmax(carga_s2[topada] / cap_kw[topada])]),
            "utilizacion_max_target": float(np.max(carga_t[topada] / cap_kw[topada])),
            "carga_total_S2_archivado_mw": float(carga_s2.sum() / 1000.0),
            "exceso_kw_target_total": float(exceso_t.sum()),
            "cota_inferior_D_puertos": d_lb,
            "cota_inferior_2D": 2.0 * d_lb,
            "presupuesto_u_star_1_08": u_star * (1.0 + EPSILON_S2),
            "infactible_a_eps_0_08_con_u_star_archivado_demostrable": bool(2.0 * d_lb > u_star * (1.0 + EPSILON_S2)),
            "S0_red_local_highs_estado": estado_local,
            "S0_red_local_highs_u": u_local,
            "regla_u_star": "u* archivado (S0 archivado cumple la red)" if not viola_s0.any() else "u*_red = S0 con red (NEOS)",
            "epsilon_relax_regla_A4_referencia": float((u_star * 1.08 + 2.0 * d_lb) * 1.02 / u_star - 1.0) if d_lb > 0 else 0.0,
        }
    return out


# ---------------------------------------------------------------------------------------------
# Métricas frente a S2 archivado
# ---------------------------------------------------------------------------------------------
def metricas_completas(ctx, nombre, n_sol, u_sol, a_sol, kw, cap_kw, topada, s2_ref: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, met = summarize_scenario(nombre, ctx.df_inputs, n_sol, u_sol, a_sol, ctx.tmat,
                                 ctx.mascara_alta_vulnerabilidad, config=ctx.config)
    carga = carga_kw(n_sol.astype(float), kw)
    df["carga_kw"] = carga
    df["tope_kw"] = np.where(topada, cap_kw, np.nan)
    df["holgura_kw"] = np.where(topada, cap_kw - carga, np.nan)
    viola = topada & (carga > cap_kw + TOL)
    # "Saturada": no cabe ni un puerto más del tipo de menor potencia del escenario.
    saturada = topada & (cap_kw - carga < min(kw) - TOL)
    cambios = comun.comparar_asignaciones(df, s2_ref)
    ref = alinear(ctx, s2_ref, ["ports_total", *comun.COLUMNAS_PUERTOS])
    delta = df["ports_total"].to_numpy() - ref["ports_total"].to_numpy()
    vul = ctx.mascara_alta_vulnerabilidad
    pc = ports_per_capita_by_quartile(df, VULNERABILIDAD_CSV, ctx.config.vulnerability_top_share)
    ref_pc = ports_per_capita_by_quartile(
        ref.assign(vulnerability_index=ctx.df_inputs["vulnerability_index"].to_numpy()), VULNERABILIDAD_CSV,
        ctx.config.vulnerability_top_share)
    fila = {
        "objective_abs_dev": float(u_sol.sum()),
        "gini_no_ponderado_6d": comun.redondear6(met["accessibility_gini"]),
        "gini_ponderado_6d": comun.redondear6(met["accessibility_weighted_gini"]),
        "cobertura_15min_vulnerables": float(met["coverage_within_threshold_vulnerable"]),
        "share_acceso_cuartil_vulnerable": float(met["share_access_most_vulnerable_quartile"]),
        "puertos_10k_cuartil_vulnerable": pc[0] if pc else float("nan"),
        "puertos_10k_resto": pc[1] if pc else float("nan"),
        "puertos_10k_cuartil_vulnerable_S2_archivado": ref_pc[0] if ref_pc else float("nan"),
        "rho_prioridad_puertos": float(met["rho_priority_ports"]),
        "zonas_cambiadas_total_vs_S2": cambios["zonas_cambiadas_total"],
        "zonas_cambiadas_sede_vs_S2": cambios["zonas_cambiadas_sede"],
        "puertos_movidos_vs_S2": cambios["puertos_movidos"],
        "puertos_movidos_por_sede_vs_S2": {
            col: float(np.abs(df[col].to_numpy() - ref[col].to_numpy()).sum() / 2.0) for col in comun.COLUMNAS_PUERTOS
        },
        "topes_violados": int(viola.sum()),
        "topes_saturados": int(saturada.sum()),
        "topes_saturados_lista": ctx.df_inputs.loc[saturada, "UTAM"].tolist(),
        "puertos_ganados_vulnerables": int(np.clip(delta[vul], 0, None).sum()),
        "puertos_perdidos_vulnerables": int(np.clip(-delta[vul], 0, None).sum()),
        "puertos_ganados_no_vulnerables": int(np.clip(delta[~vul], 0, None).sum()),
        "puertos_perdidos_no_vulnerables": int(np.clip(-delta[~vul], 0, None).sum()),
        "saldo_neto_cuartil_vulnerable": int(delta[vul].sum()),
        "zonas_que_pierden": ctx.df_inputs.loc[delta < 0, "UTAM"].tolist(),
        "zonas_que_ganan": ctx.df_inputs.loc[delta > 0, "UTAM"].tolist(),
        "metricas_precision_completa": {k: v for k, v in met.items() if k != "scenario"},
    }
    return df, fila


# ---------------------------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------------------------
def correr(ctx, escenario: str, u_star: float, diag: dict) -> None:
    config = ctx.config
    kw = ESCENARIOS[escenario]
    topada, cap_kw, _ = cargar_capacidad(ctx)
    n_z, n_v = len(ctx.df_inputs), len(config.venues)
    s2_ref = comun.leer_asignacion_paper("S2")
    manif: dict = {"escenario": escenario, "kw_por_sede": dict(zip(config.venues, kw)),
                   "time_limit_s": config.solver_time_limit, "mip_gap": config.solver_mip_gap,
                   "u_star_archivado": u_star, "diagnostico": diag[escenario]}
    t_ini = time.time()

    if diag[escenario]["utam_S0_archivado_viola"] == 0:
        u_red, term_s0, origen = u_star, "no_requerida", "u* archivado (S0 archivado factible con la red)"
    else:
        print(f"[{escenario}] S0 con red vía NEOS/CPLEX...", flush=True)
        t0 = time.time()
        with restriccion_red(kw, cap_kw, topada):
            m0, u_red = solve_s0(ctx.target, config=config)
        term_s0 = str(getattr(m0, "_termination_condition", "desconocida"))
        manif["S0_red_duracion_s"] = round(time.time() - t0, 1)
        if not solve_safe(m0):
            manif.update({"S0_red_termination": term_s0})
            (DIR_TABLAS / f"manifiesto_{escenario}.json").write_text(json.dumps(manif, indent=2, default=str))
            raise SystemExit(f"[{escenario}] S0 con red no resolvió: {term_s0}")
        n0, u0, _ = extract_solution(m0, n_z, n_v)
        a0 = comun.calcular_accesibilidad(n0, ctx.coeffs, config)
        df0, _ = summarize_scenario(f"S0_red_{escenario}", ctx.df_inputs, n0, u0, a0, ctx.tmat,
                                    ctx.mascara_alta_vulnerabilidad, config=config)
        df0.to_csv(DIR_TABLAS / f"asignacion_S0_red_{escenario}.csv", index=False)
        origen = "S0 con red (NEOS/CPLEX)"
    manif.update({"S0_red_termination": term_s0, "u_star_red": u_red, "u_star_red_origen": origen,
                  "u_star_red_sobre_u_star_pct": 100.0 * (u_red - u_star) / u_star,
                  "presupuesto_lex": u_red * (1.0 + EPSILON_S2),
                  "epsilon_equivalente_sobre_u_star_archivado": u_red * (1.0 + EPSILON_S2) / u_star - 1.0})
    print(f"[{escenario}] u*_red = {u_red:.6f} ({origen}); presupuesto = {manif['presupuesto_lex']:.6f}", flush=True)

    print(f"[{escenario}] S2 con red vía NEOS/CPLEX (eps={EPSILON_S2}, kappa={config.s2_kappa})...", flush=True)
    t0 = time.time()
    with restriccion_red(kw, cap_kw, topada):
        m2, _ = solve_equity_refined(
            target=ctx.target, coeffs=ctx.coeffs, u_star=u_red, config=config,
            nu=ctx.vector_vulnerabilidad, epsilon=EPSILON_S2, kappa=config.s2_kappa,
            floor_mask=None, floor_reference=None, use_hard_floor=False, caps=config.max_ports_matrix,
            time_limit=config.solver_time_limit, mip_gap=config.solver_mip_gap,
        )
    term = str(getattr(m2, "_termination_condition", "desconocida"))
    manif.update({"S2_red_termination": term, "S2_red_duracion_s": round(time.time() - t0, 1)})
    if not solve_safe(m2, accept_feasible=True):
        (DIR_TABLAS / f"manifiesto_{escenario}.json").write_text(json.dumps(manif, indent=2, default=str))
        raise SystemExit(f"[{escenario}] S2 con red sin solución utilizable: {term}")
    n2, u2, a2, _ = extract_solution_refined(m2, n_z, n_v)
    df2, fila = metricas_completas(ctx, f"S2_red_{escenario}", n2, u2, a2, kw, cap_kw, topada, s2_ref)
    df2.to_csv(DIR_TABLAS / f"asignacion_S2_red_{escenario}.csv", index=False)
    fila["perdida_fidelidad_vs_u_star_archivado_pct"] = 100.0 * (fila["objective_abs_dev"] - u_star) / u_star
    fila["perdida_fidelidad_vs_u_star_red_pct"] = 100.0 * (fila["objective_abs_dev"] - u_red) / u_red
    try:
        fila["mip_gap_reportado"] = str(m2._solver_results.solver.get("gap", None)) if hasattr(m2, "_solver_results") else None
    except Exception:  # noqa: BLE001
        fila["mip_gap_reportado"] = None
    manif.update(fila)
    manif["duracion_total_s"] = round(time.time() - t_ini, 1)
    (DIR_TABLAS / f"manifiesto_{escenario}.json").write_text(json.dumps(manif, indent=2, default=float))
    print(f"[{escenario}] S2 con red: termination={term}; Σu={fila['objective_abs_dev']:.6f}; "
          f"Gini={fila['gini_no_ponderado_6d']}; Gini_w={fila['gini_ponderado_6d']}; "
          f"movidos={fila['puertos_movidos_vs_S2']}; zonas={fila['zonas_cambiadas_total_vs_S2']}", flush=True)


def desviacion_realizada(ctx, df: pd.DataFrame) -> float:
    """Σ_{i,v} |n_iv − T_iv| recalculada ex post (mismo criterio que scripts/realized_deviation.py):
    la suma de u_iv del solver es la cota del presupuesto, no la desviación incurrida."""
    return float(np.abs(matriz_puertos(ctx, df) - ctx.target).sum())


def consolidar(u_star: float) -> pd.DataFrame:
    ctx = comun.preparar_contexto(DIR_SALIDAS)
    fila_ref = comun.leer_metricas_paper().set_index("scenario").loc["S2"]
    filas = [{
        "escenario": "S2_archivado", "kw_res_trab_pub": "sin red", "termination_S0": "archivada",
        "termination_S2": "archivada", "u_star_usado": u_star, "objective_abs_dev": float(fila_ref["objective_abs_dev"]),
        "gini_no_ponderado_6d": comun.redondear6(fila_ref["accessibility_gini"]),
        "gini_ponderado_6d": comun.redondear6(fila_ref["accessibility_weighted_gini"]),
        "cobertura_15min_vulnerables": float(fila_ref["coverage_within_threshold_vulnerable"]),
        "desviacion_realizada": desviacion_realizada(ctx, comun.leer_asignacion_paper("S2")),
        "desviacion_realizada_S0_usado": desviacion_realizada(ctx, comun.leer_asignacion_paper("S0")),
    }]
    for esc in ESCENARIOS:
        p = DIR_TABLAS / f"manifiesto_{esc}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        filas.append({
            "escenario": esc,
            "kw_res_trab_pub": "/".join(f"{v:g}" for v in m["kw_por_sede"].values()),
            "termination_S0": m.get("S0_red_termination"),
            "termination_S2": m.get("S2_red_termination"),
            "u_star_usado": m.get("u_star_red"),
            "u_star_red_sobre_u_star_pct": m.get("u_star_red_sobre_u_star_pct"),
            "epsilon_equivalente_sobre_u_star": m.get("epsilon_equivalente_sobre_u_star_archivado"),
            "objective_abs_dev": m.get("objective_abs_dev"),
            "desviacion_realizada": desviacion_realizada(ctx, pd.read_csv(DIR_TABLAS / f"asignacion_S2_red_{esc}.csv")),
            "desviacion_realizada_S0_usado": desviacion_realizada(
                ctx, pd.read_csv(DIR_TABLAS / f"asignacion_S0_red_{esc}.csv")
                if (DIR_TABLAS / f"asignacion_S0_red_{esc}.csv").exists() else comun.leer_asignacion_paper("S0")),
            "perdida_fidelidad_vs_u_star_archivado_pct": m.get("perdida_fidelidad_vs_u_star_archivado_pct"),
            "gini_no_ponderado_6d": m.get("gini_no_ponderado_6d"),
            "gini_ponderado_6d": m.get("gini_ponderado_6d"),
            "cobertura_15min_vulnerables": m.get("cobertura_15min_vulnerables"),
            "puertos_10k_cuartil_vulnerable": m.get("puertos_10k_cuartil_vulnerable"),
            "puertos_10k_resto": m.get("puertos_10k_resto"),
            "zonas_cambiadas_total_vs_S2": m.get("zonas_cambiadas_total_vs_S2"),
            "puertos_movidos_vs_S2": m.get("puertos_movidos_vs_S2"),
            "utam_carga_target_sobre_tope": m["diagnostico"]["utam_carga_target_sobre_tope"],
            "utam_S2_archivado_viola": m["diagnostico"]["utam_S2_archivado_viola"],
            "topes_saturados": m.get("topes_saturados"),
            "topes_violados": m.get("topes_violados"),
            "puertos_ganados_vulnerables": m.get("puertos_ganados_vulnerables"),
            "puertos_perdidos_vulnerables": m.get("puertos_perdidos_vulnerables"),
            "saldo_neto_cuartil_vulnerable": m.get("saldo_neto_cuartil_vulnerable"),
        })
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "resumen_a5.csv", index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="A5: restricción de red eléctrica dentro de S2.")
    ap.add_argument("--lanzar", action="store_true", help="Enviar realmente los trabajos a NEOS.")
    ap.add_argument("--escenario", choices=list(ESCENARIOS), default=None)
    ap.add_argument("--consolidar", action="store_true")
    args = ap.parse_args()

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    u_star = comun.leer_u_star()
    if args.consolidar:
        print(consolidar(u_star).to_string(index=False))
        return 0

    print("Preparando insumos (mismo camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS)
    diag = diagnostico(ctx, u_star)
    (DIR_TABLAS / "diagnostico_red.json").write_text(json.dumps(diag, indent=2, default=float))
    for nombre, d in diag.items():
        print(f"{nombre:6s} target>tope={d['utam_carga_target_sobre_tope']:3d}  S0arch viola={d['utam_S0_archivado_viola']:3d}"
              f"  S2arch viola={d['utam_S2_archivado_viola']:3d}  D_lb={d['cota_inferior_D_puertos']:.2f}"
              f"  2D={d['cota_inferior_2D']:.2f}  presup={d['presupuesto_u_star_1_08']:.2f}"
              f"  S0red(HiGHS)={d['S0_red_local_highs_u']}  -> {d['regla_u_star']}")
    print(f"Correo NEOS: {comun.email_neos(ctx.config)}; timelimit={ctx.config.solver_time_limit}s; gap={ctx.config.solver_mip_gap}")
    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS.")
        return 0
    if args.escenario is None:
        raise SystemExit("--lanzar exige --escenario.")
    correr(ctx, args.escenario, u_star, diag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
