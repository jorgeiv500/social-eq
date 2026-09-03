#!/usr/bin/env python
"""B4 — Contrafactual de la señal de prioridad: ¿cambia el ST-GNN la decisión de planificación?

Responde de fondo a R1.2/R1.3 (revisión JTRG-D-26-00692). B1 mostró que un bosque aleatorio
estático iguala al ST-GNN en predicción puntual y B2 que la etapa temporal (omega) sí cambia
la asignación. Faltaba la prueba directa que pide el revisor: correr la MISMA cadena
S0 -> S1 -> S2 con la prioridad zonal s_i producida por modelos más simples y medir cuánto
cambia la asignación final. Aquí la señal temporal omega_iv se deja en su valor aprendido
(ya ablacionada en B2), de modo que solo cambia la prioridad espacial s_i.

Modelos de prioridad (todos normalizados a [0, 1] con `normalize01`, como `priority_norm`):
  A  etiqueta directa   — la propia pseudo-etiqueta y (media horaria del flujo de la
                          encuesta) sin ningún modelo: lo que haría un planificador con la
                          encuesta en la mano;
  B  regresión lineal   — OLS estática estricta (26 features, OOF 5-fold; B1 variante B);
  C  bosque aleatorio   — RF estática estricta (26 features, OOF 5-fold; B1 variante B);
  D  ST-GNN             — la prioridad publicada (referencia archivada, no se re-corre).

Ninguna cifra inventada: y y las predicciones OOF salen de
`../B1_baselines/resultados/b1_predicciones_oof.csv`; todo lo demás es el pipeline oficial
vía import (`comun.preparar_contexto`, `make_targets`, `solve_*`). Las predicciones
negativas de OLS se truncan a 0 antes de normalizar (una demanda negativa no es una
prioridad); la decisión se declara en el manifiesto.

SEGURIDAD: por defecto corrida en seco (diagnóstico sin NEOS + plan). Solo `--lanzar` envía
trabajos: 3 corridas por modelo (S0m, S1m, S2m), 9 en total para A, B y C.

Uso:
    python run_b4.py                      # corrida en seco
    python run_b4.py --lanzar             # 9 corridas NEOS
    python run_b4.py --lanzar --modelos C # solo el bosque aleatorio (3 corridas)
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
    extract_solution,
    extract_solution_refined,
    make_targets,
    normalize01,
    solve_equity,
    solve_equity_refined,
    solve_s0,
    solve_safe,
    summarize_scenario,
)

AQUI = Path(__file__).resolve().parent
DIR_SALIDAS = AQUI / "salidas"
DIR_TABLAS = DIR_SALIDAS / "tablas"
RUTA_B1 = AQUI.parent / "B1_baselines" / "resultados" / "b1_predicciones_oof.csv"

MODELOS = {
    "A": ("etiqueta_directa", "y"),
    "B": ("ols_estatica_estricta", "pred_oof__ols_estatica_estricta_B"),
    "C": ("rf_estatica_estricta", "pred_oof__rf_estatica_estricta_B"),
}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B4: cadena S0-S2 con prioridades de modelos simples.")
    parser.add_argument("--lanzar", action="store_true", help="Enviar los trabajos a NEOS (sin la bandera: solo plan).")
    parser.add_argument("--modelos", default="ABC", help="Subconjunto de modelos a correr, p. ej. 'C' o 'AC'.")
    parser.add_argument("--time-limit", type=int, default=None, help="Límite del solver (s), solo S2.")
    parser.add_argument("--mip-gap", type=float, default=None, help="Gap MIP relativo, solo S2.")
    return parser.parse_args()


def spearman(a, b) -> float:
    valor = pd.Series(np.asarray(a, dtype=float)).corr(pd.Series(np.asarray(b, dtype=float)), method="spearman")
    return float(valor) if pd.notna(valor) else float("nan")


def prioridades_alternativas(df_inputs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Vectores s_i alternativos alineados con df_inputs['UTAM']."""
    b1 = pd.read_csv(RUTA_B1)
    b1["UTAM"] = b1["utam"].astype(str).str.strip().str.upper()
    base = pd.DataFrame({"UTAM": df_inputs["UTAM"].astype(str).str.strip().str.upper()})
    junto = base.merge(b1, on="UTAM", how="left")
    if junto[["y", *[col for _, col in MODELOS.values()]]].isna().any().any():
        faltan = junto.loc[junto["y"].isna(), "UTAM"].tolist()
        raise RuntimeError(f"UTAM sin fila en B1: {faltan[:10]} ...")
    salida = {}
    for clave, (_nombre, col) in MODELOS.items():
        crudo = np.clip(junto[col].to_numpy(dtype=float), 0.0, None)  # demanda negativa -> 0
        salida[clave] = normalize01(pd.Series(crudo)).to_numpy(dtype=float)
    return salida


def conjunto_con_puertos(df_asig: pd.DataFrame) -> set[str]:
    return set(df_asig.loc[df_asig["ports_total"] > 0, "UTAM"].astype(str))


def main() -> int:
    args = parsear_argumentos()
    seleccion = [m for m in "ABC" if m in args.modelos.upper()]
    if not seleccion:
        raise SystemExit("--modelos debe contener al menos una de A, B, C")

    print("Preparando insumos (mismo camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config = ctx.config
    n_zonas, n_sedes = len(ctx.df_inputs), len(config.venues)
    prioridad_paper = ctx.df_inputs["priority_norm"].to_numpy(dtype=float)
    alternativas = prioridades_alternativas(ctx.df_inputs)

    # --- Diagnóstico sin NEOS: cuánto se desplazan los targets por modelo ---
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    filas_diag = []
    variantes: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for clave in seleccion:
        df_m = ctx.df_inputs.copy()
        df_m["priority_norm"] = alternativas[clave]
        _, target_m = make_targets(df_m, config)
        variantes[clave] = (df_m, target_m)
        desplaz = {s: float(np.abs(target_m[:, k] - ctx.target[:, k]).sum() / 2.0) for k, s in enumerate(config.venues)}
        filas_diag.append(
            {
                "modelo": clave,
                "nombre": MODELOS[clave][0],
                "zonas_prioridad_cero": int((alternativas[clave] <= 1e-9).sum()),
                "rho_prioridad_vs_stgnn": spearman(alternativas[clave], prioridad_paper),
                "rho_target_total_vs_stgnn": spearman(target_m.sum(axis=1), ctx.target.sum(axis=1)),
                **{f"desplazamiento_target_{s}": v for s, v in desplaz.items()},
                "desplazamiento_target_total": float(sum(desplaz.values())),
            }
        )
    diag = pd.DataFrame(filas_diag)
    diag.to_csv(DIR_TABLAS / "diagnostico_prioridades.csv", index=False)

    asignaciones_paper = {esc: comun.leer_asignacion_paper(esc) for esc in ("S0", "S1", "S2")}
    for esc in asignaciones_paper:
        asignaciones_paper[esc]["UTAM"] = asignaciones_paper[esc]["UTAM"].astype(str)
    metricas_paper = comun.leer_metricas_paper()
    u_star_paper = comun.leer_u_star()
    eps_s1 = max(config.epsilon_relax, config.epsilon_s1)
    eps_s2 = max(config.epsilon_relax, config.epsilon_s2)
    time_limit = args.time_limit if args.time_limit is not None else config.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config.solver_mip_gap

    print("\n=== Plan del contrafactual B4 ===")
    print(f"Zonas: {n_zonas} | variables enteras: {n_zonas * n_sedes} | ST-GNN (D): zonas con prioridad ~0 = {int((prioridad_paper <= 1e-9).sum())}")
    print(diag.to_string(index=False))
    print(f"Corridas NEOS previstas: {3 * len(seleccion)} (S0m, S1m eps={eps_s1}, S2m eps={eps_s2} kappa={config.s2_kappa}) para modelos {''.join(seleccion)}")
    print(f"Correo NEOS que se usará: {comun.email_neos(config)}")
    if not args.lanzar:
        print("\nCorrida en seco: NO se envió nada a NEOS. Ejecute con --lanzar para correr.")
        return 0

    ruta_resumen = DIR_TABLAS / "resumen_contrafactual.csv"
    filas: list[dict] = []
    for esc in ("S0", "S1", "S2"):
        fp = metricas_paper.loc[metricas_paper["scenario"].astype(str).eq(esc)].iloc[0]
        filas.append(
            {
                "fuente": f"archivo_paper_{esc}", "modelo": "D", "nombre": "stgnn_publicado", "escenario": esc,
                "termination": "archivada", "u_star_modelo": u_star_paper,
                "objective_abs_dev": float(fp["objective_abs_dev"]),
                "perdida_fidelidad_vs_S0_pct": 100.0 * (float(fp["objective_abs_dev"]) - u_star_paper) / u_star_paper,
                "gini_no_ponderado_6d": comun.redondear6(fp["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(fp["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(fp["coverage_within_threshold_vulnerable"]),
                "rho_prioridad_propia_puertos": float(fp["rho_priority_ports"]),
                "zonas_con_puertos": int((asignaciones_paper[esc]["ports_total"] > 0).sum()),
            }
        )
    pd.DataFrame(filas).to_csv(ruta_resumen, index=False)

    for clave in seleccion:
        df_m, target_m = variantes[clave]
        nombre = MODELOS[clave][0]
        u_star_m: float | None = None

        def registrar(etiqueta: str, esc_ref: str, n_sol, u_sol, term: str, dur: float, params: dict) -> None:
            assert u_star_m is not None
            acceso = comun.calcular_accesibilidad(n_sol, ctx.coeffs, config)  # omega aprendida: mismos coeficientes
            df_asig, met = summarize_scenario(etiqueta, df_m, n_sol, u_sol, acceso, ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config)
            df_asig["UTAM"] = df_asig["UTAM"].astype(str)
            ruta_asig = DIR_TABLAS / f"asignacion_B4_{clave}_{esc_ref}.csv"
            df_asig.to_csv(ruta_asig, index=False)
            ref = asignaciones_paper[esc_ref]
            cambios = comun.comparar_asignaciones(df_asig, ref)
            junto = df_asig[["UTAM", "ports_total"]].merge(ref[["UTAM", "ports_total", "priority_norm"]], on="UTAM", suffixes=("_m", "_ref"))
            con_m, con_ref = conjunto_con_puertos(df_asig), conjunto_con_puertos(ref)
            fila = {
                "fuente": "corrida_nueva", "modelo": clave, "nombre": nombre, "escenario": esc_ref,
                "termination": term, "u_star_modelo": u_star_m,
                "objective_abs_dev": float(met["objective_abs_dev"]),
                "perdida_fidelidad_vs_S0_pct": 100.0 * (float(met["objective_abs_dev"]) - u_star_m) / u_star_m,
                "gini_no_ponderado_6d": comun.redondear6(met["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(met["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(met["coverage_within_threshold_vulnerable"]),
                "rho_prioridad_propia_puertos": float(met["rho_priority_ports"]),
                "rho_prioridad_stgnn_puertos": spearman(junto["priority_norm"], junto["ports_total_m"]),
                "rho_asignacion_vs_paper": spearman(junto["ports_total_m"], junto["ports_total_ref"]),
                "zonas_con_puertos": len(con_m),
                "zonas_con_puertos_paper": len(con_ref),
                "jaccard_zonas_con_puertos_vs_paper": len(con_m & con_ref) / max(len(con_m | con_ref), 1),
                "zonas_cambiadas_total_vs_paper": cambios["zonas_cambiadas_total"],
                "zonas_cambiadas_sede_vs_paper": cambios["zonas_cambiadas_sede"],
                "puertos_movidos_vs_paper": cambios["puertos_movidos"],
                "duracion_s": round(dur, 1),
            }
            filas.append(fila)
            pd.DataFrame(filas).to_csv(ruta_resumen, index=False)
            manifiesto = {
                "corrida": etiqueta, "modelo": clave, "nombre": nombre, "escenario_referencia": esc_ref,
                "construccion": "df_inputs oficial con priority_norm sustituida por normalize01(max(pred, 0)) del modelo; omega aprendida sin cambios; targets recomputados con make_targets",
                "fuente_prioridad": str(RUTA_B1.relative_to(comun.RAIZ)),
                "u_star_modelo": u_star_m, "u_star_paper_solo_contexto": u_star_paper,
                "parametros": params, "termination": term, "duracion_s": dur,
                "metricas_precision_completa": {k: v for k, v in met.items() if k != "scenario"},
                "comparacion_vs_paper": cambios,
            }
            (DIR_TABLAS / f"manifiesto_B4_{clave}_{esc_ref}.json").write_text(json.dumps(manifiesto, indent=2, default=float), encoding="utf-8")
            print(f"  {etiqueta}: termination={term} ({dur:.0f}s) | zonas con puertos {len(con_m)} (paper {len(con_ref)}) | puertos movidos vs paper {cambios['puertos_movidos']:.0f}")

        print(f"\n[{clave} {nombre}] S0 vía NEOS/CPLEX...", flush=True)
        t0 = time.time()
        modelo_s0, u_star_m = solve_s0(target_m, config=config)
        dur = time.time() - t0
        term = str(getattr(modelo_s0, "_termination_condition", "desconocida"))
        if not solve_safe(modelo_s0):
            print(f"  S0 de {clave} no resolvió (termination={term}); se salta el modelo.")
            filas.append({"fuente": "corrida_nueva", "modelo": clave, "nombre": nombre, "escenario": "S0", "termination": term})
            pd.DataFrame(filas).to_csv(ruta_resumen, index=False)
            continue
        n0, u0, _ = extract_solution(modelo_s0, n_zonas, n_sedes)
        print(f"  u*_{clave} = {u_star_m:.6f} (u* ST-GNN: {u_star_paper:.6f})")
        registrar(f"S0_{clave}", "S0", n0, u0, term, dur, {"etapa": "solve_s0"})

        print(f"[{clave} {nombre}] S1 vía NEOS/CPLEX...", flush=True)
        t0 = time.time()
        modelo_s1 = solve_equity(target_m, ctx.coeffs, u_star_m, config=config, nu=None, epsilon=eps_s1)
        dur = time.time() - t0
        term = str(getattr(modelo_s1, "_termination_condition", "desconocida"))
        if solve_safe(modelo_s1):
            n1, u1, _ = extract_solution(modelo_s1, n_zonas, n_sedes)
            registrar(f"S1_{clave}", "S1", n1, u1, term, dur, {"etapa": "solve_equity", "epsilon": eps_s1, "nu": None})
        else:
            print(f"  ADVERTENCIA: S1 de {clave} sin solución utilizable (termination={term}).")
            filas.append({"fuente": "corrida_nueva", "modelo": clave, "nombre": nombre, "escenario": "S1", "termination": term})
            pd.DataFrame(filas).to_csv(ruta_resumen, index=False)

        print(f"[{clave} {nombre}] S2 vía NEOS/CPLEX...", flush=True)
        t0 = time.time()
        modelo_s2, _ = solve_equity_refined(
            target=target_m, coeffs=ctx.coeffs, u_star=u_star_m, config=config, nu=ctx.vector_vulnerabilidad,
            epsilon=eps_s2, kappa=config.s2_kappa, floor_mask=None, floor_reference=None, use_hard_floor=False,
            caps=config.max_ports_matrix, time_limit=time_limit, mip_gap=mip_gap,
        )
        dur = time.time() - t0
        term = str(getattr(modelo_s2, "_termination_condition", "desconocida"))
        if solve_safe(modelo_s2, accept_feasible=True):
            n2, u2, _, _ = extract_solution_refined(modelo_s2, n_zonas, n_sedes)
            registrar(f"S2_{clave}", "S2", n2, u2, term, dur, {"etapa": "solve_equity_refined", "epsilon": eps_s2, "kappa": config.s2_kappa, "time_limit_s": time_limit, "mip_gap": mip_gap})
        else:
            print(f"  ADVERTENCIA: S2 de {clave} sin solución utilizable (termination={term}).")
            filas.append({"fuente": "corrida_nueva", "modelo": clave, "nombre": nombre, "escenario": "S2", "termination": term})
            pd.DataFrame(filas).to_csv(ruta_resumen, index=False)

    print(f"\nContrafactual terminado. Resumen: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
