#!/usr/bin/env python
"""S2 robusta a la señal de demanda — PREPARADO, NO LANZADO.

Variante de la S2 del paper (min sum_p w_p |A_i - A_j|, w_p = 1 + kappa (nu_i + nu_j)/2) en la
que la restricción lexicográfica de fidelidad se impone SIMULTÁNEAMENTE para las cuatro
señales de demanda de B4/B5 (A encuesta directa, B lineal, C bosque aleatorio, D ST-GNN):

    sum_iv |n_iv - t_iv^k| <= u*_k (1 + r*) (1 + epsilon)     para k = A, B, C, D

donde r* es la tolerancia minimax de fidelidad hallada en local por run_b5.py (HiGHS, se lee
de salidas/b5_manifiesto.json) y epsilon es la holgura de la S2 del paper (0.08 por defecto).
Con epsilon = 0 el problema queda en el borde de la factibilidad (solo la asignación minimax
y sus empates); con epsilon > 0 la etapa de equidad dispone de holgura dentro de la región
fiel a los cuatro modelos.

Tamaño: el mismo de la S2 publicada (426 enteras, 10.011 pares) más 3 x 426 desviaciones
continuas y 3 x 852 restricciones de valor absoluto: una corrida NEOS de la misma clase.

SEGURIDAD: por defecto solo imprime el plan. Únicamente `--lanzar` envía UN trabajo a NEOS.

Uso:
    .venv/bin/python experiments/revision_2026/B5_nucleo_robusto/run_s2_robusta.py            # plan
    .venv/bin/python experiments/revision_2026/B5_nucleo_robusto/run_s2_robusta.py --lanzar   # 1 corrida NEOS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI.parent / "B4_contrafactual_prioridad"))

import comun  # noqa: E402
import run_b4  # noqa: E402
from social_eq.followup import make_targets, solve_model, solve_safe, summarize_scenario  # noqa: E402

DIR_SALIDAS = AQUI / "salidas"


def parsear() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 robusta a las cuatro señales de demanda (preparada).")
    p.add_argument("--lanzar", action="store_true", help="Enviar el trabajo a NEOS (sin la bandera: solo plan).")
    p.add_argument("--epsilon", type=float, default=None, help="Holgura sobre la tolerancia minimax (por defecto epsilon_s2).")
    return p.parse_args()


def construir_modelo(targets: dict[str, np.ndarray], u_star: dict[str, float], factor: float, ctx):
    """Réplica de solve_equity_refined (sin piso ni topes) con cuatro restricciones de fidelidad."""
    from pyomo.environ import (
        ConcreteModel, Constraint, NonNegativeIntegers, NonNegativeReals, Objective, RangeSet, Var, minimize,
    )

    config, nu, coeffs = ctx.config, ctx.vector_vulnerabilidad, ctx.coeffs
    claves = list(targets)
    n, n_v = ctx.target.shape
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pesos = [1.0 + float(config.s2_kappa) * (float(nu[i]) + float(nu[j])) / 2.0 for i, j in pares]
    m = ConcreteModel()
    m.I, m.V, m.P, m.K = RangeSet(0, n - 1), RangeSet(0, n_v - 1), RangeSet(0, len(pares) - 1), RangeSet(0, len(claves) - 1)
    m.n = Var(m.I, m.V, domain=NonNegativeIntegers)
    m.u = Var(m.K, m.I, m.V, domain=NonNegativeReals)
    m.A = Var(m.I, domain=NonNegativeReals)
    m.z = Var(m.P, domain=NonNegativeReals)
    m.quota = Constraint(m.V, rule=lambda mm, v: sum(mm.n[i, v] for i in mm.I) == int(config.quotas[config.venues[v]]))
    m.abs1 = Constraint(m.K, m.I, m.V, rule=lambda mm, k, i, v: mm.u[k, i, v] >= mm.n[i, v] - float(targets[claves[k]][i, v]))
    m.abs2 = Constraint(m.K, m.I, m.V, rule=lambda mm, k, i, v: mm.u[k, i, v] >= float(targets[claves[k]][i, v]) - mm.n[i, v])
    m.lex = Constraint(m.K, rule=lambda mm, k: sum(mm.u[k, i, v] for i in mm.I for v in mm.V) <= float(u_star[claves[k]]) * factor)

    def acceso(mm, i):
        return mm.A[i] == sum(coef * mm.n[j, v] for v, s in enumerate(config.venues) for j, coef in coeffs[s][i])

    m.Adef = Constraint(m.I, rule=acceso)
    m.g1 = Constraint(m.P, rule=lambda mm, p: mm.z[p] >= mm.A[pares[p][0]] - mm.A[pares[p][1]])
    m.g2 = Constraint(m.P, rule=lambda mm, p: mm.z[p] >= mm.A[pares[p][1]] - mm.A[pares[p][0]])
    m.obj = Objective(expr=sum(pesos[p] * m.z[p] for p in m.P), sense=minimize)
    return m


def main() -> int:
    args = parsear()
    ruta_man = DIR_SALIDAS / "b5_manifiesto.json"
    ruta_u = DIR_SALIDAS / "b5_u_star_verificacion.csv"
    if not ruta_man.exists() or not ruta_u.exists():
        raise SystemExit("Primero ejecute run_b5.py (local, sin NEOS) para obtener u*_k y r*.")
    manifiesto = json.loads(ruta_man.read_text(encoding="utf-8"))
    r_star = float(manifiesto["minimax_fidelidad_local"]["r_mejor_factible_pct"]) / 100.0
    tabla_u = pd.read_csv(ruta_u)
    u_star = dict(zip(tabla_u["senal"], tabla_u["u_star_exacto_local"]))

    ctx = comun.preparar_contexto(DIR_SALIDAS / "aux", config=None)
    config = ctx.config
    epsilon = config.epsilon_s2 if args.epsilon is None else float(args.epsilon)
    factor = (1.0 + r_star) * (1.0 + epsilon)
    alternativas = run_b4.prioridades_alternativas(ctx.df_inputs)
    targets = {}
    for clave in "ABC":
        df_m = ctx.df_inputs.copy()
        df_m["priority_norm"] = alternativas[clave]
        targets[clave] = make_targets(df_m, config)[1]
    targets["D"] = ctx.target

    n = len(ctx.df_inputs)
    print("=== Plan S2 robusta (no se envía nada sin --lanzar) ===")
    print(f"r* minimax local = {100 * r_star:.3f} % | epsilon = {epsilon} | factor sobre u*_k = {factor:.4f}")
    for k in targets:
        print(f"  {k}: u*_k = {u_star[k]:.4f} -> presupuesto de desviación {u_star[k] * factor:.2f} "
              f"(~{(u_star[k] * factor - u_star[k]) / 2:.0f} puertos fuera del ideal de {k})")
    print(f"Variables enteras: {n * len(config.venues)} | pares: {n * (n - 1) // 2} | corridas NEOS: 1")
    print(f"Correo NEOS que se usaría: {comun.email_neos(config)}")
    if not args.lanzar:
        print("Corrida en seco: NO se envió nada a NEOS.")
        return 0

    modelo = construir_modelo(targets, u_star, factor, ctx)
    solve_model(modelo, config=config, tee=False, time_limit=config.solver_time_limit, mip_gap=config.solver_mip_gap)
    if not solve_safe(modelo, accept_feasible=True):
        print("S2 robusta sin solución utilizable.")
        return 1
    n_sol = np.array([[round(modelo.n[i, v].value) for v in range(len(config.venues))] for i in range(n)])
    u_d = np.array([[modelo.u[3, i, v].value for v in range(len(config.venues))] for i in range(n)])
    acceso = comun.calcular_accesibilidad(n_sol, ctx.coeffs, config)
    df_asig, met = summarize_scenario("S2_robusta", ctx.df_inputs, n_sol, u_d, acceso, ctx.tmat,
                                      ctx.mascara_alta_vulnerabilidad, config=config)
    df_asig.to_csv(DIR_SALIDAS / "asignacion_S2_robusta.csv", index=False)
    (DIR_SALIDAS / "manifiesto_S2_robusta.json").write_text(
        json.dumps({"r_star": r_star, "epsilon": epsilon, "factor": factor, "metricas": met}, indent=2, default=float),
        encoding="utf-8")
    print(met)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
