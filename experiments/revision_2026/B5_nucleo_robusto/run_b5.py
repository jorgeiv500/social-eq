#!/usr/bin/env python
"""B5 — Núcleo robusto y arrepentimiento cruzado entre señales de demanda (sin NEOS).

Convierte el contrafactual B4 (¿cambia el modelo de demanda la asignación?) en un resultado de
robustez para el planificador, usando SOLO asignaciones ya resueltas:

  A  etiqueta directa (encuesta, sin modelo)   -> B4_contrafactual_prioridad/salidas/tablas
  B  regresión lineal estática estricta        -> idem
  C  bosque aleatorio estático estricto        -> idem
  D  ST-GNN publicado                          -> comun.leer_asignacion_paper (tablas archivadas)

Análisis:
  1. Núcleo robusto: n_core[i,v] = min_m n_iv^m sobre las cuatro señales. Su suma son los
     puertos que cualquiera de los cuatro modelos asignaría a esa zona y sede ("sin
     arrepentimiento"). Se reporta por cuartil de vulnerabilidad y las zonas en disputa
     (rango max - min de puertos por zona).
  2. Arrepentimiento cruzado: la asignación S2 de cada señal m se evalúa contra los targets de
     cada señal k (make_targets con la prioridad de k): D_k(X_m) = sum_iv |n_iv^m - t_iv^k|,
     normalizada por u*_k (óptimo S0 de k). Matriz 4x4 y asignación minimax.
  3. Estabilidad de equidad: Gini ponderado y no ponderado, cobertura a 15 min y provisión del
     cuartil vulnerable (puertos por 10.000 habitantes) bajo las cuatro señales.
  4. Cota local de robustez en fidelidad: MIP pequeño resuelto con HiGHS EN LOCAL (no NEOS):
     min r  s.a. D_k(X) <= u*_k (1 + r) para las cuatro señales, cuotas por sede, X entero.
     Da la menor tolerancia común a la que una sola asignación es fiel a los cuatro modelos;
     es la cota que fija el epsilon mínimo de la S2 robusta preparada en run_s2_robusta.py.

Ninguna cifra inventada: todo se calcula desde las tablas archivadas y el pipeline oficial
(`comun.preparar_contexto`, `make_targets`, `weighted_gini`, `ports_per_capita_by_quartile`).
Los u*_k se leen de las corridas NEOS ya hechas y se verifican en local con el óptimo exacto de
S0 (que es separable por sede y se resuelve por redondeo).

Uso:
    .venv/bin/python experiments/revision_2026/B5_nucleo_robusto/run_b5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI.parent / "B4_contrafactual_prioridad"))

import comun  # noqa: E402
import run_b4  # noqa: E402  (solo se importan funciones; main() no se ejecuta)
from social_eq.followup import (  # noqa: E402
    coverage_within_threshold,
    gini,
    make_targets,
    normalize_zone_code,
    ports_per_capita_by_quartile,
    weighted_gini,
)

DIR_SALIDAS = AQUI / "salidas"
DIR_TABLAS_B4 = AQUI.parent / "B4_contrafactual_prioridad" / "salidas" / "tablas"
RUTA_POBLACION = comun.RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv"
SENALES = {
    "A": "etiqueta_directa",
    "B": "lineal_estatica",
    "C": "bosque_aleatorio_estatico",
    "D": "stgnn_publicado",
}
ESCENARIOS = ("S0", "S1", "S2")
LIMITE_HIGHS_S = 240  # segundos para el MIP minimax local; se reporta la cota inferior


def leer_asignacion(clave: str, escenario: str, orden_utam: list[str]) -> np.ndarray:
    """Matriz entera (zonas x sedes) alineada con el orden de df_inputs."""
    if clave == "D":
        df = comun.leer_asignacion_paper(escenario)
    else:
        df = pd.read_csv(DIR_TABLAS_B4 / f"asignacion_B4_{clave}_{escenario}.csv")
    df = df.copy()
    df["UTAM"] = df["UTAM"].map(normalize_zone_code)
    df = df.set_index("UTAM").reindex(orden_utam)
    if df[list(comun.COLUMNAS_PUERTOS)].isna().any().any():
        raise RuntimeError(f"La asignación {clave}/{escenario} no cubre todas las UTAM.")
    return df[list(comun.COLUMNAS_PUERTOS)].to_numpy(dtype=int)


def u_star_exacto(target: np.ndarray, cuotas: list[int]) -> float:
    """Óptimo de S0 sin NEOS: min sum|n - t| con sum_i n_iv = Q_v, n entero >= 0.

    Separable por sede. Partiendo de floor(t), cada unidad adicional en la zona i cuesta
    1 - 2 frac(t_i) la primera vez y +1 las siguientes; como sum floor(t) <= Q, el óptimo
    asigna el remanente a las fracciones más grandes.
    """
    total = 0.0
    for v, q in enumerate(cuotas):
        t = target[:, v]
        base = np.floor(t)
        resto = int(round(q - base.sum()))
        n = base.copy()
        if resto > 0:
            orden = np.argsort(-(t - base), kind="stable")
            n[orden[:resto]] += 1  # resto <= número de zonas en este caso; se verifica abajo
            if resto > len(t):
                raise RuntimeError("Remanente mayor que el número de zonas; no previsto.")
        total += float(np.abs(n - t).sum())
    return total


def desviacion(n: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(n - target).sum())


def minimax_fidelidad_local(targets: dict[str, np.ndarray], u_star: dict[str, float], cuotas: list[int]):
    """MIP local con HiGHS (no NEOS): min r s.a. D_k(X) <= u*_k (1 + r) para toda señal k."""
    from pyomo.environ import (
        ConcreteModel, Constraint, NonNegativeIntegers, NonNegativeReals, Objective, RangeSet,
        Var, minimize, value,
    )

    claves = list(targets)
    n_z, n_v = next(iter(targets.values())).shape
    m = ConcreteModel()
    m.I = RangeSet(0, n_z - 1)
    m.V = RangeSet(0, n_v - 1)
    m.K = RangeSet(0, len(claves) - 1)
    m.n = Var(m.I, m.V, domain=NonNegativeIntegers)
    m.u = Var(m.K, m.I, m.V, domain=NonNegativeReals)
    m.r = Var(domain=NonNegativeReals)
    m.cuota = Constraint(m.V, rule=lambda mm, v: sum(mm.n[i, v] for i in mm.I) == int(cuotas[v]))
    m.a1 = Constraint(m.K, m.I, m.V, rule=lambda mm, k, i, v: mm.u[k, i, v] >= mm.n[i, v] - float(targets[claves[k]][i, v]))
    m.a2 = Constraint(m.K, m.I, m.V, rule=lambda mm, k, i, v: mm.u[k, i, v] >= float(targets[claves[k]][i, v]) - mm.n[i, v])
    m.fiel = Constraint(
        m.K,
        rule=lambda mm, k: sum(mm.u[k, i, v] for i in mm.I for v in mm.V) <= float(u_star[claves[k]]) * (1 + mm.r),
    )
    m.obj = Objective(expr=m.r, sense=minimize)
    from pyomo.contrib.appsi.solvers import Highs

    opt = Highs()
    opt.config.time_limit = LIMITE_HIGHS_S
    opt.config.load_solution = False
    opt.config.stream_solver = True
    opt.highs_options = {"mip_rel_gap": 1e-4, "random_seed": 0, "time_limit": float(LIMITE_HIGHS_S)}
    res = opt.solve(m)
    res.solution_loader.load_vars()
    n_sol = np.array([[int(round(value(m.n[i, v]))) for v in range(n_v)] for i in range(n_z)])
    cota = getattr(res, "best_objective_bound", float("nan"))
    return n_sol, float(value(m.r)), str(res.termination_condition), float(cota)


def metricas_equidad(n: np.ndarray, ctx, orden_utam: list[str]) -> dict[str, float]:
    acceso = comun.calcular_accesibilidad(n, ctx.coeffs, ctx.config)
    vuln = ctx.vector_vulnerabilidad
    df = pd.DataFrame({"UTAM": orden_utam, "ports_total": n.sum(axis=1), "vulnerability_index": vuln})
    prov = ports_per_capita_by_quartile(df, RUTA_POBLACION, ctx.config.vulnerability_top_share)
    mask = ctx.mascara_alta_vulnerabilidad
    return {
        "gini_ponderado": comun.redondear6(weighted_gini(acceso, 1.0 + vuln)),
        "gini_no_ponderado": comun.redondear6(gini(acceso)),
        "cobertura_15min_cuartil_vulnerable": float(
            coverage_within_threshold(n, ctx.tmat, mask, ctx.config.coverage_threshold_min)
        ),
        "puertos_por_10k_cuartil_vulnerable": float(prov[0]) if prov else float("nan"),
        "puertos_por_10k_resto": float(prov[1]) if prov else float("nan"),
        "puertos_cuartil_vulnerable": int(n.sum(axis=1)[mask].sum()),
        "zonas_con_puertos": int((n.sum(axis=1) > 0).sum()),
        "zonas_vulnerables_con_puertos": int((n.sum(axis=1)[mask] > 0).sum()),
    }


def main() -> int:
    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)
    print("Preparando insumos con el camino de código oficial (sin NEOS)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS / "aux", config=None)
    config = ctx.config
    cuotas = [int(config.quotas[s]) for s in config.venues]
    orden_utam = ctx.df_inputs["UTAM"].map(normalize_zone_code).tolist()

    # --- Prioridades y targets por señal (mismo procedimiento que B4) ---
    alternativas = run_b4.prioridades_alternativas(ctx.df_inputs)
    targets: dict[str, np.ndarray] = {}
    for clave in "ABC":
        df_m = ctx.df_inputs.copy()
        df_m["priority_norm"] = alternativas[clave]
        targets[clave] = make_targets(df_m, config)[1]
    targets["D"] = ctx.target

    # --- u* por señal: archivado (NEOS) y verificación exacta local ---
    resumen_b4 = pd.read_csv(DIR_TABLAS_B4 / "resumen_contrafactual.csv")
    u_star: dict[str, float] = {}
    filas_u = []
    for clave in SENALES:
        if clave == "D":
            u_arch = comun.leer_u_star()
        else:
            fila = resumen_b4.loc[resumen_b4["modelo"].eq(clave) & resumen_b4["escenario"].eq("S0")].iloc[0]
            u_arch = float(fila["u_star_modelo"])
        u_loc = u_star_exacto(targets[clave], cuotas)
        u_star[clave] = u_loc  # óptimo exacto de S0 (el archivado de A quedó dentro del gap MIP del 1 %)
        filas_u.append({"senal": clave, "nombre": SENALES[clave], "u_star_archivado_neos": u_arch,
                        "u_star_exacto_local": u_loc, "diferencia": u_loc - u_arch,
                        "desviacion_realizada_S0_archivada": desviacion(leer_asignacion(clave, "S0", orden_utam), targets[clave])})
    tabla_u = pd.DataFrame(filas_u)
    tabla_u.to_csv(DIR_SALIDAS / "b5_u_star_verificacion.csv", index=False)
    print(tabla_u.to_string(index=False))

    asign = {(c, e): leer_asignacion(c, e, orden_utam) for c in SENALES for e in ESCENARIOS}
    total_sistema = int(sum(cuotas))
    for (c, e), n in asign.items():
        if [int(x) for x in n.sum(axis=0)] != cuotas:
            raise RuntimeError(f"Asignación {c}/{e} no cumple las cuotas.")

    vuln = ctx.vector_vulnerabilidad
    cortes = np.quantile(vuln, [0.25, 0.50, 0.75])
    cuartil = 1 + (vuln >= cortes[0]).astype(int) + (vuln >= cortes[1]).astype(int) + (vuln >= cortes[2]).astype(int)
    if not np.array_equal(cuartil == 4, ctx.mascara_alta_vulnerabilidad):
        raise RuntimeError("El cuartil 4 no coincide con la máscara de alta vulnerabilidad del paper.")
    pobl = pd.read_csv(RUTA_POBLACION)
    pobl["UTAM"] = pobl["UTAM"].map(normalize_zone_code)
    poblacion = pobl.set_index("UTAM").reindex(orden_utam)["dane_population"].fillna(0.0).to_numpy()

    # === 1. Núcleo robusto ===
    filas_nucleo, filas_cuartil = [], []
    for e in ESCENARIOS:
        pila = np.stack([asign[(c, e)] for c in SENALES])  # señales x zonas x sedes
        nucleo, techo = pila.min(axis=0), pila.max(axis=0)
        tot_zona = pila.sum(axis=2)  # señales x zonas
        rango_zona = tot_zona.max(axis=0) - tot_zona.min(axis=0)
        filas_nucleo.append({
            "escenario": e,
            "puertos_sistema": total_sistema,
            "puertos_nucleo": int(nucleo.sum()),
            "fraccion_nucleo": nucleo.sum() / total_sistema,
            **{f"nucleo_{s}": int(nucleo[:, k].sum()) for k, s in enumerate(config.venues)},
            **{f"fraccion_nucleo_{s}": nucleo[:, k].sum() / cuotas[k] for k, s in enumerate(config.venues)},
            "puertos_nucleo_por_zona_total": int(tot_zona.min(axis=0).sum()),
            "fraccion_nucleo_por_zona_total": tot_zona.min(axis=0).sum() / total_sistema,
            "puertos_en_disputa_sum_max_menos_min": int((techo - nucleo).sum()),
            "zonas_con_puertos_bajo_las_4": int((tot_zona.min(axis=0) > 0).sum()),
            "zonas_con_puertos_bajo_alguna": int((tot_zona.max(axis=0) > 0).sum()),
            "zonas_rango_0": int((rango_zona == 0).sum()),
            "zonas_rango_ge_10": int((rango_zona >= 10).sum()),
            "zonas_rango_ge_25": int((rango_zona >= 25).sum()),
            "rango_zona_mediana": float(np.median(rango_zona)),
            "rango_zona_max": int(rango_zona.max()),
        })
        for q in (1, 2, 3, 4):
            mq = cuartil == q
            pob_q = poblacion[mq & (poblacion > 0)].sum()
            fila = {
                "escenario": e, "cuartil_vulnerabilidad": q, "zonas": int(mq.sum()),
                "poblacion_dane": float(poblacion[mq].sum()),
                "puertos_nucleo": int(nucleo[mq].sum()),
                "fraccion_del_nucleo_total": nucleo[mq].sum() / nucleo.sum(),
                "nucleo_por_10k": 1e4 * nucleo[mq & (poblacion > 0)].sum() / pob_q,
            }
            for c in SENALES:
                p_q = int(asign[(c, e)][mq].sum())
                fila[f"puertos_{c}"] = p_q
                fila[f"fraccion_nucleo_sobre_{c}"] = nucleo[mq].sum() / p_q if p_q else float("nan")
            fila["puertos_min_senales"] = min(fila[f"puertos_{c}"] for c in SENALES)
            fila["puertos_max_senales"] = max(fila[f"puertos_{c}"] for c in SENALES)
            filas_cuartil.append(fila)
        if e == "S2":
            zonas = pd.DataFrame({
                "UTAM": orden_utam,
                "cuartil_vulnerabilidad": cuartil,
                "vulnerability_index": vuln,
                "poblacion_dane": poblacion,
                **{f"puertos_{c}": tot_zona[k] for k, c in enumerate(SENALES)},
                "puertos_nucleo_zona_sede": nucleo.sum(axis=1),
                "puertos_min_zona": tot_zona.min(axis=0),
                "puertos_max_zona": tot_zona.max(axis=0),
                "rango_zona": rango_zona,
                **{f"nucleo_{s}": nucleo[:, k] for k, s in enumerate(config.venues)},
            }).sort_values("rango_zona", ascending=False)
            zonas.to_csv(DIR_SALIDAS / "b5_zonas_S2.csv", index=False)
    pd.DataFrame(filas_nucleo).to_csv(DIR_SALIDAS / "b5_nucleo_resumen.csv", index=False)
    pd.DataFrame(filas_cuartil).to_csv(DIR_SALIDAS / "b5_nucleo_por_cuartil.csv", index=False)

    # === 2. Arrepentimiento cruzado (filas: asignación; columnas: señal evaluadora) ===
    candidatos = {f"{e}_{c}": asign[(c, e)] for e in ESCENARIOS for c in SENALES}

    # === 4. Cota local de robustez en fidelidad (HiGHS, no NEOS) ===
    n_rob, r_rob, term_rob, cota_rob = minimax_fidelidad_local(targets, u_star, cuotas)
    if [int(x) for x in n_rob.sum(axis=0)] != cuotas:
        raise RuntimeError("La asignación minimax local no cumple las cuotas.")
    candidatos["minimax_fidelidad_local"] = n_rob

    filas_reg = []
    for nombre, n in candidatos.items():
        fila = {"asignacion": nombre}
        for k in SENALES:
            d = desviacion(n, targets[k])
            fila[f"desviacion_{k}"] = d
            fila[f"perdida_pct_{k}"] = 100.0 * (d - u_star[k]) / u_star[k]
            # u*_k es solo el error de redondeo entero (~100 sobre 10.870 puertos), así que la
            # pérdida relativa a u* se dispara; la lectura útil es en puertos: como sum n = sum t
            # por sede, (D_k - u*_k)/2 son los puertos colocados fuera del ideal de k.
            fila[f"exceso_puertos_{k}"] = (d - u_star[k]) / 2.0
            fila[f"exceso_pct_sistema_{k}"] = 100.0 * (d - u_star[k]) / 2.0 / total_sistema
        fila["exceso_puertos_max"] = max(fila[f"exceso_puertos_{k}"] for k in SENALES)
        fila["exceso_pct_sistema_max"] = 100.0 * fila["exceso_puertos_max"] / total_sistema
        fila["perdida_max_pct"] = max(fila[f"perdida_pct_{k}"] for k in SENALES)
        fila["senal_peor"] = max(SENALES, key=lambda k: fila[f"perdida_pct_{k}"])
        fila["perdida_media_pct"] = float(np.mean([fila[f"perdida_pct_{k}"] for k in SENALES]))
        fila.update(metricas_equidad(n, ctx, orden_utam))
        filas_reg.append(fila)
    regret = pd.DataFrame(filas_reg)
    regret.to_csv(DIR_SALIDAS / "b5_arrepentimiento_cruzado.csv", index=False)

    s2 = regret[regret["asignacion"].str.startswith("S2_")]
    ganador = s2.loc[s2["perdida_max_pct"].idxmin()]
    matriz = s2.set_index("asignacion")[[f"perdida_pct_{k}" for k in SENALES]]
    matriz.columns = [f"evaluada_con_{k}" for k in SENALES]
    matriz.to_csv(DIR_SALIDAS / "b5_matriz_arrepentimiento_S2.csv")
    matriz_p = s2.set_index("asignacion")[[f"exceso_puertos_{k}" for k in SENALES]]
    matriz_p.columns = [f"evaluada_con_{k}" for k in SENALES]
    matriz_p.to_csv(DIR_SALIDAS / "b5_matriz_exceso_puertos_S2.csv")

    # === 3. Estabilidad de equidad bajo las cuatro señales (S2) ===
    estabilidad = regret[regret["asignacion"].isin([f"S2_{c}" for c in SENALES] + ["minimax_fidelidad_local"])][
        ["asignacion", "gini_ponderado", "gini_no_ponderado", "cobertura_15min_cuartil_vulnerable",
         "puertos_por_10k_cuartil_vulnerable", "puertos_por_10k_resto", "puertos_cuartil_vulnerable",
         "zonas_con_puertos", "zonas_vulnerables_con_puertos"]
    ]
    estabilidad.to_csv(DIR_SALIDAS / "b5_estabilidad_equidad.csv", index=False)
    pd.DataFrame({"UTAM": orden_utam, **{f"ports_{s}": n_rob[:, k] for k, s in enumerate(config.venues)},
                  "ports_total": n_rob.sum(axis=1)}).to_csv(DIR_SALIDAS / "b5_asignacion_minimax_fidelidad_local.csv", index=False)

    # Cotejo con las cifras ya publicadas en B4 (Gini ponderado de S2)
    cotejo = {}
    for c in "ABC":
        fila = resumen_b4.loc[resumen_b4["modelo"].eq(c) & resumen_b4["escenario"].eq("S2")].iloc[0]
        cotejo[c] = {"b4": float(fila["gini_ponderado_6d"]),
                     "b5": float(regret.loc[regret["asignacion"].eq(f"S2_{c}"), "gini_ponderado"].iloc[0])}
    fila_d = resumen_b4.loc[resumen_b4["modelo"].eq("D") & resumen_b4["escenario"].eq("S2")].iloc[0]
    cotejo["D"] = {"b4": float(fila_d["gini_ponderado_6d"]),
                   "b5": float(regret.loc[regret["asignacion"].eq("S2_D"), "gini_ponderado"].iloc[0])}

    manifiesto = {
        "descripcion": "B5 núcleo robusto y arrepentimiento cruzado; sin corridas NEOS",
        "fuentes": {"B4": str(DIR_TABLAS_B4.relative_to(comun.RAIZ)), "paper": str(comun.TABLAS_PAPER.relative_to(comun.RAIZ)),
                    "poblacion": str(RUTA_POBLACION.relative_to(comun.RAIZ))},
        "u_star": u_star,
        "minimax_S2": {"asignacion": ganador["asignacion"], "perdida_max_pct": float(ganador["perdida_max_pct"]),
                       "exceso_puertos_max": float(ganador["exceso_puertos_max"])},
        "minimax_fidelidad_local": {"r_mejor_factible_pct": 100.0 * r_rob, "cota_inferior_r_pct": 100.0 * cota_rob, "termination": term_rob, "limite_s": LIMITE_HIGHS_S, "solver": "HiGHS local (appsi_highs)"},
        "cotejo_gini_ponderado_S2": cotejo,
    }
    (DIR_SALIDAS / "b5_manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n== Núcleo robusto ==")
    print(pd.DataFrame(filas_nucleo).T.to_string())
    print("\n== Núcleo por cuartil (S2) ==")
    print(pd.DataFrame(filas_cuartil).query("escenario == 'S2'").T.to_string())
    print("\n== Matriz de pérdida de fidelidad cruzada (S2, %) ==")
    print(matriz.round(2).to_string())
    print(matriz_p.round(1).to_string())
    print(f"Minimax entre las S2: {ganador['asignacion']} ({ganador['perdida_max_pct']:.2f} %)")
    print(f"Minimax local de fidelidad: r = {100 * r_rob:.3f} % (cota {100 * cota_rob:.3f} %, {term_rob})")
    print("\n== Estabilidad de equidad ==")
    print(estabilidad.to_string(index=False))
    print("\nCotejo Gini ponderado S2 (B4 vs recalculado):", cotejo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
