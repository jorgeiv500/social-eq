#!/usr/bin/env python
"""A7 — Tabla resumen de las corridas con elegibilidad por uso del suelo frente al S2 archivado.

Lee los manifiestos `salidas/tablas/manifiesto_*.json` que escribe `run_a7.py --lanzar` y las
tablas archivadas del paper. Local, sin NEOS. Escribe `salidas/tablas/resumen_uso_suelo.csv`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

import comun  # noqa: E402
from social_eq.followup import ports_per_capita_by_quartile  # noqa: E402

TABLAS = AQUI / "salidas" / "tablas"
VULN_CSV = comun.RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv"


def main() -> int:
    met = comun.leer_metricas_paper().set_index("scenario").loc["S2"]
    pc = ports_per_capita_by_quartile(comun.leer_asignacion_paper("S2"), VULN_CSV)
    u_star = comun.leer_u_star()
    filas = [{
        "corrida": "S2 archivado (sin uso del suelo)", "tau": None, "termination": "archivada",
        "u_star_ref": u_star, "presupuesto_desviacion": u_star * 1.08,
        "objective_abs_dev": float(met["objective_abs_dev"]),
        "pares_excluidos": 0, "topes_activos": 0, "violaciones_de_tope": 0,
        "puertos_movidos_vs_S2": 0.0, "zonas_cambiadas_total_vs_S2": 0, "zonas_cambiadas_sede_vs_S2": 0,
        "gini_ponderado_6d": comun.redondear6(met["accessibility_weighted_gini"]),
        "gini_no_ponderado_6d": comun.redondear6(met["accessibility_gini"]),
        "cobertura_15min_vulnerables": float(met["coverage_within_threshold_vulnerable"]),
        "share_acceso_cuartil_vulnerable": float(met["share_access_most_vulnerable_quartile"]),
        "puertos_10k_cuartil_vulnerable": pc[0], "puertos_10k_resto": pc[1],
    }]
    for ruta in sorted(TABLAS.glob("manifiesto_*.json")):
        m = json.loads(ruta.read_text(encoding="utf-8"))
        fila = {"corrida": m["corrida"], "tau": m["tau"], "termination": m["termination"],
                "u_star_ref": m["u_star_cap"], "presupuesto_desviacion": m.get("presupuesto_desviacion"),
                "pares_excluidos": m["pares_excluidos"]}
        if "metricas_precision_completa" in m:
            x = m["metricas_precision_completa"]
            c = m["comparacion_vs_S2"]
            fila.update({
                "objective_abs_dev": x["objective_abs_dev"],
                "topes_activos": m["topes_activos"], "violaciones_de_tope": m["violaciones_de_tope"],
                "puertos_movidos_vs_S2": c["puertos_movidos"],
                "zonas_cambiadas_total_vs_S2": c["zonas_cambiadas_total"],
                "zonas_cambiadas_sede_vs_S2": c["zonas_cambiadas_sede"],
                "gini_ponderado_6d": comun.redondear6(x["accessibility_weighted_gini"]),
                "gini_no_ponderado_6d": comun.redondear6(x["accessibility_gini"]),
                "cobertura_15min_vulnerables": x["coverage_within_threshold_vulnerable"],
                "share_acceso_cuartil_vulnerable": x["share_access_most_vulnerable_quartile"],
                "puertos_10k_cuartil_vulnerable": m["puertos_10k_cuartil_vulnerable"],
                "puertos_10k_resto": m["puertos_10k_resto"],
            })
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(TABLAS / "resumen_uso_suelo.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
