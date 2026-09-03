"""A4 — diagnóstico en seco de los topes de capacidad eléctrica por UTAM (R1.8, R2.1).

Implementa la sección "Diagnóstico previo sin NEOS" del README de este experimento:
para cada escenario de potencia por puerto, convierte la holgura de subestación
repartida a UTAM (`salidas/tablas/capacidad_utam.csv`) en un tope de puertos,
comprueba dónde muerde frente al target del modelo y a la asignación S2 archivada,
y evalúa la cota inferior de desviación forzada.

No envía nada a NEOS: es determinista y local.

Uso:
    python experiments/revision_2026/A4_topes_capacidad/run_a4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from comun import (  # noqa: E402
    COLUMNAS_PUERTOS,
    SEDES,
    leer_asignacion_paper,
    leer_u_star,
    preparar_contexto,
)

# Potencia nominal por puerto (kW). Categorías AFIR / EAFO, ver README §"De MVA a topes".
ESCENARIOS_KW = {"AC7.4": 7.4, "AC22": 22.0, "DC50": 50.0}
# Regla de cobertura declarada en el README: solo se topa lo que está mapeado.
UMBRAL_COBERTURA = 0.50
EPSILON_S2 = 0.08

SALIDAS = AQUI / "salidas" / "tablas"


def main() -> None:
    SALIDAS.mkdir(parents=True, exist_ok=True)
    ctx = preparar_contexto(AQUI / "salidas")
    u_star = leer_u_star()
    s2 = leer_asignacion_paper("S2")

    cap = pd.read_csv(SALIDAS / "capacidad_utam.csv")
    base = ctx.df_inputs[["UTAM"]].copy()
    base["target_total"] = ctx.target.sum(axis=1)
    for k, sede in enumerate(SEDES):
        base[f"target_{sede}"] = ctx.target[:, k]
    base = base.merge(cap[["UTAM", "cobertura_frac", "cap_i_MVA"]], on="UTAM", how="left")
    base = base.merge(
        s2[["UTAM", "ports_total", *COLUMNAS_PUERTOS]], on="UTAM", how="left"
    )
    if base[["cap_i_MVA", "ports_total"]].isna().any().any():
        faltan = base.loc[base["cap_i_MVA"].isna() | base["ports_total"].isna(), "UTAM"]
        raise RuntimeError(f"UTAM sin capacidad o sin asignación S2: {list(faltan)}")

    topada = base["cobertura_frac"] >= UMBRAL_COBERTURA
    filas, resumen = [], {}
    for nombre, kw in ESCENARIOS_KW.items():
        tope = base["cap_i_MVA"] * 1000.0 / kw
        # "Sin tope" se representa con la cuota total del sistema: no restrictivo.
        tope_efectivo = np.where(topada, tope, float(base["target_total"].sum()))
        deficit_target = np.maximum(0.0, base["target_total"] - tope_efectivo)
        deficit_s2 = np.maximum(0.0, base["ports_total"] - tope_efectivo)
        d_total = float(deficit_target.sum())
        # Cota inferior por conservación de masa dentro de cada sede: sum(u) >= 2*D.
        presupuesto = u_star * (1.0 + EPSILON_S2)
        resumen[nombre] = {
            "kw_por_puerto": kw,
            "utam_con_tope": int(topada.sum()),
            "utam_donde_muerde_vs_target": int((deficit_target > 0).sum()),
            "utam_donde_muerde_vs_S2": int((deficit_s2 > 0).sum()),
            "deficit_forzado_D_vs_target": d_total,
            "deficit_max_zona": float(deficit_target.max()),
            "cota_inferior_2D": 2.0 * d_total,
            "presupuesto_fidelidad_u_star_1_08": presupuesto,
            "infactible_a_eps_0_08_demostrable": bool(2.0 * d_total > presupuesto),
            "capacidad_total_topada_puertos": float(np.where(topada, tope, 0.0).sum()),
        }
        for _, r in base.assign(tope=tope_efectivo, def_t=deficit_target, def_s2=deficit_s2).iterrows():
            filas.append(
                {
                    "escenario": nombre,
                    "UTAM": r["UTAM"],
                    "cobertura_frac": r["cobertura_frac"],
                    "cap_i_MVA": r["cap_i_MVA"],
                    "con_tope": bool(r["cobertura_frac"] >= UMBRAL_COBERTURA),
                    "tope_puertos": r["tope"],
                    "target_total": r["target_total"],
                    "ports_total_S2": r["ports_total"],
                    "deficit_vs_target": r["def_t"],
                    "deficit_vs_S2": r["def_s2"],
                }
            )

    pd.DataFrame(filas).to_csv(SALIDAS / "diagnostico_topes.csv", index=False)
    resumen["_meta"] = {
        "fecha": "2026-09-01",
        "u_star": u_star,
        "umbral_cobertura": UMBRAL_COBERTURA,
        "epsilon": EPSILON_S2,
        "cuotas": dict(ctx.config.quotas),
        "fuente_capacidad": "salidas/tablas/capacidad_utam.csv (POT 555, SDP, CC BY 4.0)",
        "sin_neos": True,
    }
    (SALIDAS / "resumen_diagnostico.json").write_text(json.dumps(resumen, indent=2))
    for nombre in ESCENARIOS_KW:
        r = resumen[nombre]
        print(
            f"{nombre:6s} topes={r['utam_con_tope']:3d}  muerde(target)={r['utam_donde_muerde_vs_target']:3d}"
            f"  muerde(S2)={r['utam_donde_muerde_vs_S2']:3d}  D={r['deficit_forzado_D_vs_target']:10.2f}"
            f"  2D={r['cota_inferior_2D']:10.2f}  presupuesto={r['presupuesto_fidelidad_u_star_1_08']:.2f}"
            f"  infactible={r['infactible_a_eps_0_08_demostrable']}"
        )


if __name__ == "__main__":
    main()
