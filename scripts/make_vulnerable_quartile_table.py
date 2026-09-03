"""Tabla de provisión por zona en el cuartil vulnerable (Suplemento, Apéndice E).

Para cada una de las 36 UTAM del cuartil superior de vulnerabilidad: localidad, población
censal, puertos asignados en S0 y S2, puertos por 10.000 habitantes en S2 e índice de
vulnerabilidad. Ordenada de menor a mayor provisión per cápita: es la lista de zonas donde una
cuota dirigida tendría más efecto. Escribe el CSV en las tablas del pipeline y un fragmento
LaTeX opcional.

Uso: python scripts/make_vulnerable_quartile_table.py [--tex RUTA.tex]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from social_eq.followup import normalize_zone_code  # noqa: E402

TABLAS = ROOT / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
SHAPEFILE = ROOT / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=None, help="escribir también la tabla LaTeX completa (tabular con booktabs)")
    args = ap.parse_args()

    vul = pd.read_csv(ROOT / "data" / "inputs" / "utam_social_vulnerability.csv")
    vul["UTAM"] = vul["UTAM"].map(normalize_zone_code)
    s0 = pd.read_csv(TABLAS / "scenario_S0.csv"); s0["UTAM"] = s0["UTAM"].map(normalize_zone_code)
    s2 = pd.read_csv(TABLAS / "scenario_S2.csv"); s2["UTAM"] = s2["UTAM"].map(normalize_zone_code)
    loc = gpd.read_file(SHAPEFILE)[["UTAM", "LOCNombre", "MUNNombre"]]
    loc["UTAM"] = loc["UTAM"].map(normalize_zone_code)

    d = (s2[["UTAM", "ports_total", "vulnerability_index"]].rename(columns={"ports_total": "ports_S2"})
         .merge(s0[["UTAM", "ports_total"]].rename(columns={"ports_total": "ports_S0"}), on="UTAM")
         .merge(vul[["UTAM", "dane_population"]], on="UTAM", how="left")
         .merge(loc, on="UTAM", how="left"))
    corte = d["vulnerability_index"].quantile(0.75)
    q = d.loc[d["vulnerability_index"] >= corte].copy()
    q["ports_per_10k_S2"] = 1e4 * q["ports_S2"] / q["dane_population"]
    q["locality"] = q["LOCNombre"].fillna(q["MUNNombre"]).astype(str).str.title()
    q = q.sort_values("ports_per_10k_S2", na_position="last").reset_index(drop=True)
    out = q[["UTAM", "locality", "dane_population", "ports_S0", "ports_S2", "ports_per_10k_S2", "vulnerability_index"]]
    out.to_csv(TABLAS / "table_vulnerable_quartile_provision.csv", index=False)
    print(f"{len(out)} zonas del cuartil vulnerable -> {TABLAS / 'table_vulnerable_quartile_provision.csv'}")
    print(out.round(3).to_string(index=False))

    if args.tex:
        filas = []
        for _, r in out.iterrows():
            pob = f"{int(r.dane_population):,}" if pd.notna(r.dane_population) else "---"
            pc = f"{r.ports_per_10k_S2:.2f}" if pd.notna(r.ports_per_10k_S2) else "---"
            filas.append(f"{r.UTAM} & {r.locality} & {pob} & {int(r.ports_S0)} & {int(r.ports_S2)} & {pc} & {r.vulnerability_index:.3f} \\\\")
        cuerpo = "\n".join(filas)
        tabla = (
            "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
            "UTAM & Locality & Population & Ports S0 & Ports S2 & Ports per 10,000 (S2) & $\\nu_i$ \\\\\n"
            "\\midrule\n" + cuerpo + "\n\\bottomrule\n\\end{tabular}\n"
        )
        args.tex.write_text(tabla, encoding="utf-8")
        print(f"tabla LaTeX completa -> {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
