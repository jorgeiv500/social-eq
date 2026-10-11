#!/usr/bin/env python
"""Parte 1b — Candidatos por cuartil de vulnerabilidad y contraste de sesgo de mapeo OSM.

Toma el cuartil de vulnerabilidad ν (cuartil 4 = vulnerable) y la población DANE por UTAM de
`experiments/revision_2026/B5_nucleo_robusto/salidas/b5_zonas_S2.csv`; ese archivo ya verifica que
el cuartil 4 coincide con la máscara de alta vulnerabilidad del artículo (run_b5.py). El área de
cada UTAM sale del shapefile UTAM2023 del pipeline en EPSG:9377.

Requiere haber corrido `parte1_parqueaderos.py` (lee salidas/parqueaderos_candidatos.csv).

Contraste de sesgo de mapeo: OSM es información voluntaria y suele estar más completa donde hay
más mapeadores (zonas centrales y de mayor ingreso). La capa catastral (UAECD) proviene de la
identificación predial oficial y no depende de voluntarios. Si OSM sub-representa las zonas
vulnerables, su participación en el cuartil 4 debe ser menor que la de Catastro. Se reporta el
cociente de participaciones OSM/Catastro por cuartil (1 = igual reparto; <1 = OSM sub-representa).
Advertencia: Catastro cuenta manzanas donde «depósitos y parqueaderos» es el uso PREDOMINANTE, lo
que no es lo mismo que la presencia de un parqueadero; el contraste es indicativo, no una prueba.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in AQUI.parents if (p / "pyproject.toml").exists())
SALIDAS = AQUI / "salidas"
ZONAS_B5 = RAIZ / "experiments" / "revision_2026" / "B5_nucleo_robusto" / "salidas" / "b5_zonas_S2.csv"
UTAM_SHP = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp"
FUENTES = {"OSM": "osm", "ZPP_TerminalTransporte": "zpp", "Catastro_UAECD_2026": "catastro"}


def canon(v):
    m = re.match(r"^(UTAM|UPR)0*(\d+)$", str(v).strip().upper())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def main():
    cand = pd.read_csv(SALIDAS / "parqueaderos_candidatos.csv")
    z = pd.read_csv(ZONAS_B5)[["UTAM", "cuartil_vulnerabilidad", "vulnerability_index", "poblacion_dane"]]
    z["UTAM"] = z["UTAM"].map(canon)
    u = gpd.read_file(UTAM_SHP).to_crs("EPSG:9377")
    u["UTAM"] = u["UTAM"].map(canon)
    u["area_km2"] = u.area / 1e6
    z = z.merge(u[["UTAM", "area_km2"]], on="UTAM", how="left")
    assert z["area_km2"].notna().all() and len(z) == 142

    for f, s in FUENTES.items():
        c = cand[cand.fuente == f]
        z[f"n_{s}"] = z["UTAM"].map(c.groupby("UTAM").size()).fillna(0).astype(int)
        z[f"cupos_{s}"] = z["UTAM"].map(c.groupby("UTAM")["capacidad"].sum()).fillna(0.0)
    z["n_total"] = z[[f"n_{s}" for s in FUENTES.values()]].sum(axis=1)
    z["cupos_conocidos"] = z[[f"cupos_{s}" for s in FUENTES.values()]].sum(axis=1)
    osm = cand[cand.fuente == "OSM"]
    z["area_parking_osm_m2"] = z["UTAM"].map(osm.groupby("UTAM")["area_m2"].sum()).fillna(0.0)
    z.to_csv(SALIDAS / "candidatos_por_utam_vulnerabilidad.csv", index=False)

    filas = []
    for q, g in z.groupby("cuartil_vulnerabilidad"):
        pob, area = g["poblacion_dane"].sum(), g["area_km2"].sum()
        fila = {"cuartil": int(q), "utam": len(g), "utam_con_poblacion": int((g.poblacion_dane > 0).sum()),
                "poblacion_dane": pob, "area_km2": round(area, 2)}
        for col in ["n_total", "n_osm", "n_zpp", "n_catastro", "cupos_conocidos", "cupos_zpp", "cupos_osm"]:
            fila[col] = g[col].sum()
            fila[f"{col}_por_10k_hab"] = round(1e4 * g[col].sum() / pob, 3) if pob else np.nan
            fila[f"{col}_por_km2"] = round(g[col].sum() / area, 3)
        fila["area_parking_osm_ha_por_km2"] = round(g["area_parking_osm_m2"].sum() / 1e4 / area, 3)
        for s in ["total", "osm", "zpp", "catastro"]:
            fila[f"utam_sin_{s}"] = int((g[f"n_{s}"] == 0).sum())
        fila["utam_sin_capacidad_conocida"] = int((g["cupos_conocidos"] == 0).sum())
        filas.append(fila)
    t = pd.DataFrame(filas)
    for s in ["osm", "catastro", "zpp"]:
        t[f"participacion_{s}"] = (t[f"n_{s}"] / t[f"n_{s}"].sum()).round(4)
    t["participacion_poblacion"] = (t["poblacion_dane"] / t["poblacion_dane"].sum()).round(4)
    t["participacion_area"] = (t["area_km2"] / t["area_km2"].sum()).round(4)
    t["cociente_osm_vs_catastro"] = (t["participacion_osm"] / t["participacion_catastro"]).round(3)
    t.to_csv(SALIDAS / "candidatos_por_cuartil_vulnerabilidad.csv", index=False)

    # Robustez del contraste: solo UTAM urbanas con población (excluye UPR/borde rural sin población DANE)
    zu = z[z.poblacion_dane > 0]
    rob = zu.groupby("cuartil_vulnerabilidad")[["n_osm", "n_catastro"]].sum()
    rob = rob / rob.sum()
    rob["cociente"] = rob["n_osm"] / rob["n_catastro"]
    # Correlación de rangos entre UTAM: densidad OSM vs densidad catastro (por km²)
    from scipy.stats import spearmanr
    rho = spearmanr(zu["n_osm"] / zu["area_km2"], zu["n_catastro"] / zu["area_km2"]).statistic
    rho_v_osm = spearmanr(zu["vulnerability_index"], zu["n_osm"] / zu["area_km2"]).statistic
    rho_v_cat = spearmanr(zu["vulnerability_index"], zu["n_catastro"] / zu["area_km2"]).statistic

    vul = z[z.cuartil_vulnerabilidad == 4]
    d = {
        "utam_vulnerables": int(len(vul)),
        "utam_vulnerables_sin_candidato": {s: int((vul[f"n_{s}"] == 0).sum()) for s in ["total", "osm", "zpp", "catastro"]},
        "utam_vulnerables_sin_capacidad_conocida": int((vul["cupos_conocidos"] == 0).sum()),
        "utam_vulnerables_sin_ninguna_fuente_lista": sorted(vul.loc[vul.n_total == 0, "UTAM"]),
        "contraste_solo_utam_con_poblacion": rob.round(4).reset_index().to_dict(orient="records"),
        "spearman_densidad_osm_vs_catastro_utam_pobladas": round(float(rho), 3),
        "spearman_nu_vs_densidad_osm": round(float(rho_v_osm), 3),
        "spearman_nu_vs_densidad_catastro": round(float(rho_v_cat), 3),
    }
    (SALIDAS / "diagnostico_parte1b_vulnerabilidad.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 250)
    print(t.T.to_string())
    print(json.dumps(d, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
