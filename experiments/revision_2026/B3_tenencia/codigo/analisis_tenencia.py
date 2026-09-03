"""B3 — validación del proxy de necesidad con tenencia de la vivienda (R1.1).

Correlaciona el porcentaje de hogares en arriendo o subarriendo (Encuesta
Multipropósito 2021, DANE–SDP) con las variables del modelo, siguiendo el
diseño registrado ex ante en `../README.md`:

1. principal: UPZ -> UTAM por interpolación areal (EPSG:3116), excluyendo las
   UTAM cubiertas en menos del 50 % por UPZ con dato publicado;
2. chequeo de escala: 20 localidades (empalme directo por `LOCNombre`);
3. Spearman (principal) y Pearson (complementaria), n explícito por fila.

Todo es local y determinista: no usa NEOS ni re-corre ningún escenario.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

AQUI = Path(__file__).resolve().parent
B3 = AQUI.parent
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FUENTE = B3 / "datos_fuente"
SALIDAS = B3 / "salidas"
CRS_METRICO = 3116
COBERTURA_MINIMA = 0.50


def normalizar(texto: str) -> str:
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn"
    )
    return " ".join(sin_tildes.lower().split())


def main() -> None:
    SALIDAS.mkdir(parents=True, exist_ok=True)

    upz_tab = pd.read_csv(FUENTE / "extraidos" / "em2021_tenencia_upz.csv")
    loc_tab = pd.read_csv(FUENTE / "extraidos" / "em2021_tenencia_localidades.csv")
    upz_geo = gpd.read_file(FUENTE / "upz_bogota_ideca.geojson")
    utam = gpd.read_file(RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp")

    upz_geo["cod"] = pd.to_numeric(upz_geo["CODIGO_UPZ"], errors="coerce")
    upz = upz_geo.merge(upz_tab[["cod", "pct_arriendo_subarriendo"]], on="cod", how="left")

    # --- interpolación areal UPZ -> UTAM ---
    utam_m = utam.to_crs(CRS_METRICO)
    upz_m = upz.to_crs(CRS_METRICO)
    utam_m["area_utam"] = utam_m.geometry.area
    corte = gpd.overlay(
        utam_m[["UTAM", "LOCNombre", "area_utam", "geometry"]],
        upz_m[["cod", "pct_arriendo_subarriendo", "geometry"]],
        how="intersection",
    )
    corte["area_int"] = corte.geometry.area
    con_dato = corte.dropna(subset=["pct_arriendo_subarriendo"]).copy()
    con_dato["ponderado"] = con_dato["pct_arriendo_subarriendo"] * con_dato["area_int"]
    agg = con_dato.groupby("UTAM").agg(
        area_con_dato=("area_int", "sum"), suma_ponderada=("ponderado", "sum")
    )
    agg = agg.join(utam_m.set_index("UTAM")["area_utam"])
    agg["cobertura_upz_con_dato"] = agg["area_con_dato"] / agg["area_utam"]
    agg["pct_arriendo_subarriendo"] = agg["suma_ponderada"] / agg["area_con_dato"]
    agg = agg.reset_index()

    excluidas = agg.loc[agg["cobertura_upz_con_dato"] < COBERTURA_MINIMA, ["UTAM", "cobertura_upz_con_dato"]].copy()
    sin_interseccion = set(utam_m["UTAM"]) - set(agg["UTAM"])
    excluidas = pd.concat([
        excluidas.assign(motivo="cobertura de UPZ con dato inferior al 50 %"),
        pd.DataFrame({"UTAM": sorted(sin_interseccion), "cobertura_upz_con_dato": 0.0,
                      "motivo": "sin intersección con la capa UPZ (fuera de Bogotá urbana)"}),
    ], ignore_index=True)
    excluidas.to_csv(SALIDAS / "utam_excluidas.csv", index=False)

    validas = agg.loc[agg["cobertura_upz_con_dato"] >= COBERTURA_MINIMA].copy()

    # --- variables del modelo ---
    vuln = pd.read_csv(RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    prio = pd.read_csv(RAIZ / "data" / "inputs" / "priority_utam_scores.csv")
    s2 = pd.read_csv(RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables" / "scenario_S2.csv")
    col_v = "vulnerability_index"
    col_p = "priority_norm" if "priority_norm" in prio.columns else prio.columns[1]
    df = (validas.merge(vuln[["UTAM", col_v, "dane_population"]], on="UTAM", how="inner")
                 .merge(prio[["UTAM", col_p]], on="UTAM", how="inner")
                 .merge(s2[["UTAM", "ports_total"]], on="UTAM", how="inner"))
    df["ports_per_10k"] = df["ports_total"] / df["dane_population"] * 1e4
    df.to_csv(SALIDAS / "utam_tenencia.csv", index=False)

    # --- chequeo de escala por localidad ---
    loc_tab["clave"] = loc_tab["nombre"].map(normalizar)
    por_loc = (df.merge(utam[["UTAM", "LOCNombre"]], on="UTAM", how="left")
                 .assign(clave=lambda d: d["LOCNombre"].map(normalizar))
                 .groupby("clave")
                 .agg(vulnerability_index=(col_v, "mean"), priority_norm=(col_p, "mean"),
                      ports_total=("ports_total", "sum"), dane_population=("dane_population", "sum")))
    por_loc = por_loc.join(loc_tab.set_index("clave")["pct_arriendo_subarriendo"], how="inner")
    por_loc["ports_per_10k"] = por_loc["ports_total"] / por_loc["dane_population"] * 1e4

    filas = []
    for nivel, datos, cols in [
        ("UTAM (interpolación areal)", df, [(col_v, "vulnerability_index"), (col_p, "priority_norm"),
                                            ("ports_total", "ports_total_S2"), ("ports_per_10k", "ports_per_10k_S2")]),
        ("Localidad (chequeo de escala)", por_loc, [("vulnerability_index", "vulnerability_index"),
                                                    ("priority_norm", "priority_norm"),
                                                    ("ports_total", "ports_total_S2"),
                                                    ("ports_per_10k", "ports_per_10k_S2")]),
    ]:
        for col, etiqueta in cols:
            sub = datos[["pct_arriendo_subarriendo", col]].dropna()
            rho, p_rho = spearmanr(sub["pct_arriendo_subarriendo"], sub[col])
            r, p_r = pearsonr(sub["pct_arriendo_subarriendo"], sub[col])
            filas.append({"nivel": nivel, "variable": etiqueta, "n": len(sub),
                          "spearman_rho": rho, "spearman_p": p_rho,
                          "pearson_r": r, "pearson_p": p_r})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDAS / "tabla_correlaciones.csv", index=False)

    # --- figura (en inglés, apta para el suplemento) ---
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.6))
    for eje, (col, titulo) in zip(ejes, [(col_v, "Vulnerability index"),
                                         (col_p, "ST-GNN priority (normalized)"),
                                         ("ports_total", "Allocated ports (S2)")]):
        eje.scatter(df["pct_arriendo_subarriendo"], df[col], s=18, alpha=0.75, color="#2F6F8F")
        rho, _ = spearmanr(df["pct_arriendo_subarriendo"], df[col])
        eje.set_xlabel("Households renting or subletting (%)")
        eje.set_ylabel(titulo)
        eje.set_title(f"Spearman $\\rho$ = {rho:.3f}", fontsize=10)
        eje.grid(alpha=0.25)
    fig.suptitle("Housing tenure (Multipurpose Survey 2021) against model variables, by UTAM", fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDAS / "fig_tenure_validation.png", dpi=200)
    fig.savefig(SALIDAS / "fig_tenure_validation.pdf")

    print(f"UTAM válidas: {len(df)} · excluidas: {len(excluidas)}")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
