#!/usr/bin/env python
"""A7 — Cruce espacial de las áreas de actividad del POT (Decreto 555 de 2021) con las UTAM.

Sin red y sin NEOS. Lee los crudos de `datos/` (descargados el 2026-10-10, ver
`datos/MANIFIESTO.md`) y el shapefile UTAM2023 del pipeline, y produce por UTAM:

- área de la UTAM y área cubierta por la capa de áreas de actividad (la capa
  solo zonifica suelo urbano y de expansión, art. 240 del Decreto 555);
- área de cada área de actividad (AAE-RVIS, AAE-RAE, AAP-GSU, AAP-RSU, AAGSM, PEMP);
- área de "sectores incompatibles con el uso residencial" (art. 241 num. 3);
- fracción de suelo compatible con cada sede (residential / workplace / public)
  según la correspondencia declarada en el README (art. 240-241).

Salida: `salidas/tablas/uso_suelo_utam.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in AQUI.parents if (p / "pyproject.toml").exists())

AREAS = AQUI / "datos" / "area_actividad_pot555.geojson"
INCOMPAT = AQUI / "datos" / "sectores_incompatibles_residencial_pot555.geojson"
UTAM_SHP = RAIZ / ".tmp_build/utam_shp/UTAM2023/UTAM2023.shp"
SALIDA = AQUI / "salidas" / "tablas" / "uso_suelo_utam.csv"
CRS_TRABAJO = "EPSG:9377"  # MAGNA-SIRGAS origen nacional, métrico (igual que A4)

CODIGOS = ("AAERVIS", "AAERAE", "AAPGSU", "AAPRSU", "AAGSM", "PEMP")

# Correspondencia declarada (README §"Regla"): qué áreas de actividad son compatibles con cada sede.
# residential: zonas cuyo uso principal es el residencial (AAP, ambas zonas; art. 240 num. 2) y la
#   zona receptora de vivienda de interés social del AAE (art. 240 num. 1a); se descuenta el suelo
#   de los "sectores incompatibles con el uso residencial" (art. 241 num. 3).
# workplace y public: suelo donde el POT admite mayor intensidad de usos económicos y de servicios:
#   AAE (ambas zonas; "corredores de alta y media capacidad ... mayor intensidad en la mezcla de
#   sus usos", art. 240 num. 1) y AAGSM ("usos que promueven el desarrollo económico ... servicios
#   para todos los habitantes", art. 240 num. 3).
# PEMP (BIC nacional, p. ej. Centro Histórico): se rige por su propia resolución, que la capa no
#   describe; para no fabricar una restricción se trata como compatible con las tres sedes.
COMPATIBLE = {
    "residential": ("AAPGSU", "AAPRSU", "AAERVIS", "PEMP"),
    "workplace": ("AAERVIS", "AAERAE", "AAGSM", "PEMP"),
    "public": ("AAERVIS", "AAERAE", "AAGSM", "PEMP"),
}


def main() -> int:
    import geopandas as gpd

    for ruta in (AREAS, INCOMPAT, UTAM_SHP):
        if not ruta.exists():
            raise SystemExit(f"Falta el insumo {ruta}")

    utam = gpd.read_file(UTAM_SHP).to_crs(CRS_TRABAJO)[["UTAM", "geometry"]]
    utam["area_utam_m2"] = utam.geometry.area

    areas = gpd.read_file(AREAS).to_crs(CRS_TRABAJO)[["CODIGO_AREA_ACTIVIDAD", "geometry"]]
    desconocidos = set(areas["CODIGO_AREA_ACTIVIDAD"]) - set(CODIGOS)
    if desconocidos:
        raise SystemExit(f"Códigos de área de actividad no previstos: {desconocidos}")
    areas["geometry"] = areas.geometry.make_valid()
    # Disolver por código evita contar dos veces solapes entre polígonos del mismo código.
    areas = areas.dissolve(by="CODIGO_AREA_ACTIVIDAD").reset_index()

    inter = gpd.overlay(utam[["UTAM", "geometry"]], areas, how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area
    piv = inter.pivot_table(index="UTAM", columns="CODIGO_AREA_ACTIVIDAD", values="a", aggfunc="sum").fillna(0.0)
    for c in CODIGOS:
        if c not in piv:
            piv[c] = 0.0
    piv = piv[list(CODIGOS)].add_prefix("m2_")

    # Área cubierta = unión de la capa dentro de la UTAM (no la suma, por si hay solapes entre códigos).
    union = gpd.GeoDataFrame(geometry=[areas.geometry.union_all()], crs=CRS_TRABAJO)
    cub = gpd.overlay(utam[["UTAM", "geometry"]], union, how="intersection", keep_geom_type=True)
    cub["m2_cubierta"] = cub.geometry.area
    cub = cub.groupby("UTAM")["m2_cubierta"].sum()

    inc = gpd.read_file(INCOMPAT).to_crs(CRS_TRABAJO)
    inc["geometry"] = inc.geometry.make_valid()
    inc_u = gpd.GeoDataFrame(geometry=[inc.geometry.union_all()], crs=CRS_TRABAJO)
    # Suelo residencial-compatible que cae en sectores incompatibles (se descuenta).
    res_geom = areas[areas["CODIGO_AREA_ACTIVIDAD"].isin(COMPATIBLE["residential"])]
    res_inc = gpd.overlay(res_geom, inc_u, how="intersection", keep_geom_type=True)
    res_inc = gpd.overlay(utam[["UTAM", "geometry"]], res_inc, how="intersection", keep_geom_type=True)
    res_inc["a"] = res_inc.geometry.area
    res_inc = res_inc.groupby("UTAM")["a"].sum().rename("m2_res_en_incompatible")

    tabla = utam.drop(columns="geometry").set_index("UTAM")
    tabla = tabla.join(piv).join(cub).join(res_inc).fillna(0.0)
    tabla["cobertura_frac"] = (tabla["m2_cubierta"] / tabla["area_utam_m2"]).clip(upper=1.0)
    suma_codigos = tabla[[f"m2_{c}" for c in CODIGOS]].sum(axis=1)
    tabla["solape_codigos_m2"] = suma_codigos - tabla["m2_cubierta"]

    denom = tabla["m2_cubierta"].where(tabla["m2_cubierta"] > 0)
    for sede, cods in COMPATIBLE.items():
        m2 = tabla[[f"m2_{c}" for c in cods]].sum(axis=1)
        if sede == "residential":
            m2 = (m2 - tabla["m2_res_en_incompatible"]).clip(lower=0.0)
        tabla[f"m2_compatible_{sede}"] = m2
        # Fracción del suelo zonificado (cubierto) que es compatible con la sede.
        tabla[f"frac_compatible_{sede}"] = (m2 / denom).clip(upper=1.0)
    for c in CODIGOS:
        tabla[f"frac_{c}"] = tabla[f"m2_{c}"] / denom
    tabla["area_utam_km2"] = tabla["area_utam_m2"] / 1e6
    tabla = tabla.reset_index().sort_values("UTAM")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)
    print(f"UTAM: {len(tabla)}; escrito {SALIDA}")
    print(f"Solape máximo entre códigos dentro de una UTAM: {tabla['solape_codigos_m2'].max():.1f} m2")
    for u in (0.05, 0.50, 0.95):
        print(f"UTAM con cobertura >= {u:.0%}: {int((tabla['cobertura_frac'] >= u).sum())}")
    print(tabla[[f"frac_compatible_{s}" for s in COMPATIBLE]].describe().round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
