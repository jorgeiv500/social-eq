#!/usr/bin/env python
"""A4 — Preparación de topes de capacidad eléctrica por UTAM (sin NEOS).

Reproduce y extiende el cruce espacial documentado en
`datos/MANIFIESTO.md` (verificación de fuentes) y en
`experiments/revision_2026/A4_topes_capacidad/datos/MANIFIESTO.md`:

1. Holgura por subestación en la red de 11.4 kV:
   `headroom_MVA = CAPACIDAD_ × (1 − CARGA_11)` (ambos atributos publicados por
   la SDP en el conjunto "Capacidad eléctrica. POT Bogotá D.C.", CC BY 4.0).
2. Superposición en EPSG:9377 con el shapefile UTAM2023 ya usado por el
   proyecto; la holgura de cada subestación se reparte entre las UTAM en
   proporción al área intersectada de su zona de servicio.
3. NOVEDAD frente al demo (`datos_a4/cap_i_demo_utam.csv`): se calcula y
   publica la **fracción de cobertura** de cada UTAM por las zonas de servicio
   mapeadas, porque la regla de topes del experimento depende de ella
   (una intersección de astilla no es dato de capacidad de toda la UTAM).

Salida: `salidas/tablas/capacidad_utam.csv` con columnas
    UTAM, area_utam_km2, area_cubierta_km2, cobertura_frac, cap_i_MVA
y validación cruzada contra el demo archivado (misma metodología => mismos
valores de cap_i dentro de tolerancia numérica).

No consulta la red ni NEOS; todo es local y reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

GPKG = AQUI / "datos" / "capacidad_electrica_pot555.gpkg"
DEMO = AQUI / "datos" / "cap_i_demo_utam.csv"
UTAM_SHP = RAIZ / ".tmp_build/utam_shp/UTAM2023/UTAM2023.shp"
SALIDA = AQUI / "salidas/tablas/capacidad_utam.csv"

CRS_TRABAJO = "EPSG:9377"  # origen nacional MAGNA-SIRGAS, métrico (el mismo del demo)


def main() -> int:
    import geopandas as gpd

    for ruta in (GPKG, DEMO, UTAM_SHP):
        if not ruta.exists():
            raise SystemExit(f"Falta el insumo {ruta}")

    subest = gpd.read_file(GPKG).to_crs(CRS_TRABAJO)
    utam = gpd.read_file(UTAM_SHP).to_crs(CRS_TRABAJO)

    # Holgura publicada por subestación (red 11.4 kV).
    subest = subest[["ESTACION", "CAPACIDAD_", "CARGA_11", "geometry"]].copy()
    subest["headroom_MVA"] = subest["CAPACIDAD_"].astype(float) * (1.0 - subest["CARGA_11"].astype(float))
    # El GPKG trae una fila por polígono de la zona de servicio; la holgura es
    # POR SUBESTACIÓN, así que se reparte por área dentro de cada subestación.
    subest["area_pieza"] = subest.geometry.area
    area_subest = subest.groupby("ESTACION")["area_pieza"].transform("sum")
    subest["frac_pieza"] = subest["area_pieza"] / area_subest

    utam = utam[["UTAM", "geometry"]].copy()
    utam["area_utam_m2"] = utam.geometry.area

    inter = gpd.overlay(utam, subest, how="intersection", keep_geom_type=True)
    inter["area_inter_m2"] = inter.geometry.area
    # Reparto proporcional al área intersectada dentro de cada subestación
    # (idéntico al demo): cap aportado = headroom × (área intersectada / área
    # total de la zona de servicio de esa subestación).
    area_zona_servicio = subest.groupby("ESTACION")["area_pieza"].sum().rename("area_servicio_m2")
    inter = inter.merge(area_zona_servicio, on="ESTACION", how="left")
    inter["cap_aportada_MVA"] = inter["headroom_MVA"] * inter["area_inter_m2"] / inter["area_servicio_m2"]

    por_utam = (
        inter.groupby("UTAM")
        .agg(area_cubierta_m2=("area_inter_m2", "sum"), cap_i_MVA=("cap_aportada_MVA", "sum"))
        .reset_index()
    )
    tabla = utam.drop(columns="geometry").merge(por_utam, on="UTAM", how="left")
    tabla[["area_cubierta_m2", "cap_i_MVA"]] = tabla[["area_cubierta_m2", "cap_i_MVA"]].fillna(0.0)
    tabla["cobertura_frac"] = tabla["area_cubierta_m2"] / tabla["area_utam_m2"]
    tabla["area_utam_km2"] = tabla["area_utam_m2"] / 1e6
    tabla["area_cubierta_km2"] = tabla["area_cubierta_m2"] / 1e6
    tabla = tabla[["UTAM", "area_utam_km2", "area_cubierta_km2", "cobertura_frac", "cap_i_MVA"]]
    tabla = tabla.sort_values("UTAM").reset_index(drop=True)

    # Validación cruzada contra el demo archivado (mismo método => mismo cap_i).
    demo = pd.read_csv(DEMO)
    junto = tabla.merge(demo, on="UTAM", how="outer", indicator=True)
    solo_demo = junto[junto["_merge"] == "right_only"]
    if len(solo_demo):
        print(f"ADVERTENCIA: {len(solo_demo)} UTAM del demo no aparecen en el recálculo:", solo_demo["UTAM"].tolist())
    con_demo = junto[junto["_merge"] == "both"].copy()
    con_demo["dif_MVA"] = (con_demo["cap_i_MVA"] - con_demo["cap_i_demo_utam"]
                           if "cap_i_demo_utam" in con_demo else con_demo["cap_i_MVA"] - con_demo["cap_i_MVA_holgura11kV"])
    max_dif = con_demo["dif_MVA"].abs().max()
    print(f"Validación vs demo: {len(con_demo)} UTAM comparadas; |dif| máxima = {max_dif:.6f} MVA")
    if max_dif > 0.05:
        print("ADVERTENCIA: el recálculo difiere del demo en más de 0.05 MVA; revisar metodología antes de usar.")

    print(f"Total holgura repartida: {tabla['cap_i_MVA'].sum():.1f} MVA "
          f"(la verificación reporta ~896–897 MVA en zonas de servicio)")
    for umbral in (0.05, 0.50, 0.95):
        n = int((tabla["cobertura_frac"] >= umbral).sum())
        print(f"UTAM con cobertura >= {umbral:.0%}: {n} de {len(tabla)}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)
    print(f"Escrito {SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
