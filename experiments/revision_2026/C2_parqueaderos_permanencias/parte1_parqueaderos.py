#!/usr/bin/env python
"""Parte 1 — Capa de parqueaderos candidatos en Bogotá (extensión «de la zona al sitio»).

Une tres fuentes descargadas y documentadas en `datos/MANIFIESTO.md`:

  1. OpenStreetMap, `amenity=parking` (nodos, vías cerradas y relaciones multipolígono),
     leído del extracto Geofabrik de Colombia con el controlador OSM de GDAL (vía pyogrio).
     Overpass no respondió el 2026-10-11 (ver manifiesto).
  2. Zonas de Parqueo Pago (ZPP), Terminal de Transporte S.A. — tramos viales con cupos.
  3. Catastro (UAECD), «Grupo de uso económico predominante por manzana», vigencia 2026,
     categoría DEPOSITOS Y PARQUEADEROS (la categoría catastral mezcla depósitos y parqueaderos;
     no se puede separar con esta capa).

Cada candidato queda como PUNTO (nodo OSM; punto representativo de polígonos; punto medio de
tramos ZPP) con su ZAT (ZAT2023.shp, carpeta EM2023_DIR) y su UTAM (shapefile UTAM2023 del
pipeline), ambos cruzados en EPSG:9377. No se imputa ninguna capacidad.

Uso:
    python parte1_parqueaderos.py --pbf RUTA/colombia-261009.osm.pbf
El .pbf (331 MB) no se versiona; el manifiesto da URL y MD5 para recuperarlo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in AQUI.parents if (p / "pyproject.toml").exists())
DATOS, SALIDAS = AQUI / "datos", AQUI / "salidas"
EM2023_DIR = Path(os.environ.get(
    "EM2023_DIR",
    "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes"))
ZAT_SHP = EM2023_DIR / "ZAT2023.shp"
UTAM_SHP = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp"
CARGA_OSM = Path(__file__).resolve().parent / "datos" / "osm_estaciones_bogota.csv"
CRS = "EPSG:9377"
CAJA_BOGOTA = (-74.25, 4.45, -73.98, 4.84)  # lon_min, lat_min, lon_max, lat_max (filtro previo; el definitivo es la ZAT)
ETIQUETAS = ["amenity", "name", "capacity", "parking", "access", "fee", "operator", "building", "park_ride"]


def canon_utam(v) -> str | None:
    m = re.match(r"^(UTAM|UPR)0*(\d+)$", str(v).strip().upper())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def hstore(s) -> dict:
    """Convierte el campo `other_tags` de GDAL ("k"=>"v","k2"=>"v2") en diccionario."""
    if not isinstance(s, str):
        return {}
    return dict(re.findall(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"', s))


def capacidad_num(x):
    """Solo enteros explícitos; cualquier otro texto ('yes', '10-20', ...) queda sin capacidad."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    return float(s) if re.fullmatch(r"\d+", s) else np.nan


# ------------------------------------------------------------------ fuentes
def leer_osm(pbf: Path) -> gpd.GeoDataFrame:
    os.environ.setdefault("OSM_MAX_TMPFILE_SIZE", "1024")
    filas = []
    for capa, tipo_osm in [("points", "node"), ("multipolygons", None), ("lines", "way")]:
        g = pyogrio.read_dataframe(pbf, layer=capa, bbox=CAJA_BOGOTA,
                                   where="other_tags LIKE '%\"amenity\"=>\"parking\"%'" if capa != "multipolygons"
                                   else "amenity = 'parking' OR other_tags LIKE '%\"amenity\"=>\"parking\"%'")
        if g.empty:
            continue
        tags = g["other_tags"].map(hstore)
        for c in ETIQUETAS:
            col = g[c] if c in g.columns else pd.Series(None, index=g.index)
            g["t_" + c] = [col.iloc[i] if pd.notna(col.iloc[i]) else tags.iloc[i].get(c) for i in range(len(g))]
        g = g[g["t_amenity"] == "parking"].copy()
        if capa == "multipolygons":
            g["osm_tipo"] = np.where(g["osm_id"].notna(), "relation", "way")
            g["osm_ref"] = g["osm_id"].fillna(g["osm_way_id"])
        else:
            g["osm_tipo"] = tipo_osm
            g["osm_ref"] = g["osm_id"]
        g["capa_gdal"] = capa
        filas.append(g)
    osm = pd.concat(filas, ignore_index=True)
    osm = gpd.GeoDataFrame(osm, geometry="geometry", crs="EPSG:4326").to_crs(CRS)
    osm["area_m2"] = np.where(osm.geom_type.isin(["Polygon", "MultiPolygon"]), osm.area, np.nan)
    osm["geometry"] = osm.geometry.representative_point()
    out = gpd.GeoDataFrame({
        "id": "osm_" + osm["osm_tipo"] + "_" + osm["osm_ref"].astype(str),
        "fuente": "OSM",
        "nombre": osm["t_name"],
        "capacidad": osm["t_capacity"].map(capacidad_num),
        "capacidad_texto": osm["t_capacity"],
        "tipo": osm["t_parking"].fillna("sin_etiqueta"),
        "acceso": osm["t_access"].fillna("sin_etiqueta"),
        "tarifa": osm["t_fee"],
        "operador": osm["t_operator"],
        "area_m2": osm["area_m2"],
        "cupos_electricos": np.nan,
        "geometria_original": osm["capa_gdal"].map({"points": "punto", "multipolygons": "poligono", "lines": "linea"}),
    }, geometry=osm.geometry, crs=CRS)
    return out.drop_duplicates("id")


def leer_zpp() -> gpd.GeoDataFrame:
    z = gpd.read_file(DATOS / "zonas_parqueo_pago_rest.geojson").to_crs(CRS)
    pts = z.geometry.interpolate(0.5, normalized=True)
    return gpd.GeoDataFrame({
        "id": "zpp_" + z["OBJECTID"].astype(str),
        "fuente": "ZPP_TerminalTransporte",
        "nombre": z["EJE"].astype(str) + " entre " + z["INICIO"].astype(str) + " y " + z["FIN"].astype(str),
        "capacidad": pd.to_numeric(z["CUP_AUT_"], errors="coerce"),  # cupos de automóvil
        "capacidad_texto": "CUP_AUT_=" + z["CUP_AUT_"].astype(str) + "; CUP_TOTAL_=" + z["CUP_TOTAL_"].astype(str),
        "tipo": "via_publica_tarifada",
        "acceso": "public",
        "tarifa": "yes",
        "operador": "Terminal de Transporte S.A.",
        "area_m2": np.nan,
        "cupos_electricos": pd.to_numeric(z["CUP_ELEC_"], errors="coerce"),
        "geometria_original": "linea",
        "estado_zpp": z["OPERATIVO"],
    }, geometry=pts, crs=CRS)


def leer_catastro() -> gpd.GeoDataFrame:
    c = gpd.read_file(DATOS / "catastro_manzanas_depositos_parqueaderos_2026.geojson").to_crs(CRS)
    return gpd.GeoDataFrame({
        "id": "cat_" + c["MANCODIGO"].astype(str),
        "fuente": "Catastro_UAECD_2026",
        "nombre": "Manzana " + c["MANCODIGO"].astype(str),
        "capacidad": np.nan,
        "capacidad_texto": None,
        "tipo": "manzana_depositos_y_parqueaderos",
        "acceso": "desconocido",
        "tarifa": None,
        "operador": None,
        "area_m2": c.area,
        "cupos_electricos": np.nan,
        "geometria_original": "poligono",
    }, geometry=c.geometry.representative_point(), crs=CRS)


# ------------------------------------------------------------------ cruce y diagnóstico
def cargar_zonas():
    zat = gpd.read_file(ZAT_SHP).to_crs(CRS)
    zat = zat[zat["MUNCod"] == 11001][["ZAT", "geometry"]].copy()
    zat["ZAT"] = zat["ZAT"].astype(int)
    utam = gpd.read_file(UTAM_SHP).to_crs(CRS)
    utam["UTAM"] = utam["UTAM"].map(canon_utam)
    return zat, utam[["UTAM", "UTAMNombre", "geometry"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pbf", type=Path, required=True, help="extracto Geofabrik colombia-261009.osm.pbf")
    a = ap.parse_args()
    SALIDAS.mkdir(exist_ok=True)

    zat, utam = cargar_zonas()
    osm, zpp, cat = leer_osm(a.pbf), leer_zpp(), leer_catastro()
    capa = pd.concat([osm, zpp, cat], ignore_index=True)
    capa = gpd.GeoDataFrame(capa, geometry="geometry", crs=CRS)
    capa = gpd.sjoin(capa, zat, how="left", predicate="within").drop(columns="index_right")
    capa = gpd.sjoin(capa, utam, how="left", predicate="within").drop(columns="index_right")
    fuera = capa["ZAT"].isna()
    print(f"descartados fuera de las ZAT de Bogotá: {int(fuera.sum())} "
          f"({capa.loc[fuera, 'fuente'].value_counts().to_dict()})")
    capa = capa[~fuera].copy()
    capa["ZAT"] = capa["ZAT"].astype(int)

    # Posible duplicado OSM ↔ catastro: el punto OSM cae dentro de una manzana catastral de parqueaderos
    manz = gpd.read_file(DATOS / "catastro_manzanas_depositos_parqueaderos_2026.geojson").to_crs(CRS)
    dentro = gpd.sjoin(capa[capa.fuente == "OSM"], manz[["MANCODIGO", "geometry"]], predicate="within")
    capa["osm_en_manzana_catastral"] = capa["id"].isin(dentro["id"])
    capa = capa.reset_index(drop=True)

    capa.to_file(SALIDAS / "parqueaderos_candidatos.gpkg", layer="candidatos", driver="GPKG")
    ll = capa.to_crs("EPSG:4326")
    tabla = pd.DataFrame(capa.drop(columns="geometry"))
    tabla["x_9377"], tabla["y_9377"] = capa.geometry.x.round(2), capa.geometry.y.round(2)
    tabla["lon"], tabla["lat"] = ll.geometry.x.round(7), ll.geometry.y.round(7)
    tabla.to_csv(SALIDAS / "parqueaderos_candidatos.csv", index=False)

    # ---------------- diagnóstico
    d = {}
    d["candidatos_total"] = int(len(capa))
    d["por_fuente"] = capa["fuente"].value_counts().to_dict()
    d["con_capacidad_total"] = int(capa["capacidad"].notna().sum())
    d["pct_con_capacidad_total"] = round(100 * capa["capacidad"].notna().mean(), 1)
    d["pct_con_capacidad_por_fuente"] = (capa.groupby("fuente")["capacidad"].apply(lambda s: round(100 * s.notna().mean(), 1))).to_dict()
    d["cupos_conocidos_por_fuente"] = capa.groupby("fuente")["capacidad"].sum().round(0).to_dict()
    d["zpp_cupos_electricos"] = float(capa.loc[capa.fuente == "ZPP_TerminalTransporte", "cupos_electricos"].sum())
    d["zpp_tramos_con_cupo_electrico"] = int((capa["cupos_electricos"] > 0).sum())
    d["zpp_estado"] = capa.get("estado_zpp", pd.Series(dtype=str)).value_counts().to_dict()
    o = capa[capa.fuente == "OSM"]
    d["osm_tipo_parking"] = o["tipo"].value_counts().to_dict()
    d["osm_acceso"] = o["acceso"].value_counts().to_dict()
    d["osm_tarifa"] = o["tarifa"].fillna("sin_etiqueta").value_counts().to_dict()
    d["osm_con_operador"] = int(o["operador"].notna().sum())
    d["osm_geometria"] = o["geometria_original"].value_counts().to_dict()
    d["osm_en_manzana_catastral"] = int(capa["osm_en_manzana_catastral"].sum())
    # acceso público o sin restricción declarada (excluye private/customers/... explícitos)
    restr = {"private", "no", "customers", "permit", "residents", "delivery", "employees"}
    d["osm_acceso_restringido_explicito"] = int(o["acceso"].isin(restr).sum())

    n_zat, n_utam = len(zat), len(utam)
    zc = capa.groupby("ZAT").size()
    uc = capa.groupby("UTAM").size()
    d["zat_bogota"] = n_zat
    d["zat_sin_candidato"] = int(n_zat - zc.index.isin(zat["ZAT"]).sum())
    d["zat_sin_candidato_osm_zpp"] = int(n_zat - capa[capa.fuente != "Catastro_UAECD_2026"]["ZAT"].nunique())
    d["zat_sin_candidato_con_capacidad"] = int(n_zat - capa[capa.capacidad.notna()]["ZAT"].nunique())
    d["utam_total"] = n_utam
    d["utam_sin_candidato"] = int(n_utam - uc.index.isin(utam["UTAM"]).sum())
    d["utam_sin_candidato_lista"] = sorted(set(utam["UTAM"]) - set(uc.index))
    d["utam_sin_candidato_con_capacidad"] = int(n_utam - capa[capa.capacidad.notna()]["UTAM"].nunique())
    d["candidatos_sin_utam"] = int(capa["UTAM"].isna().sum())

    # cobertura por zona (tabla)
    cob_u = utam[["UTAM", "UTAMNombre"]].copy()
    for f, nom in [("OSM", "n_osm"), ("ZPP_TerminalTransporte", "n_zpp"), ("Catastro_UAECD_2026", "n_catastro")]:
        cob_u[nom] = cob_u["UTAM"].map(capa[capa.fuente == f].groupby("UTAM").size()).fillna(0).astype(int)
    cob_u["n_total"] = cob_u[["n_osm", "n_zpp", "n_catastro"]].sum(axis=1)
    cob_u["cupos_conocidos"] = cob_u["UTAM"].map(capa.groupby("UTAM")["capacidad"].sum()).fillna(0)
    cob_u.sort_values("n_total").to_csv(SALIDAS / "cobertura_por_utam.csv", index=False)
    cob_z = zat[["ZAT"]].copy()
    for f, nom in [("OSM", "n_osm"), ("ZPP_TerminalTransporte", "n_zpp"), ("Catastro_UAECD_2026", "n_catastro")]:
        cob_z[nom] = cob_z["ZAT"].map(capa[capa.fuente == f].groupby("ZAT").size()).fillna(0).astype(int)
    cob_z["n_total"] = cob_z[["n_osm", "n_zpp", "n_catastro"]].sum(axis=1)
    cob_z["cupos_conocidos"] = cob_z["ZAT"].map(capa.groupby("ZAT")["capacidad"].sum()).fillna(0)
    cob_z.to_csv(SALIDAS / "cobertura_por_zat.csv", index=False)

    # comparación con los 42 puntos de carga OSM ya usados en la revisión (E2)
    ev = pd.read_csv(CARGA_OSM)
    ev = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.lon, ev.lat), crs="EPSG:4326").to_crs(CRS)
    ev = gpd.sjoin(ev, zat, how="left", predicate="within").drop(columns="index_right")
    ev = gpd.sjoin(ev, utam[["UTAM", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    cand = capa[["id", "fuente", "geometry"]]
    near = gpd.sjoin_nearest(ev, cand, how="left", distance_col="dist_m")
    near = near.sort_values("dist_m").drop_duplicates(["osm_type", "osm_id"])
    near_osm = gpd.sjoin_nearest(ev, cand[cand.fuente == "OSM"], how="left", distance_col="dist_m_osm") \
        .sort_values("dist_m_osm").drop_duplicates(["osm_type", "osm_id"])
    comp = near[["osm_type", "osm_id", "nombre", "operador", "ZAT", "UTAM", "id", "fuente", "dist_m"]].rename(
        columns={"id": "candidato_mas_cercano", "fuente": "fuente_candidato"})
    comp = comp.merge(near_osm[["osm_type", "osm_id", "dist_m_osm"]], on=["osm_type", "osm_id"])
    comp["dist_m"], comp["dist_m_osm"] = comp["dist_m"].round(1), comp["dist_m_osm"].round(1)
    comp.to_csv(SALIDAS / "carga_existente_vs_candidatos.csv", index=False)
    d["carga_osm_puntos"] = int(len(ev))
    d["carga_osm_en_bogota_zat"] = int(ev["ZAT"].notna().sum())
    for r in (50, 100, 250):
        d[f"carga_osm_con_candidato_a_{r}m"] = int((comp["dist_m"] <= r).sum())
        d[f"carga_osm_con_parking_osm_a_{r}m"] = int((comp["dist_m_osm"] <= r).sum())
    d["carga_osm_mediana_dist_m"] = float(comp["dist_m"].median())
    d["carga_osm_en_zat_sin_candidato"] = int((~comp["ZAT"].isin(zc.index) & comp["ZAT"].notna()).sum())
    d["carga_osm_utam_con_carga"] = int(ev["UTAM"].nunique())

    (SALIDAS / "diagnostico_parte1.json").write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(d, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
