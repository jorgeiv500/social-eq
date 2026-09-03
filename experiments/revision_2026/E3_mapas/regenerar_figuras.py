"""Regenera las figuras de mapa del manuscrito SIN correr NEOS — Frente E (E3).

Alimenta las funciones de figura de `src/social_eq/followup.py` con los datos
archivados de la corrida oficial del paper:

- shapefile UTAM2023: `outputs/social_equity_followup_outputs/utam_shp/UTAM2023/`
- asignaciones por escenario: `outputs/.../paper_v6_experiments/tables/scenario_S*.csv`
- cambios S2 vs S1: `outputs/.../tables/table_allocation_changes_S2_vs_S1.csv`
  (la figura 6 los recalcula desde scenario_S1/S2 y aquí se verifica que coincidan)
- capa provisional de estaciones existentes (E2, R3.5/R2.m1):
  `experiments/revision_2026/E3_mapas/datos/osm_overpass_bogota_charging.json`

Las figuras nuevas se guardan en `salidas/`. NO se toca `outputs/` ni
la carpeta de figuras del manuscrito (fuera del repositorio).

Uso:
    python experiments/revision_2026/E3_mapas/regenerar_figuras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.affinity import rotate as rotate_geometry
from shapely.geometry import Point

# Raíz del repo: E3_mapas/ -> experimentos/ -> r1_2026-08/ -> revision/ -> raíz
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(RAIZ / "src"))

from social_eq.followup import (  # noqa: E402
    MAP_EDGE,
    MAP_OUTLINE,
    MAP_SCALE,
    PAPER_TEXT,
    REGION_HATCH,
    REGION_HATCH_COLOR,
    ZONE_FILLS,
    ZONE_LEGEND_LABELS,
    add_city_centre_marker,
    add_locality_labels,
    add_north_arrow,
    add_scale_bar,
    draw_map_outline,
    fig_allocation_composite,
    fig_net_change_S2_minus_S1,
    fig_social_context_panels,
    label_halo,
    normalize_zone_code,
    prepare_context_layers,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TABLAS = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
SHAPEFILE = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"
OSM_JSON = Path(__file__).resolve().parent / "datos" / "osm_overpass_bogota_charging.json"
SALIDAS = Path(__file__).resolve().parent / "salidas"

S3_PROTECTED_SHARE = 0.15  # ModelConfig.s3_protected_share de la corrida oficial


def cargar_capas() -> tuple[gpd.GeoDataFrame, float, tuple[float, float]]:
    utam = gpd.read_file(SHAPEFILE)
    utam["UTAM"] = utam["UTAM"].map(normalize_zone_code)
    if utam.crs is None:
        utam = utam.set_crs(4326)
    return prepare_context_layers(utam)


def cargar_escenarios() -> dict[str, pd.DataFrame]:
    escenarios: dict[str, pd.DataFrame] = {}
    for nombre in ("S0", "S1", "S2", "S3"):
        ruta = TABLAS / f"scenario_{nombre}.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta la tabla archivada {ruta}")
        df = pd.read_csv(ruta)
        df["UTAM"] = df["UTAM"].map(normalize_zone_code)
        escenarios[nombre] = df
    return escenarios


def verificar_cambios_vs_tabla(escenarios: dict[str, pd.DataFrame]) -> None:
    """El delta S2-S1 recalculado debe coincidir con la tabla archivada."""
    tabla = pd.read_csv(TABLAS / "table_allocation_changes_S2_vs_S1.csv")
    tabla["UTAM"] = tabla["UTAM"].map(normalize_zone_code)
    delta = escenarios["S2"][["UTAM", "ports_total"]].merge(
        escenarios["S1"][["UTAM", "ports_total"]], on="UTAM", suffixes=("_S2", "_S1")
    )
    delta["net_change"] = delta["ports_total_S2"] - delta["ports_total_S1"]
    recalculado = delta.loc[delta["net_change"].ne(0)].set_index("UTAM")["net_change"].sort_index()
    archivado = tabla.set_index("UTAM")["net_change"].sort_index()
    # `export_section_tables` publica solo las 5 mayores ganancias y las 5 mayores
    # perdidas, asi que la tabla archivada es un SUBCONJUNTO del delta cuando hay mas de
    # cinco cambios de cada signo (con los insumos corregidos hay 8 y 8). La verificacion
    # correcta es que ese subconjunto coincida zona a zona, no que ambos tengan el mismo
    # tamano.
    faltantes = sorted(set(archivado.index) - set(recalculado.index))
    discrepantes = [u for u in archivado.index if u in recalculado.index and recalculado[u] != archivado[u]]
    if faltantes or discrepantes:
        raise RuntimeError(
            f"El delta S2-S1 recalculado no coincide con table_allocation_changes_S2_vs_S1.csv "
            f"(faltan {faltantes}; discrepan {discrepantes}); "
            "revisar insumos antes de regenerar la figura 6."
        )


def cargar_estaciones_osm(rotation_deg: float, origin: tuple[float, float]) -> tuple[gpd.GeoDataFrame, str]:
    """Puntos de carga existentes (OSM Overpass) en el marco rotado del mapa.

    Capa PROVISIONAL (fuente colaborativa); otro agente cerrará el veredicto de
    fuentes y podrá reemplazarla. Devuelve también la fecha base del volcado.
    """
    datos = json.loads(OSM_JSON.read_text(encoding="utf-8"))
    filas = []
    for elemento in datos.get("elements", []):
        if "lat" in elemento and "lon" in elemento:
            lat, lon = elemento["lat"], elemento["lon"]
        elif "center" in elemento:
            lat, lon = elemento["center"]["lat"], elemento["center"]["lon"]
        else:
            continue
        filas.append({"osm_id": elemento.get("id"), "geometry": Point(lon, lat)})
    if not filas:
        raise RuntimeError(f"El crudo OSM {OSM_JSON} no contiene puntos con coordenadas.")
    puntos = gpd.GeoDataFrame(filas, crs=4326).to_crs(3116)
    puntos["geometry"] = puntos.geometry.apply(lambda geom: rotate_geometry(geom, rotation_deg, origin=origin))
    fecha = str(datos.get("osm3s", {}).get("timestamp_osm_base", ""))[:10]
    return puntos, fecha


def fig_study_area_context(
    salidas: Path,
    utam_map: gpd.GeoDataFrame,
    rotation_deg: float,
    estaciones: gpd.GeoDataFrame | None,
    fecha_osm: str,
) -> None:
    """Figura de contexto del área de estudio (E2; base para R3.5/R2.m1).

    Base de tres clases del autor (urbano/rural/región), límites y etiquetas de
    localidades, jerarquía urbana (centro histórico) y, si hay datos, la capa
    provisional de estaciones de carga existentes (OSM).
    """
    extent = tuple(utam_map.total_bounds)
    fig, ax = plt.subplots(1, 1, figsize=(7.05, 3.60), constrained_layout=True)

    for clase, relleno in ZONE_FILLS.items():
        subconjunto = utam_map.loc[utam_map["zone_class"].eq(clase)]
        if not subconjunto.empty:
            subconjunto.plot(ax=ax, color=relleno, edgecolor=MAP_EDGE, linewidth=0.3)
    anillo = utam_map.loc[utam_map["zone_class"].eq("region")]
    if not anillo.empty:
        anillo.plot(
            ax=ax,
            facecolor="none",
            edgecolor=REGION_HATCH_COLOR,
            linewidth=0.5,
            hatch=REGION_HATCH,
            alpha=0.5,
            zorder=4,
        )
    # Límites de localidad (unidad administrativa mayor dentro de Bogotá).
    if "LOCNombre" in utam_map.columns:
        localidades = utam_map.dropna(subset=["LOCNombre"]).dissolve(by="LOCNombre")
        localidades.boundary.plot(ax=ax, color="#9a9186", linewidth=0.75, zorder=5)
    draw_map_outline(ax, utam_map)

    reservado = add_north_arrow(ax, rotation_deg, extent=extent)
    reservado += add_city_centre_marker(ax, utam_map, fontsize=7.2)
    add_locality_labels(ax, utam_map, fontsize=7.2, reserved=reservado)

    if estaciones is not None and not estaciones.empty:
        ax.scatter(
            estaciones.geometry.x,
            estaciones.geometry.y,
            s=17,
            color="#1d5a8f",
            edgecolor="white",
            linewidth=0.5,
            zorder=8,
            label="_nolegend_",
        )

    add_scale_bar(ax, utam_map)

    asas = [
        Patch(facecolor=ZONE_FILLS["urban"], edgecolor=MAP_EDGE, label=ZONE_LEGEND_LABELS["urban"]),
        Patch(facecolor=ZONE_FILLS["rural"], edgecolor=MAP_EDGE, label=ZONE_LEGEND_LABELS["rural"]),
        Patch(
            facecolor=ZONE_FILLS["region"],
            edgecolor=REGION_HATCH_COLOR,
            hatch=REGION_HATCH,
            linewidth=0.6,
            label=ZONE_LEGEND_LABELS["region"],
        ),
    ]
    if estaciones is not None and not estaciones.empty:
        asas.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor="#1d5a8f",
                markeredgecolor="white",
                label=f"Existing public charging station (OSM, n = {len(estaciones)})",
            )
        )
    asas.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=10,
            markerfacecolor=PAPER_TEXT,
            markeredgecolor="white",
            label="Historic city centre",
        )
    )
    ax.legend(
        handles=asas,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    ax.set_title(
        "Study area: Bogota and its functional urban region",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
        color=PAPER_TEXT,
        pad=8,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.01)
    # dpi 330: al ancho de impresion (0.98\textwidth) 300 ppp de lienzo dejarian
    # 294 ppp efectivos, justo por debajo del minimo de 300 de Elsevier.
    fig.savefig(salidas / "fig_study_area_context.png", dpi=330, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fecha_txt = fecha_osm or "date not recorded in the export"
    caption = (
        "Fig. X (provisional caption, E2). Study area. The 142 transport analysis zones (UTAM) of the "
        "2023 mobility survey frame, grouped into urban Bogota (111 UTAM), rural Bogota (5 UPR units) "
        "and the ring of 26 neighbouring municipalities (hatched). For 22 of the 26 units in that ring "
        "no usable DANE census population is available, so population-based indicators are reported for "
        "Bogota only. Dots mark existing public EV charging stations as mapped in OpenStreetMap "
        f"(Overpass export, OSM base {fecha_txt}; n = 42). OSM is a collaborative source and may "
        "undercount private or recently opened stations; this layer is provisional and will be replaced "
        "if an official registry is adopted. The star marks the historic city centre; labels name major "
        "localities (Bogota) and municipalities (region)."
    )
    (salidas / "fig_study_area_context_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    SALIDAS.mkdir(parents=True, exist_ok=True)
    utam_map, rotation_deg, origin = cargar_capas()
    if len(utam_map) != 142:
        raise RuntimeError(f"El shapefile debe tener 142 unidades; tiene {len(utam_map)}.")
    conteo = utam_map["zone_class"].value_counts().to_dict()
    if conteo != {"urban": 111, "region": 26, "rural": 5}:
        raise RuntimeError(f"Clasificación de tres clases inesperada: {conteo}")

    escenarios = cargar_escenarios()
    verificar_cambios_vs_tabla(escenarios)

    df_inputs = escenarios["S0"][["UTAM", "priority_norm", "vulnerability_index"]].copy()
    vulnerabilidad = df_inputs["vulnerability_index"].to_numpy(dtype=float)
    mascara_protegida = vulnerabilidad >= np.quantile(vulnerabilidad, 1.0 - S3_PROTECTED_SHARE)

    fig_social_context_panels(SALIDAS, utam_map, df_inputs, mascara_protegida, rotation_deg)
    fig_allocation_composite(SALIDAS, utam_map, escenarios, rotation_deg)
    fig_net_change_S2_minus_S1(SALIDAS, utam_map, escenarios, rotation_deg)

    estaciones, fecha_osm = (None, "")
    if OSM_JSON.exists():
        estaciones, fecha_osm = cargar_estaciones_osm(rotation_deg, origin)
    fig_study_area_context(SALIDAS, utam_map, rotation_deg, estaciones, fecha_osm)

    print(f"Figuras regeneradas en {SALIDAS}:")
    for ruta in sorted(SALIDAS.glob("fig_*.png")):
        print(f"  - {ruta.name}")
    print(f"Rotación aplicada al marco: {rotation_deg:.2f} grados (antihorario).")
    print(f"Unidades protegidas S3 (cuantil {1 - S3_PROTECTED_SHARE:.2f}): {int(mascara_protegida.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
