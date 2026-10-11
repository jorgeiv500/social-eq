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
    python experiments/revision_2026/E3_mapas/regenerar_figuras.py              # Figs. 2, 4, 6 y 7 en salidas/
    python experiments/revision_2026/E3_mapas/regenerar_figuras.py --solo-fig2  # Fig. 2 y copia a latex/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from matplotlib.transforms import Bbox

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
    classify_zone_class,
    draw_map_outline,
    fig_allocation_composite,
    fig_net_change_S2_minus_S1,
    fig_social_context_panels,
    label_halo,
    normalize_zone_code,
    prepare_context_layers,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartografia_jtg as cj  # noqa: E402

TABLAS = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
SHAPEFILE = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"
OSM_JSON = Path(__file__).resolve().parent / "datos" / "osm_overpass_bogota_charging.json"
SALIDAS = Path(__file__).resolve().parent / "salidas"
ALTO_FIG2 = 4.75  # pulgadas; ancho fijo de 180 mm (0.98 textwidth)
LATEX_FIGS = RAIZ / "revision" / "r1_2026-08" / "latex" / "figures"

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


def cargar_estaciones_osm() -> tuple[gpd.GeoDataFrame, str]:
    """Puntos de carga existentes (OSM Overpass) en EPSG:3116, norte arriba (sin rotación).

    Capa PROVISIONAL (fuente colaborativa). Devuelve también la fecha base del volcado.
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
    puntos = gpd.GeoDataFrame(filas, crs=4326).to_crs(cj.CRS_MAPA)
    fecha = str(datos.get("osm3s", {}).get("timestamp_osm_base", ""))[:10]
    return puntos, fecha


def capas_norte_arriba() -> gpd.GeoDataFrame:
    """UTAM2023 en EPSG:3116 sin rotar, con la clase urbano/rural/región del autor."""
    utam = gpd.read_file(SHAPEFILE)
    if utam.crs is None:
        utam = utam.set_crs(4326)
    utam = utam.to_crs(cj.CRS_MAPA)
    utam["UTAM"] = utam["UTAM"].map(normalize_zone_code)
    utam["zone_class"] = utam["UTAM"].map(classify_zone_class)
    return utam


def marco_urbano(utam: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Marco urbano común del estilo cartográfico (URBAN_FRAME_3116), el mismo de las Figs. 4, 6, 7 y 10."""
    x0, y0, x1, y1 = utam.loc[utam["zone_class"].eq("urban")].total_bounds
    m = cj.MARCO_URBANO
    if not (m[0] <= x0 and m[1] <= y0 and x1 <= m[2] and y1 <= m[3]):
        raise RuntimeError("El marco urbano común no contiene las 111 UTAM urbanas.")
    return m


def _pintar_clases(ax, utam: gpd.GeoDataFrame, lw_utam: float) -> None:
    urb = utam.loc[utam["zone_class"].eq("urban")]
    rur = utam.loc[utam["zone_class"].eq("rural")]
    reg = utam.loc[utam["zone_class"].eq("region")]
    reg.plot(ax=ax, facecolor=cj.VECINOS_RELLENO, edgecolor=cj.VECINOS_BORDE, linewidth=lw_utam, zorder=1)
    reg.plot(ax=ax, facecolor="none", edgecolor=cj.VECINOS_RAYADO, hatch="///", linewidth=0, zorder=2)
    reg.boundary.plot(ax=ax, color=cj.VECINOS_BORDE, linewidth=lw_utam, zorder=3)
    rur.plot(ax=ax, facecolor=cj.RURAL, edgecolor="#8f9a80", linewidth=lw_utam, zorder=2)
    urb.plot(ax=ax, facecolor=cj.URBANO, edgecolor="#c2b39a", linewidth=lw_utam, zorder=2)
    # Límite distrital limpio: el buffer de ida y vuelta cierra las astillas entre polígonos UTAM
    # (sin él, la disolución deja trazos sueltos del límite dentro de la ciudad).
    bogota = utam.loc[utam["zone_class"].ne("region")].unary_union.buffer(20).buffer(-20)
    gpd.GeoSeries([bogota], crs=utam.crs).boundary.plot(ax=ax, color="#4D4D4D", linewidth=0.9, zorder=6)
    # Huecos interiores del marco UTAM (3,15 km² al sur, entre Usme y Ciudad Bolívar): sin dato, gris propio
    huecos = [Polygon(r) for r in bogota.interiors if Polygon(r).area > 5e4]
    if huecos:
        gpd.GeoSeries(huecos, crs=utam.crs).plot(ax=ax, facecolor=cj.SIN_DATO, edgecolor="#4D4D4D",
                                                 linewidth=0.5, zorder=4)


def fig_study_area_context(salidas: Path, utam: gpd.GeoDataFrame, estaciones: gpd.GeoDataFrame,
                           fecha_osm: str) -> Path:
    """Fig. 2 (R2.m1, R3.5, R3.6): área de estudio con norte arriba, en dos paneles.

    (a) Distrito completo (con Sumapaz) y anillo de municipios vecinos y rectángulo del panel (b);
    la localización en Colombia va en texto, en la columna de la leyenda. (b) Núcleo urbano (el marco de la Fig. 10) con
    localidades, centro histórico y las 42 estaciones OSM. Leyenda en una columna a la derecha.
    Se dibuja al ancho de impresión (0.98 textwidth = 180 mm), así que los tamaños son reales.
    """
    cj.estilo_base()
    ancho_in = 180 / 25.4
    fig = plt.figure(figsize=(ancho_in, ALTO_FIG2))
    # Posiciones en fracción de figura; (a) y (b) comparten alto
    alto, base = 0.945, 0.010
    xa0, ya0, xa1, ya1 = utam.total_bounds
    pad_a = 2500
    ext_a = (xa0 - pad_a, ya0 - pad_a, xa1 + pad_a, ya1 + pad_a)
    ext_b = marco_urbano(utam)
    alto_in = alto * ALTO_FIG2
    ancho_a = alto_in * (ext_a[2] - ext_a[0]) / (ext_a[3] - ext_a[1]) / ancho_in
    ancho_b = alto_in * (ext_b[2] - ext_b[0]) / (ext_b[3] - ext_b[1]) / ancho_in
    x_a = 0.006
    x_b = x_a + ancho_a + 0.035
    ax_a = fig.add_axes([x_a, base, ancho_a, alto])
    ax_b = fig.add_axes([x_b, base, ancho_b, alto])
    x_ley = x_b + ancho_b + 0.022

    # ---------- (a) Distrito y región ----------
    _pintar_clases(ax_a, utam, lw_utam=0.25)
    ax_a.set_xlim(ext_a[0], ext_a[2]); ax_a.set_ylim(ext_a[1], ext_a[3]); ax_a.set_aspect("equal")
    ax_a.add_patch(Rectangle((ext_b[0], ext_b[1]), ext_b[2] - ext_b[0], ext_b[3] - ext_b[1], fill=False,
                             edgecolor=cj.TEXTO, linewidth=0.9, linestyle=(0, (3, 1.5)), zorder=8))
    ax_a.text(ext_b[2] + 900, ext_b[1] + 1500, "(b)", ha="left", va="bottom", fontsize=cj.TAM_MIN,
              fontweight="bold", color=cj.TEXTO, path_effects=cj.halo(), zorder=9)
    cj.marco(ax_a)
    cj.rotulo_panel(ax_a, "a")
    # Sumapaz y Bogotá urbana como referencia; municipios grandes del anillo
    sumapaz = utam.loc[utam["LOCNombre"].astype(str).str.upper().eq("SUMAPAZ")].dissolve()
    p = sumapaz.geometry.representative_point().iloc[0]
    candidatos_a = [dict(x=p.x, y=p.y - 6000, s="Sumapaz", ha="center", va="center", fontsize=cj.TAM_TEXTO,
                         color="#3F4A33", path_effects=cj.halo(), zorder=9)]
    reg = utam.loc[utam["zone_class"].eq("region")]
    for mun in ("ZIPAQUIRA", "LA CALERA", "FACATATIVA"):
        sub = reg.loc[reg["MUNNombre"].astype(str).str.upper().eq(mun)]
        if sub.empty:
            continue
        q = sub.dissolve().geometry.representative_point().iloc[0]
        ha = "center"
        if ext_b[0] < q.x < ext_b[2] + 8000 and ext_b[1] < q.y < ext_b[3]:
            # sobre la línea del recuadro (b) el rótulo la pisaría: se lleva al lado este, a la misma altura
            q, ha = Point(ext_b[2] + 1200, q.y), "left"
        candidatos_a.append(dict(x=q.x, y=q.y, s=cj.nombre(mun), ha=ha, va="center", fontsize=cj.TAM_MIN,
                                 fontstyle="italic", color=cj.MUNICIPIO, path_effects=cj.halo(), zorder=9))
    # escala 0-10-20 km y norte, abajo a la izquierda y arriba a la derecha de (a)
    # 0-10-20 km: el estilo común propone 0-25-50 km para la región, pero el panel (a) mide ~70 km
    # de ancho y una barra de 50 km lo cruzaría entero y pisaría Sumapaz (desviación registrada).
    cj.barra_escala(ax_a, ext_a[0] + 0.50 * (ext_a[2] - ext_a[0]), ext_a[1] + 0.025 * (ext_a[3] - ext_a[1]),
                    km_total=20, n_segmentos=4, marcas=[0, 10, 20])
    cj.flecha_norte(ax_a, 0.90, 0.92, alto_frac=0.05)

    # Localización en texto, no en mapa: la política de mapas de Elsevier (guía de JTG) no admite
    # localizadores que muestren un área mayor que la estudiada. Coordenadas del centroide urbano.
    centro_urbano = utam.loc[utam["zone_class"].eq("urban")].dissolve().geometry.centroid.to_crs(4326).iloc[0]
    texto_localizacion = (f"Bogotá D.C., capital of Colombia\n"
                          f"{centro_urbano.y:.1f}° N, {abs(centro_urbano.x):.1f}° W")

    # ---------- (b) núcleo urbano ----------
    _pintar_clases(ax_b, utam, lw_utam=0.3)
    locs = utam.loc[utam["zone_class"].eq("urban")].dissolve(by="LOCNombre")
    locs["geometry"] = locs.geometry.buffer(20).buffer(-20)
    locs.boundary.plot(ax=ax_b, color="#6E6253", linewidth=0.75, zorder=5)
    ax_b.set_xlim(ext_b[0], ext_b[2]); ax_b.set_ylim(ext_b[1], ext_b[3]); ax_b.set_aspect("equal")
    cj.marco(ax_b)
    cj.rotulo_panel(ax_b, "b")
    ax_b.scatter(estaciones.geometry.x, estaciones.geometry.y, s=13, color=cj.PUNTO, edgecolor="white",
                 linewidth=0.45, zorder=10)
    cand = utam.loc[utam["LOCNombre"].astype(str).str.upper().eq("CANDELARIA")].dissolve()
    centro = cand.geometry.representative_point().iloc[0]
    ax_b.scatter([centro.x], [centro.y], marker="*", s=80, color=cj.TEXTO, edgecolor="white",
                 linewidth=0.5, zorder=11)
    cj.barra_escala(ax_b, ext_b[0] + 0.05 * (ext_b[2] - ext_b[0]), ext_b[1] + 0.022 * (ext_b[3] - ext_b[1]),
                    km_total=5, n_segmentos=4, marcas=[0, 2.5, 5])
    cj.flecha_norte(ax_b)
    cj.credito(ax_b)

    # Leyenda (columna derecha, sin caja, con títulos)
    unidades = [
        Patch(facecolor=cj.URBANO, edgecolor="#c2b39a", lw=0.5, label="Urban Bogotá (111 UTAM)"),
        Patch(facecolor=cj.RURAL, edgecolor="#8f9a80", lw=0.5, label="Rural Bogotá (5 UPR units)"),
        Patch(facecolor="white", edgecolor=cj.VECINOS_RAYADO, hatch="///", lw=0.5,
              label="Neighbouring municipalities\n(26 units, 20 municipalities)"),
    ]
    referencia = [
        Line2D([], [], marker="o", ls="", markersize=4.2, markerfacecolor=cj.PUNTO, markeredgecolor="white",
               markeredgewidth=0.45, label=f"Public charging station\n(OSM, n = {len(estaciones)})"),
        Line2D([], [], marker="*", ls="", markersize=9, markerfacecolor=cj.TEXTO, markeredgecolor="white",
               markeredgewidth=0.5, label="Historic city centre"),
        Line2D([], [], color="#4D4D4D", lw=0.9, label="Bogotá D.C. boundary"),
        Line2D([], [], color="#6E6253", lw=0.75, label="Locality boundary"),
        Patch(facecolor=cj.SIN_DATO, edgecolor="#4D4D4D", lw=0.5, label="Outside the survey zones"),
        Line2D([], [], color=cj.TEXTO, lw=0.9, ls=(0, (3, 1.5)), label="Extent of panel (b)"),
    ]
    kw = dict(frameon=False, fontsize=cj.TAM_TEXTO, title_fontsize=cj.TAM_TEXTO, alignment="left",
              handlelength=1.8, handleheight=1.1, labelspacing=0.6, borderaxespad=0, borderpad=0)
    l1 = fig.legend(handles=unidades, title="Spatial units (UTAM, 2023)", loc="upper left",
                    bbox_to_anchor=(x_ley, base + alto), **kw)
    l1.get_title().set_fontweight("bold")
    fig.canvas.draw()
    y_l2 = fig.transFigure.inverted().transform(l1.get_window_extent())[0, 1] - 0.045
    l2 = fig.legend(handles=referencia, title="Reference", loc="upper left", bbox_to_anchor=(x_ley, y_l2), **kw)
    l2.get_title().set_fontweight("bold")
    fig.canvas.draw()
    y_l3 = fig.transFigure.inverted().transform(l2.get_window_extent())[0, 1] - 0.045
    fig.text(x_ley, y_l3, "Location", ha="left", va="top", fontsize=cj.TAM_TEXTO, fontweight="bold",
             color=cj.TEXTO)
    fig.text(x_ley, y_l3 - 0.035, texto_localizacion, ha="left", va="top", fontsize=cj.TAM_TEXTO,
             color=cj.TEXTO, linespacing=1.35)

    # Rótulos de (a) y de (b), con control de choques medido por el renderizador
    cj.rotulos_sin_choque(ax_a, candidatos_a)
    candidatos_b = [dict(x=centro.x + 450, y=centro.y - 350, s="City centre", ha="left", va="top",
                         fontsize=cj.TAM_MIN, color=cj.TEXTO, linespacing=0.95, path_effects=cj.halo(), zorder=11)]
    orden = locs.assign(area=locs.geometry.area).sort_values("area", ascending=False)
    for nombre_loc, fila in orden.iterrows():
        if str(nombre_loc).upper() == "CANDELARIA":
            continue
        q = fila.geometry.representative_point()
        texto = cj.nombre(nombre_loc)
        if " " in texto and len(texto) > 11:
            texto = texto.replace(" ", "\n", 1)
        candidatos_b.append(dict(x=q.x, y=q.y, s=texto, ha="center", va="center", fontsize=cj.TAM_MIN,
                                 color=cj.LOCALIDAD, linespacing=0.95, path_effects=cj.halo(), zorder=9))
    for mun in ("CHIA", "COTA", "FUNZA", "MOSQUERA", "SOACHA", "LA CALERA"):
        sub = reg.loc[reg["MUNNombre"].astype(str).str.upper().eq(mun)]
        if sub.empty:
            continue
        q = sub.dissolve().geometry.representative_point().iloc[0]
        candidatos_b.append(dict(x=q.x, y=q.y, s=cj.nombre(mun), ha="center", va="center", fontsize=cj.TAM_MIN,
                                 fontstyle="italic", color=cj.MUNICIPIO, path_effects=cj.halo(), zorder=9))
    rend = fig.canvas.get_renderer()
    reservado = [t.get_window_extent(rend) for t in ax_b.texts]  # escala, norte, crédito
    # Las estaciones y la estrella también se reservan: ningún rótulo las tapa
    for coleccion in ax_b.collections[-2:]:
        for xy in coleccion.get_offsets():
            px, py = ax_b.transData.transform(xy)
            reservado.append(Bbox.from_extents(px - 9, py - 9, px + 9, py + 9))
    puestos = cj.rotulos_sin_choque(ax_b, candidatos_b, ocupado_px=reservado)
    print("Rótulos en (b):", ", ".join(p.replace("\n", " ") for p in puestos))

    destino = salidas / "fig_study_area_context.png"
    fig.savefig(destino, dpi=400, facecolor="white")
    fig.savefig(salidas / "fig_study_area_context.pdf", facecolor="white")
    plt.close(fig)

    caption = (
        "Fig. 2. Study area. (a) Survey zones (UTAM, 2023): urban Bogotá (111), rural Bogotá (5 UPR units) "
        "and 26 units in 20 neighbouring municipalities (hatched; no census population for 22 of them). "
        "(b) Urban Bogotá with the 42 public charging stations mapped in "
        f"OpenStreetMap ({fecha_osm}), a provisional inventory. Data: SDM (2023); © OpenStreetMap contributors. "
        "Map lines delineate study areas and do not necessarily depict accepted national boundaries."
    )
    (salidas / "fig_study_area_context_caption.txt").write_text(caption + "\n", encoding="utf-8")
    return destino


def copiar_a_latex(origen: Path, nombre: str) -> None:
    """Copia la figura a latex/figures/, guardando la versión anterior como <nombre>_prev.png (una vez)."""
    import shutil
    destino = LATEX_FIGS / nombre
    previa = LATEX_FIGS / nombre.replace(".png", "_prev.png")
    if destino.exists() and not previa.exists():
        shutil.copy2(destino, previa)
    shutil.copy2(origen, destino)
    print(f"Copiada a {destino} (anterior en {previa.name})")


def main_solo_fig2() -> int:
    """Regenera SOLO la Fig. 2 y la copia a latex/figures/ (no toca las Figs. 4, 6 y 7)."""
    SALIDAS.mkdir(parents=True, exist_ok=True)
    utam = capas_norte_arriba()
    conteo = utam["zone_class"].value_counts().to_dict()
    if len(utam) != 142 or conteo != {"urban": 111, "region": 26, "rural": 5}:
        raise RuntimeError(f"Clasificación de tres clases inesperada: {conteo}")
    estaciones, fecha_osm = cargar_estaciones_osm()
    if len(estaciones) != 42:
        raise RuntimeError(f"Se esperaban 42 estaciones OSM; hay {len(estaciones)}.")
    destino = fig_study_area_context(SALIDAS, utam, estaciones, fecha_osm)
    copiar_a_latex(destino, "fig_study_area_context.png")
    return 0


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
    poblacion = pd.read_csv(RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    fig_allocation_composite(SALIDAS, utam_map, escenarios, rotation_deg, population=poblacion)
    fig_net_change_S2_minus_S1(SALIDAS, utam_map, escenarios, rotation_deg)

    # Fig. 2: norte arriba, sin la rotación de las demás (ver main_solo_fig2)
    estaciones, fecha_osm = cargar_estaciones_osm()
    fig_study_area_context(SALIDAS, capas_norte_arriba(), estaciones, fecha_osm)

    print(f"Figuras regeneradas en {SALIDAS}:")
    for ruta in sorted(SALIDAS.glob("fig_*.png")):
        print(f"  - {ruta.name}")
    print(f"Rotación aplicada al marco: {rotation_deg:.2f} grados (antihorario).")
    print(f"Unidades protegidas S3 (cuantil {1 - S3_PROTECTED_SHARE:.2f}): {int(mascara_protegida.sum())}")
    return 0


if __name__ == "__main__":
    # --solo-fig2: regenera solo la Fig. 2 y la copia a latex/figures/ (las Figs. 4, 6 y 7 son de otro frente)
    raise SystemExit(main_solo_fig2() if "--solo-fig2" in sys.argv else main())
