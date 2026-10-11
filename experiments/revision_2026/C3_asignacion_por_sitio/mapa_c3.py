"""C3 — Fig. 10 del manuscrito: sitios con puertos (nivel 2, principal) y brecha por UTAM.

Escribe `salidas/mapa_sitios_brecha.png` (180 mm de ancho a 400 ppp → 2835 px) y su PDF, y copia el
PNG a `revision/r1_2026-08/latex/figures/fig_site_allocation.png` (la versión anterior queda como
`fig_site_allocation_prev.png`). Estilo común de mapas (`plan_calidad/figuras_jtg/estilo_cartografico_comun.md`):
norte arriba, EPSG:3116, marco urbano común, flecha y escala #4F4A44, rótulos "(a)", "(b)", Arial,
localidades con tilde y halo, centro histórico. Textos de la figura en inglés.
No cambia ningún dato: lee las salidas ya calculadas por `asignar_c3.py`.

Uso: python mapa_c3.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(AQUI.parent / "E3_mapas"))
from social_eq.followup import classify_zone_class, normalize_zone_code  # noqa: E402
import cartografia_jtg as cj  # noqa: E402

SAL = AQUI / "salidas"
SHP_UTAM = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp"
LATEX_FIG = RAIZ / "revision" / "r1_2026-08" / "latex" / "figures" / "fig_site_allocation.png"
CRS_DATOS = 9377  # las coordenadas x, y de asignacion_por_sitio_principal.csv (asignar_c3.py)

# Brecha de cupo, 5 clases (guía §3, ColorBrewer BrBG mitad marrón). La clase 0 % es un gris cálido
# con borde, distinto del exterior (blanco).
CLASES = [-0.5, 0.5, 25, 50, 75, 100.5]
COLORES = ["#E9E5DD", "#F6E8C3", "#DFC27D", "#BF812D", "#8C510A"]
ETIQUETAS = ["0 (quota fully sited)", "1–25", "25–50", "50–75", "75–100"]
# Puertos por sitio: 887 sitios con 1, 200 con 2, 428 con 3, 3 con 4 y 1 con 6 (no hay sitios de ≥ 8).
TAMANOS = {1: 2.5, 2: 7.0, 3: 14.0, 4: 26.0}  # área en pt²; 4 = "4–6 ports"
BARRA_RESTO = "#BDBDBD"
BARRA_Q4 = cj.TEXTO  # #1B1B1B: el mismo negro del contorno del cuartil vulnerable en (a)


def tamano(p: pd.Series) -> np.ndarray:
    return np.array([TAMANOS[min(int(v), 4)] for v in p])


def main() -> int:
    cj.estilo_base()
    ut_all = gpd.read_file(SHP_UTAM).to_crs(cj.CRS_MAPA)
    ut_all["UTAM"] = ut_all["UTAM"].map(normalize_zone_code)
    ut_all["zone_class"] = ut_all["UTAM"].map(classify_zone_class)
    vecinos = ut_all[ut_all["MUNCodigo"] != 11001]
    ut = ut_all[ut_all["MUNCodigo"] == 11001].dissolve(by="UTAM", as_index=False)
    b = pd.read_csv(SAL / "brecha_por_utam.csv")
    b = b[(b["escenario"] == "principal") & (b["conjunto_fuentes"] == "ambas")]
    ut = ut.merge(b[["UTAM", "cuota_sitio", "brecha", "cuartil"]], on="UTAM", how="left")
    ut["pct_brecha"] = np.where(ut["cuota_sitio"] > 0,
                                100 * ut["brecha"] / ut["cuota_sitio"].where(ut["cuota_sitio"] > 0, 1), np.nan)
    s = pd.read_csv(SAL / "asignacion_por_sitio_principal.csv")
    s = s[s["puertos"] > 0]
    s = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["x"], s["y"]), crs=CRS_DATOS).to_crs(cj.CRS_MAPA)

    ext = cj.MARCO_URBANO
    ancho_in, alto_in = 180 / 25.4, 6.35
    fig = plt.figure(figsize=(ancho_in, alto_in))
    alto_ax, base = 0.955, 0.008
    ancho_ax = alto_ax * alto_in * (ext[2] - ext[0]) / (ext[3] - ext[1]) / ancho_in
    ax = fig.add_axes([0.006, base, ancho_ax, alto_ax])
    x_der = 0.006 + ancho_ax + 0.035

    # ---------- (a) mapa ----------
    cmap = ListedColormap(COLORES); norm = BoundaryNorm(CLASES, cmap.N)
    vecinos.plot(ax=ax, facecolor="white", edgecolor=cj.VECINOS_BORDE, linewidth=0.3, zorder=1)
    vecinos.plot(ax=ax, facecolor="none", edgecolor=cj.VECINOS_RAYADO, hatch="///", linewidth=0, zorder=1)
    ut.plot(ax=ax, column="pct_brecha", cmap=cmap, norm=norm, edgecolor="#A39B8F", linewidth=0.3, zorder=2,
            missing_kwds={"color": "white", "edgecolor": "#B9B0A3", "linewidth": 0.5})  # cuota 0 (UTAM2, UTAM3, UPR5)
    # Hueco interior del marco UTAM (3,15 km² al sur): fuera de las zonas de la encuesta, gris como en la Fig. 2
    bogota = ut.unary_union.buffer(20).buffer(-20)
    huecos = [Polygon(r) for r in getattr(bogota, "interiors", []) if Polygon(r).area > 5e4]
    if huecos:
        gpd.GeoSeries(huecos, crs=ut.crs).plot(ax=ax, facecolor=cj.SIN_DATO, edgecolor="#A39B8F", linewidth=0.3, zorder=2)
    locs = ut_all[ut_all["zone_class"].eq("urban")].dissolve(by="LOCNombre")
    locs["geometry"] = locs.geometry.buffer(20).buffer(-20)
    locs.boundary.plot(ax=ax, color="#6E6253", linewidth=0.6, zorder=3)
    ut[ut["cuartil"] == 4].boundary.plot(ax=ax, color=cj.TEXTO, linewidth=1.0, zorder=4)
    ax.scatter(s.geometry.x, s.geometry.y, s=tamano(s["puertos"]), c=cj.PUNTO, edgecolors="white",
               linewidths=0.3, alpha=0.85, zorder=5)
    cand = ut_all[ut_all["LOCNombre"].astype(str).str.upper().eq("CANDELARIA")].dissolve()
    centro = cand.geometry.representative_point().iloc[0]
    ax.scatter([centro.x], [centro.y], marker="*", s=90, color=cj.TEXTO, edgecolor="white", linewidth=0.6, zorder=7)
    ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3]); ax.set_aspect("equal")
    cj.marco(ax)
    cj.rotulo_panel(ax, "a")
    cj.flecha_norte(ax)
    cj.barra_escala(ax, ext[0] + 0.05 * (ext[2] - ext[0]), ext[1] + 0.022 * (ext[3] - ext[1]),
                    km_total=5, n_segmentos=4, marcas=[0, 2.5, 5])
    cj.credito(ax)

    # ---------- leyenda (sin caja, títulos en negrita, como la Fig. 2) ----------
    kw = dict(frameon=False, fontsize=cj.TAM_TEXTO, title_fontsize=cj.TAM_TEXTO, alignment="left",
              handlelength=1.8, handleheight=1.1, labelspacing=0.5, borderaxespad=0, borderpad=0)
    h1 = [Patch(facecolor=c, edgecolor="#A39B8F", lw=0.4, label=l) for c, l in zip(COLORES, ETIQUETAS)]
    l1 = fig.legend(handles=h1, title="Workplace + public quota\nwithout an eligible site (%)", loc="upper left",
                    bbox_to_anchor=(x_der, base + alto_ax), **kw)
    l1.get_title().set_fontweight("bold")
    h2 = [Line2D([], [], marker="o", ls="", markerfacecolor=cj.PUNTO, markeredgecolor="white", markeredgewidth=0.3,
                 markersize=np.sqrt(TAMANOS[k]), label=lab)
          for k, lab in ((1, "1"), (2, "2"), (3, "3"), (4, "4–6"))]
    h3 = [Line2D([], [], color=cj.TEXTO, lw=1.0, label="Most vulnerable quartile"),
          Line2D([], [], color="#6E6253", lw=0.6, label="Locality boundary"),
          Line2D([], [], marker="*", ls="", markersize=9, markerfacecolor=cj.TEXTO, markeredgecolor="white",
                 markeredgewidth=0.5, label="Historic city centre"),
          Patch(facecolor="white", edgecolor="#B9B0A3", lw=0.5, label="No workplace or public quota"),
          Patch(facecolor=cj.SIN_DATO, edgecolor="#A39B8F", lw=0.4, label="Outside the survey zones"),
          Patch(facecolor="white", edgecolor=cj.VECINOS_RAYADO, hatch="///", lw=0.5,
                label="Neighbouring municipalities")]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    y1 = inv.transform(l1.get_window_extent())[0, 1] - 0.035
    l2 = fig.legend(handles=h2, title="Ports per car park", loc="upper left", bbox_to_anchor=(x_der, y1),
                    ncol=4, columnspacing=0.9, handletextpad=0.3, **{**kw, "handlelength": 1.0})
    l2.get_title().set_fontweight("bold")
    fig.canvas.draw()
    y2 = inv.transform(l2.get_window_extent())[0, 1] - 0.035
    l3 = fig.legend(handles=h3, loc="upper left", bbox_to_anchor=(x_der, y2), **kw)
    fig.canvas.draw()
    y3 = inv.transform(l3.get_window_extent())[0, 1]

    # rótulos de localidad y centro (7 pt, halo), sin pisar flecha, escala, crédito ni la estrella
    rend = fig.canvas.get_renderer()
    reservado = [t.get_window_extent(rend) for t in ax.texts]
    px, py = ax.transData.transform((centro.x, centro.y))
    from matplotlib.transforms import Bbox
    reservado.append(Bbox.from_extents(px - 8, py - 8, px + 8, py + 8))
    candidatos = [dict(x=centro.x, y=centro.y, s="City centre", ha="left", va="top", fontsize=cj.TAM_MIN,
                       color=cj.TEXTO, path_effects=cj.halo(), zorder=8, xytext_extra=(5, -4))]
    orden = locs.assign(area=locs.geometry.area).sort_values("area", ascending=False)
    for nombre_loc, fila in orden.iterrows():
        if str(nombre_loc).upper() == "CANDELARIA":
            continue
        q = fila.geometry.representative_point()
        texto = cj.nombre(nombre_loc)
        if " " in texto and len(texto) > 11:
            texto = texto.replace(" ", "\n", 1)
        candidatos.append(dict(x=q.x, y=q.y, s=texto, ha="center", va="center", fontsize=cj.TAM_MIN,
                               color=cj.LOCALIDAD, linespacing=0.95, path_effects=cj.halo(), zorder=8))
    # el desplazamiento propio del rótulo del centro se aplica moviendo su ancla unos metros
    c0 = candidatos[0]; c0.pop("xytext_extra")
    c0["x"] += 350; c0["y"] -= 250
    puestos = cj.rotulos_sin_choque(ax, candidatos, ocupado_px=reservado)
    print("Rótulos en (a):", ", ".join(p.replace("\n", " ") for p in puestos))

    # ---------- (b) fracción instalable por cuartil ----------
    r = pd.read_csv(SAL / "resultados_escenarios.csv")
    r = r[(r["fuente"] == "ambas") & (r["estimador"] == "RF_rezago")]
    base_r = r[r["escenario"] == "principal"].iloc[0]
    lo = r[r["escenario"] == "capacidad_5"].iloc[0]
    hi = r[r["escenario"] == "capacidad_20"].iloc[0]
    grupos = [("Q1 (least\nvulnerable)", "frac_Q1_bogota"), ("Q2", "frac_Q2_bogota"), ("Q3", "frac_Q3_bogota"),
              ("Q4 (most\nvulnerable)", "frac_Q4_bogota")]
    bx_x = x_der + 0.085
    bx_top = y3 - 0.09
    bx = fig.add_axes([bx_x, 0.085, 0.985 - bx_x, bx_top - 0.085])
    yy = np.arange(len(grupos))[::-1]
    for y_, (lab, col) in zip(yy, grupos):
        c = BARRA_Q4 if lab.startswith("Q4") else BARRA_RESTO
        bx.barh(y_, 100 * base_r[col], color=c, height=0.62)
        # bigote negro con halo blanco: se ve igual sobre la barra gris, la negra y el fondo
        for kwb in (dict(lw=0.8), dict(ls="", marker="|", ms=5)):
            bx.plot([100 * lo[col], 100 * hi[col]], [y_, y_], color=cj.TEXTO, path_effects=cj.halo(2.2), **kwb)
        # etiqueta en negro, fuera de la barra y de los bigotes
        bx.text(max(100 * base_r[col], 100 * hi[col]) + 2, y_, f"{100 * base_r[col]:.0f}%", ha="left", va="center",
                fontsize=cj.TAM_MIN, color=cj.TEXTO)
    bx.set_yticks(yy); bx.set_yticklabels([g[0] for g in grupos], fontsize=cj.TAM_MIN, linespacing=0.95)
    bx.set_xlim(0, 113); bx.set_xticks([0, 25, 50, 75, 100])
    bx.spines["bottom"].set_bounds(0, 100)
    bx.tick_params(axis="both", labelsize=cj.TAM_MIN, length=2.5, width=0.5)
    bx.set_xlabel("Quota that fits in car parks (%)", fontsize=cj.TAM_MIN)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        bx.spines[sp].set_linewidth(0.5)
    fig.text(x_der, bx_top + 0.03, "(b)", fontsize=cj.TAM_PANEL, fontweight="bold", color=cj.TEXTO, va="bottom")

    fig.savefig(SAL / "mapa_sitios_brecha.png", dpi=400, facecolor="white")
    fig.savefig(SAL / "mapa_sitios_brecha.pdf", facecolor="white")
    plt.close(fig)
    from PIL import Image
    im = Image.open(SAL / "mapa_sitios_brecha.png"); print("px:", im.size, "dpi:", im.info.get("dpi"))

    previa = LATEX_FIG.with_name("fig_site_allocation_prev.png")
    if LATEX_FIG.exists() and not previa.exists():
        shutil.copy2(LATEX_FIG, previa)
    shutil.copy2(SAL / "mapa_sitios_brecha.png", LATEX_FIG)
    print(f"Copiada a {LATEX_FIG} (anterior en {previa.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
