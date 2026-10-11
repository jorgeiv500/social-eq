"""C3 — Mapa a ancho de impresión: sitios con puertos (nivel 2, principal) y brecha por UTAM.

Escribe `salidas/mapa_sitios_brecha.png` (7,5 × 8,4 in a 320 dpi → 2400 px de ancho) y su PDF.
Textos del mapa en inglés (van al suplemento del artículo). Uso: python mapa_c3.py
"""
from __future__ import annotations

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

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from social_eq.followup import normalize_zone_code  # noqa: E402

SAL = AQUI / "salidas"
SHP_UTAM = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp"
NOTA = "Map lines delineate study areas and do not necessarily depict accepted national boundaries."


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    ut = gpd.read_file(SHP_UTAM).to_crs(9377)
    ut["UTAM"] = ut["UTAM"].map(normalize_zone_code)
    ut = ut[ut["MUNCodigo"] == 11001].dissolve(by="UTAM", as_index=False)
    b = pd.read_csv(SAL / "brecha_por_utam.csv")
    b = b[(b["escenario"] == "principal") & (b["conjunto_fuentes"] == "ambas")]
    ut = ut.merge(b[["UTAM", "cuota_sitio", "brecha", "cuartil"]], on="UTAM", how="left")
    ut["pct_brecha"] = np.where(ut["cuota_sitio"] > 0, 100 * ut["brecha"] / ut["cuota_sitio"].where(ut["cuota_sitio"] > 0, 1), np.nan)
    s = pd.read_csv(SAL / "asignacion_por_sitio_principal.csv")
    s = s[s["puertos"] > 0]

    # extensión: área urbana (UTAM no rurales) con un margen
    urb = ut[ut["UTAM"].str.startswith("UTAM")]
    x0, y0, x1, y1 = urb.total_bounds
    pad = 1500
    fig = plt.figure(figsize=(7.5, 7.9))
    ax = fig.add_axes([0.01, 0.085, 0.60, 0.905])
    clases = [-0.5, 0.5, 25, 50, 75, 100.5]
    colores = ["#f3f1ec", "#e3d3bf", "#cfae8c", "#b07f5e", "#7f4f3a"]
    cmap = ListedColormap(colores); norm = BoundaryNorm(clases, cmap.N)
    ut.plot(ax=ax, column="pct_brecha", cmap=cmap, norm=norm, edgecolor="#8a8a8a", linewidth=0.35)
    ut[ut["cuartil"] == 4].boundary.plot(ax=ax, color="#2b2b2b", linewidth=0.9)
    tam = np.clip(s["puertos"], 1, 12)
    ax.scatter(s["x"], s["y"], s=2.2 + 1.6 * tam, c="#1f4e5f", edgecolors="white", linewidths=0.15, alpha=0.85, zorder=3)
    ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#6b6b6b"); sp.set_linewidth(0.6)

    # norte
    xa, ya = x1 + pad - 2200, y1 + pad - 4200
    ax.annotate("", xy=(xa, ya + 2600), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", color="#2b2b2b", lw=1.2, mutation_scale=12))
    ax.text(xa, ya + 2900, "N", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2b2b2b")
    # escala (5 km)
    xs, ys = x0 - pad + 1500, y1 + pad - 2600
    for i, c in enumerate(["#2b2b2b", "white", "#2b2b2b", "white", "#2b2b2b"]):
        ax.add_patch(plt.Rectangle((xs + i * 1000, ys), 1000, 350, facecolor=c, edgecolor="#2b2b2b", lw=0.6, zorder=4))
    for v in (0, 2.5, 5):
        ax.text(xs + v * 1000, ys + 550, f"{v:g}", ha="center", va="bottom", fontsize=7, zorder=5)
    ax.text(xs + 5000 + 300, ys + 175, "km", ha="left", va="center", fontsize=7, zorder=5)

    h = [Patch(facecolor=c, edgecolor="#8a8a8a", lw=0.4, label=l) for c, l in
         zip(colores, ["0 % (quota fully sited)", "1–25 %", "25–50 %", "50–75 %", "75–100 %"])]
    h += [Line2D([], [], color="#2b2b2b", lw=1.0, label="Most vulnerable quartile (UTAM)")]
    for p_, l in ((1, "1 port"), (3, "3 ports"), (8, "≥ 8 ports")):
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="#1f4e5f", markeredgecolor="white", markeredgewidth=0.2,
                        markersize=np.sqrt(2.2 + 1.6 * p_), label=f"Site with charger: {l}"))
    leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.625, 0.985), fontsize=7, frameon=True, framealpha=0.92, edgecolor="#9a9a9a",
                    title="Share of workplace + public quota\nwithout an eligible site (gap)", title_fontsize=7.5)
    leg._legend_box.align = "left"
    tot = int(b["cuota_sitio"].sum()); br = int(b["brecha"].sum())
    fig.text(0.02, 0.012,
             f"Level 2 (site) allocation of the published S2 workplace and public ports within each UTAM; off-street car parks "
             f"(OSM + Cadastre), 10 % of\nspaces electrified, demand = car-driver arrivals (RF with spatial lag). Bogotá urban area shown; "
             f"gap {br:,} of {tot:,} ports in all\n142 UTAMs, incl. rural and neighbouring-municipality UTAMs outside the frame. EPSG:9377.\n{NOTA}",
             fontsize=6.3, color="#3a3a3a", va="bottom")
    fig.savefig(SAL / "mapa_sitios_brecha.png", dpi=320)
    fig.savefig(SAL / "mapa_sitios_brecha.pdf")
    from PIL import Image
    im = Image.open(SAL / "mapa_sitios_brecha.png"); print("px:", im.size, "dpi:", im.info.get("dpi"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
