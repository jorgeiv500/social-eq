#!/usr/bin/env python3
"""Figura 1 de JTRG-D-26-00692: diagrama de flujo metodológico.

Estilo: tres columnas con
encabezado en cursiva gris, cajas de esquinas redondeadas con título en negrita y contenido
concreto, bloque del modelo en un único azul suave, resto en grises; flechas sólidas para el
flujo principal y discontinuas para la sensibilidad. Criterio JTG: letra de 7 pt al ancho de
doble columna (se dibuja a 7 in), sin partición de palabras, PDF vectorial.

Todos los datos de las cajas salen de revision/r1_2026-08/latex/main.tex (tabla de datos,
§ parametrización, robustez y nivel de sitio). Los textos de la figura van en inglés.

Uso: .venv/bin/python scripts/build_workflow_figure.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision" / "r1_2026-08" / "latex" / "figures"
STEM = "fig_workflow"

MM = 1 / 25.4  # pulgadas por milímetro
W = 178.0  # ancho de dibujo en mm (≈ 7 in, doble columna)

INK = "#1a1a1a"
MUTED = "#5b5b5b"
GREY = "#f0f0f0"
WHITE = "#ffffff"
ACCENT, ACCENT_EDGE = "#dce6f1", "#2b5d8a"
EDGE = "#7a7a7a"
ARROW = "#3d3d3d"
DASH = (0, (3, 2))

FS = 7.0  # pt; tamaño único de letra
LINE = 3.15  # mm por línea a 7 pt con interlineado 1,25
PAD_TOP = 2.0
TITLE_GAP = 3.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS,
    "mathtext.fontset": "stixsans",  # ν distinguible de v (en Arial cursiva se confunden)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def main() -> None:
    H = 118.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    def box(x, top, w, text, fc=GREY, ec=EDGE, ls="-", h=None, lw=0.6):
        """Caja con título en negrita (primera línea) y cuerpo; devuelve (x, y, w, h)."""
        lines = text.split("\n")
        h = h or (PAD_TOP + TITLE_GAP + LINE * (len(lines) - 1) + 1.6)
        y = top - h
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        ax.text(x + 2, top - PAD_TOP, lines[0], ha="left", va="top", fontsize=FS,
                fontweight="bold", color=INK, zorder=3)
        if len(lines) > 1:
            ax.text(x + 2, top - PAD_TOP - TITLE_GAP, "\n".join(lines[1:]), ha="left", va="top",
                    fontsize=FS, linespacing=1.25, color=INK, zorder=3)
        return (x, y, w, h)

    def arrow(p, q, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=ARROW, ls=ls, zorder=1, shrinkA=0, shrinkB=0))

    def seg(p, q, ls="-"):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=ARROW, lw=0.7, ls=ls, zorder=1,
                solid_capstyle="butt")

    def mid_y(b, f=0.5):
        return b[1] + b[3] * f

    # Columnas: x y ancho (mm)
    X1, W1 = 1.0, 52.0
    X2, W2 = 63.0, 52.0
    X3, W3 = 125.0, 52.0
    TOP = H - 7.0

    for x, lab in ((X1, "Inputs (public data)"), (X2, "Learning and allocation"),
                   (X3, "Outputs and checks")):
        ax.text(x, H - 1.5, lab, ha="left", va="top", fontsize=FS, color=MUTED,
                fontstyle="italic")

    # --- Columna 1: insumos -------------------------------------------------------
    G = 2.6  # separación vertical entre cajas
    survey = box(X1, TOP, W1,
                 "Travel survey\n"
                 "2023 household mobility survey\n"
                 "(142 zones); expanded trips by\n"
                 "origin, destination, hour, purpose")
    census = box(X1, survey[1] - G, W1,
                 "Census vulnerability proxies\n"
                 "DANE: persons per dwelling and\n"
                 "population density → index $\\nu_i$")
    quotas = box(X1, census[1] - G, W1,
                 "EV fleet and venue quotas\n"
                 "Registered EVs, Nov 2020–Nov 2024;\n"
                 "EVI-Pro Lite: 10,870 ports\n"
                 "(residential, workplace, public)")
    network = box(X1, quotas[1] - G, W1,
                  "Network times\n"
                  "OD travel-time matrix between\n"
                  "zones → accessibility impedance")
    layers = box(X1, network[1] - G, W1,
                 "Real-world layers\n"
                 "Substation headroom; Decree 555\n"
                 "zoning (2021); car parks from\n"
                 "OpenStreetMap and Cadastre")
    sens = box(X1, layers[1] - 4.5, W1,
               "Sensitivity\n"
               "Decay $\\lambda_v$ ±25%; uniform venue\n"
               "weights; $\\kappa$ from 2 to 6; PCA index;\n"
               "soft-floor S3 ($\\gamma$, $\\beta$) as a diagnostic",
               fc=WHITE, ls=DASH)

    # --- Columna 2: aprendizaje y asignación --------------------------------------
    demand = box(X2, TOP, W2,
                 "Demand estimation\n"
                 "Interchangeable module: ST-GNN\n"
                 "(reference) or random forest with\n"
                 "spatial lags; leakage-free test on\n"
                 "a household split")
    alloc_top = demand[1] - 6.0
    core = box(X2, alloc_top, W2,
               "Lexicographic allocation\n"
               "Targets $T_{iv}=\\pi_{iv}Q_v$ under fixed quotas\n"
               "S0 demand fidelity: optimum $U^{\\star}$\n"
               "S1 territorial equalization, $\\varepsilon$ = 0.03\n"
               "S2 socially weighted equalization,\n"
               "     $\\varepsilon$ = 0.08, $\\kappa$ = 4\n"
               "Fidelity budget $(1+\\varepsilon)\\,U^{\\star}$\n"
               "Caps: grid headroom, zoning eligibility\n"
               "\n"
               "Output: integer ports $n_{iv}$\n"
               "by zone and venue",
               fc=ACCENT, ec=ACCENT_EDGE, lw=0.8)
    site = box(X2, core[1] - 6.0, W2,
               "Site level (indicative)\n"
               "Workplace and public ports of each\n"
               "zone to its car parks, up to 10%\n"
               "of bays; residential ports not sited")

    # --- Columna 3: resultados y comprobaciones -----------------------------------
    equity = box(X3, alloc_top, W3,
                 "Equity diagnostics\n"
                 "Weighted Gini; ports per 10,000\n"
                 "residents; coverage of vulnerable\n"
                 "zones within 15 min")
    robust = box(X3, equity[1] - 5.0, W3,
                 "Robustness and benchmarks\n"
                 "Tolerance frontier ($\\varepsilon$ up to 100);\n"
                 "direct-equity benchmark;\n"
                 "demand-model counterfactual;\n"
                 "growth to 2028 and 2030")
    plan = box(X3, site[1] + site[3], W3,
               "Planning reading\n"
               "Ports to lift the vulnerable\n"
               "quartile to the mean: about 350\n"
               "(observed chain) to 942 (baseline\n"
               "chain); installable share by quartile")

    # --- Flechas -------------------------------------------------------------------
    xr1 = X1 + W1
    # Encuesta -> estimación de demanda
    arrow((xr1, mid_y(survey)), (X2, mid_y(survey)))
    # Censo, cuotas, red -> asignación (horizontales, recortadas al rango de la caja)
    for b in (census, quotas, network):
        yy = min(max(mid_y(b), core[1] + 10.0), core[1] + core[3] - 4)
        arrow((xr1, mid_y(b)), (X2, yy))
    # Capas reales -> asignación (topes) y -> nivel de sitio (parqueaderos), por un tronco común
    xb = xr1 + 4.5
    yl = mid_y(layers)
    y_caps = core[1] + 5.0
    seg((xr1, yl), (xb, yl))
    seg((xb, mid_y(site)), (xb, y_caps))
    arrow((xb, y_caps), (X2, y_caps))
    arrow((xb, mid_y(site)), (X2, mid_y(site)))
    # Sensibilidad -> asignación (discontinua, por el canal entre columnas)
    xs = X2 - 2.2
    ys = mid_y(sens)
    seg((xr1, ys), (xs, ys), ls=DASH)
    seg((xs, ys), (xs, core[1] + 1.8), ls=DASH)
    arrow((xs, core[1] + 1.8), (X2, core[1] + 1.8), ls=DASH)
    # Flujo vertical de la columna central
    xc = X2 + W2 / 2
    arrow((xc, demand[1]), (xc, core[1] + core[3]))
    arrow((xc, core[1]), (xc, site[1] + site[3]))
    # Asignación -> diagnósticos y robustez; sitio -> lectura de planificación
    xr2 = X2 + W2
    arrow((xr2, mid_y(equity)), (X3, mid_y(equity)))
    arrow((xr2, mid_y(robust)), (X3, mid_y(robust)))
    arrow((xr2, mid_y(site)), (X3, mid_y(site)))

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{STEM}.png"
    prev = OUT / f"{STEM}_prev.png"
    if png.exists() and not prev.exists():
        shutil.copy2(png, prev)  # conserva la versión TikZ anterior
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"escrita {OUT / STEM}.pdf y .png")


if __name__ == "__main__":
    main()
