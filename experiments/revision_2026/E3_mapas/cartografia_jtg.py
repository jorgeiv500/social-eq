"""Elementos cartográficos comunes de las Figs. 2 y 10 (guía JTG, revisión R1 2026-10).

Lo usan `E3_mapas/regenerar_figuras.py` (Fig. 2, área de estudio) y
`C3_asignacion_por_sitio/mapa_c3.py` (Fig. 10, asignación por sitio). Norte arriba,
sin rotación: todas las capas van en EPSG:3116 (MAGNA-SIRGAS / Colombia Bogotá zone, metros),
el CRS de dibujo del estilo común (`estilo_cartografico_comun.md`, §1).

Parámetros tomados de `revision/r1_2026-08/plan_calidad/figuras_jtg/especificacion_figuras.md`
(§1.3 elementos cartográficos, §1.5 tipografía, §3 paleta) y del estilo común
`estilo_cartografico_comun.md` (flecha, escala, rótulos, rayado, sin dato, paletas reservadas).
No se importan las funciones de `followup.py` para no depender de su edición en curso; los
valores son los mismos.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle

CRS_MAPA = 3116
# Marco urbano común (URBAN_FRAME_3116 en followup.py): Figs. 2(b), 4, 6, 7, 10 y C1
MARCO_URBANO = (980500.0, 981500.0, 1010500.0, 1029000.0)
ESCALA = "#4F4A44"           # flecha de norte y barra de escala
LOCALIDAD = "#4A443C"
MUNICIPIO = "#6C655C"
FUENTE = "Arial"
TEXTO = "#1B1B1B"
GRIS_MARCO = "#7F7F7F"
TAM_PANEL = 9.0      # rótulo "(a)" en negrita
TAM_TEXTO = 7.5      # leyenda, escala, rótulos de lugar
TAM_MIN = 7.0        # nada por debajo de 7 pt a tamaño impreso

# Paleta común (guía §3)
PUNTO = "#1B1B1B"            # puntos sobre coropleta, borde blanco
VECINOS_RELLENO = "#FFFFFF"  # municipios vecinos: rayado gris sobre blanco
VECINOS_RAYADO = "#8D8478"   # rayado /// del estilo común
VECINOS_BORDE = "#7F7F7F"
URBANO = "#F2E6D0"
RURAL = "#DCE6CF"
SIN_DATO = "#D9D9D9"
# El naranja #D55E00 queda reservado al escenario S2 (estilo común §3): aquí no se usa.

# Nombres con tilde (el shapefile los trae en mayúsculas y sin tilde; 'ANTONIO NARINO' sin Ñ)
NOMBRES = {
    "USAQUEN": "Usaquén", "ENGATIVA": "Engativá", "FONTIBON": "Fontibón", "CIUDAD BOLIVAR": "Ciudad Bolívar",
    "SAN CRISTOBAL": "San Cristóbal", "ANTONIO NARINO": "Antonio Nariño", "LOS MARTIRES": "Los Mártires",
    "CANDELARIA": "La Candelaria", "CHIA": "Chía", "ZIPAQUIRA": "Zipaquirá", "FACATATIVA": "Facatativá",
    "CAJICA": "Cajicá", "SOPO": "Sopó", "SIBATE": "Sibaté", "BOJACA": "Bojacá", "GACHANCIPA": "Gachancipá",
    "TOCANCIPA": "Tocancipá", "CHOACHÍ": "Choachí", "BOGOTA": "Bogotá",
}


def nombre(raw: str) -> str:
    clave = str(raw).upper().strip()
    return NOMBRES.get(clave, clave.title().replace(" De ", " de "))


def estilo_base() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FUENTE, "Helvetica", "DejaVu Sans"],
        "font.size": TAM_TEXTO,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
    })


def halo(ancho: float = 2.0) -> list:
    return [patheffects.withStroke(linewidth=ancho, foreground="white", alpha=0.9)]


def marco(ax, color: str = GRIS_MARCO, ancho: float = 0.5) -> None:
    """Neat line fina alrededor del mapa, sin ticks."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(ancho)


def rotulo_panel(ax, letra: str, dx_pt: float = 0.0) -> None:
    """"(a)" en negrita 9 pt, arriba a la izquierda y fuera del área de datos."""
    ax.annotate(f"({letra})", xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=TAM_PANEL, fontweight="bold", color=TEXTO)


def flecha_norte(ax, x: float = 0.92, y: float = 0.90, alto_frac: float = 0.05) -> None:
    """Flecha de norte vertical (norte arriba): triángulo relleno #4F4A44 y "N" negrita 7 pt encima.

    (x, y) en fracción de ejes es la PUNTA; por defecto 0,92/0,90 como en el estilo común.
    """
    ax.annotate("", xy=(x, y), xytext=(x, y - alto_frac), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28", color=ESCALA, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=12)
    ax.text(x, y + 0.004, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=TAM_MIN, fontweight="bold", color=ESCALA, path_effects=halo(1.8), zorder=12)


def barra_escala(ax, x0: float, y0: float, km_total: float, n_segmentos: int = 4,
                 marcas: list[float] | None = None) -> None:
    """Barra alternada blanco/#4F4A44 anclada en (x0, y0) en metros; rótulos "0", "2.5", "5 km"."""
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    alto = 0.009 * (ymax - ymin)
    paso = km_total * 1000 / n_segmentos
    for i in range(n_segmentos):
        ax.add_patch(Rectangle((x0 + i * paso, y0), paso, alto, facecolor=ESCALA if i % 2 == 0 else "white",
                               edgecolor=ESCALA, lw=0.5, zorder=12))
    marcas = marcas if marcas is not None else [0, km_total / 2, km_total]
    for v in marcas:
        num = f"{v:g}"
        ultimo = v == marcas[-1]
        # "5 km": el número queda centrado en su marca; " km" sigue a la derecha (cifra Arial = 0,556 em)
        dx = -0.5 * len(num) * 0.556 * TAM_MIN if ultimo else 0.0
        ax.annotate(num + (" km" if ultimo else ""), xy=(x0 + v * 1000, y0 + alto), xytext=(dx, 1.5),
                    textcoords="offset points", ha="left" if ultimo else "center", va="bottom", fontsize=TAM_MIN,
                    color=ESCALA, path_effects=halo(1.6), zorder=12)


def credito(ax, texto: str = "© OpenStreetMap contributors") -> None:
    ax.text(0.985, 0.012, texto, transform=ax.transAxes, ha="right", va="bottom", fontsize=TAM_MIN,
            color="#4D4D4D", path_effects=halo(1.6), zorder=12)


def rotulos_sin_choque(ax, candidatos: list[dict], ocupado_px: list | None = None,
                       desplazamientos: tuple = ((0, 0), (0, 7), (0, -7), (8, 0), (-8, 0), (6, 6), (-6, -6),
                                                 (6, -6), (-6, 6))) -> list:
    """Coloca rótulos en orden de prioridad; si chocan, prueba desplazamientos cortos (en pt) y,
    si ninguno cabe, descarta el rótulo.

    candidatos: dicts con x, y (dato), s (texto) y kwargs de `ax.annotate`. Las cajas se miden con
    el renderizador, así que la figura debe tener ya su tamaño y límites definitivos.
    `ocupado_px`: lista de Bbox en píxeles ya ocupadas (flecha, escala, puntos, leyenda).
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cajas = list(ocupado_px or [])
    axbb = ax.get_window_extent(rend)
    puestos = []
    for c in candidatos:
        c = dict(c)
        x, y, s = c.pop("x"), c.pop("y"), c.pop("s")
        for dx, dy in desplazamientos:
            t = ax.annotate(s, xy=(x, y), xytext=(dx, dy), textcoords="offset points", **c)
            bb = t.get_window_extent(rend).expanded(1.04, 1.08)
            dentro = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
            if dentro and not any(bb.overlaps(o) for o in cajas):
                cajas.append(bb); puestos.append(s)
                break
            t.remove()
    return puestos
