"""E7 — Regenera TODAS las figuras del manuscrito al tamaño al que se imprimen.

Contexto (R2.m2, reabierto el 2026-09-02). El Revisor 2 escribió "Maps such as
Figures 3 and 5 are too small and difficult to read". La pasada del Frente E
cambió las proporciones pero no el problema de fondo: las figuras se dibujan con
un `figsize` de 8.6–15.3 pulgadas de ancho y se colocan en el `.tex` a 5.1–7.1
pulgadas, así que su texto se imprime al 46–79 % del tamaño con que se rotuló y
la fuente más pequeña de cada figura queda entre 3.2 y 5.2 pt.

La corrección es dibujar cada figura al ancho al que se imprime: el `fontsize` de
matplotlib está en puntos y el lienzo en pulgadas, así que reducir el lienzo sin
tocar las fuentes hace que un rótulo de 8 pt se imprima a 8 pt.

Este script NO resuelve nada: alimenta `export_figures` de
`src/social_eq/followup.py` con los datos archivados de la corrida oficial, igual
que `E3_mapas/regenerar_figuras.py` hace con los mapas, y escribe en `salidas/`.
Nunca toca `outputs/` ni la carpeta de figuras del manuscrito.

Uso:
    python experiments/revision_2026/E7_figuras_impresion/regenerar_todas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(RAIZ / "src"))

from social_eq.followup import (  # noqa: E402
    ProjectPaths,
    export_figures,
    normalize_zone_code,
)

TABLAS = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
SHAPEFILE = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"
ENTRADAS = RAIZ / "data" / "inputs"
SALIDAS = Path(__file__).resolve().parent / "salidas"

S3_PROTECTED_SHARE = 0.15  # ModelConfig.s3_protected_share de la corrida oficial


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


def main() -> int:
    SALIDAS.mkdir(parents=True, exist_ok=True)

    utam = gpd.read_file(SHAPEFILE)
    utam["UTAM"] = utam["UTAM"].map(normalize_zone_code)
    if utam.crs is None:
        utam = utam.set_crs(4326)

    escenarios = cargar_escenarios()

    # priority_norm y vulnerability_index son insumos del modelo: iguales en los
    # cuatro escenarios. Se comprueba antes de usarlos.
    base = escenarios["S0"][["UTAM", "priority_norm", "vulnerability_index"]].copy()
    for nombre, df in escenarios.items():
        otro = df[["UTAM", "priority_norm", "vulnerability_index"]].sort_values("UTAM").reset_index(drop=True)
        ref = base.sort_values("UTAM").reset_index(drop=True)
        if not np.allclose(otro[["priority_norm", "vulnerability_index"]], ref[["priority_norm", "vulnerability_index"]]):
            raise RuntimeError(f"Los insumos de {nombre} no coinciden con los de S0; revisar las tablas archivadas.")
    df_inputs = base

    # Conjunto protegido de S3: cuantil superior de vulnerabilidad (mismo criterio
    # que la corrida oficial, ModelConfig.s3_protected_share = 0.15).
    corte = float(df_inputs["vulnerability_index"].quantile(1.0 - S3_PROTECTED_SHARE))
    protected_mask_s3 = (df_inputs["vulnerability_index"].to_numpy(dtype=float) >= corte)
    print(f"Unidades protegidas S3 (cuantil {1.0 - S3_PROTECTED_SHARE:.2f}): {int(protected_mask_s3.sum())}")

    robust_df = pd.read_csv(TABLAS / "table_robustness.csv")

    paths = ProjectPaths(
        root=RAIZ,
        input_dir=ENTRADAS,
        output_dir=SALIDAS,
        experiment_dir=SALIDAS,
        figures_dir=SALIDAS,
        tables_dir=TABLAS,      # se leen las tablas archivadas, no se escriben
        reference_dir=None,
    )

    export_figures(
        paths=paths,
        utam=utam,
        df_inputs=df_inputs,
        loss_hist=None,          # la figura de historia de entrenamiento no va al manuscrito
        scenario_outputs=escenarios,
        protected_mask_s3=protected_mask_s3,
        robust_df=robust_df,
    )

    # Figura 9 con las superposiciones del paquete de revision (barrido A1, benchmark A2):
    # el pipeline solo conoce S0-S3; la version del manuscrito anade el barrido de tolerancia
    # y el benchmark de equidad directa, ambos con sus puertos per capita.
    from social_eq.followup import fig_policy_tradeoff, ports_per_capita_by_quartile  # noqa: E402
    EXP = Path(__file__).resolve().parents[1]
    a1 = pd.read_csv(EXP / "A1_barrido_epsilon" / "salidas" / "tablas" / "resumen_barrido_epsilon.csv")
    a2 = pd.read_csv(EXP / "A2_benchmark_equidad" / "salidas" / "tablas" / "resumen_benchmark.csv")
    sweep = a1.loc[a1["fuente"].eq("corrida_nueva"), ["epsilon", "perdida_fidelidad_vs_S0_pct", "gini_ponderado_6d"]]
    bench = a2.loc[a2["fuente"].eq("corrida_nueva"), ["variante", "perdida_fidelidad_vs_S0_pct", "gini_ponderado_6d"]]
    bench_asig = pd.read_csv(EXP / "A2_benchmark_equidad" / "salidas" / "tablas" / "asignacion_benchmark_ponderado_k4.csv")
    bench_asig["UTAM"] = bench_asig["UTAM"].map(normalize_zone_code)
    vuln_csv = ENTRADAS / "utam_social_vulnerability.csv"
    bench_pc = ports_per_capita_by_quartile(bench_asig, vuln_csv, 0.25)
    percapita = {n: ports_per_capita_by_quartile(df, vuln_csv, 0.25) for n, df in escenarios.items()}
    tradeoff = pd.read_csv(TABLAS / "table_policy_tradeoff.csv")
    eqm = pd.read_csv(TABLAS / "table_equity_metrics_final.csv")
    s3 = eqm.loc[eqm["scenario"].astype(str).eq("S3")].iloc[0] if eqm["scenario"].astype(str).eq("S3").any() else None
    # Barras de la frontera de tolerancia (A1 ampliado): provision per capita a eps = 10 y 100
    pc_eps = pd.read_csv(EXP / "A1_barrido_epsilon" / "salidas" / "tablas" / "percapita_por_epsilon.csv")
    frontera = [
        (f"eps =\n{float(r['epsilon']):g}", (float(r["puertos_10k_cuartil_vulnerable"]), float(r["puertos_10k_resto"])))
        for _, r in pc_eps.iterrows() if float(r["epsilon"]) in (10.0, 100.0)
    ]
    fig_policy_tradeoff(SALIDAS, tradeoff, robust_df, s3, percapita, sweep_df=sweep, benchmark_df=bench, benchmark_percapita=bench_pc, frontier_percapita=frontera)
    print("Figura 9 con barrido y benchmark; puertos por 10k (vulnerable, resto):", {k: tuple(round(x, 2) for x in v) for k, v in percapita.items()}, "| benchmark k4:", tuple(round(x, 2) for x in bench_pc))

    generadas = sorted(p.name for p in SALIDAS.glob("*.png"))
    print(f"Figuras escritas en {SALIDAS.relative_to(RAIZ)}: {len(generadas)}")
    for nombre in generadas:
        print("  -", nombre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
