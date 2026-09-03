"""Cobertura del cuartil vulnerable a varios umbrales de tiempo de viaje.

El pipeline archiva la cobertura a 15 minutos (C^15). El manuscrito afirma además que la
cobertura es completa a umbrales más estrictos: este script la calcula a 5, 10, 15, 20 y 30
minutos para S0, S1, S2 y S3 con la misma matriz de tiempos y el mismo conjunto de zonas
vulnerables que el pipeline, y archiva el resultado en las tablas del paquete.

Uso: python scripts/coverage_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from social_eq.followup import (  # noqa: E402
    InputFiles,
    ModelConfig,
    ProjectPaths,
    coverage_within_threshold,
    load_minimal_inputs,
    normalize_zone_code,
)

TABLAS = ROOT / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
UMBRALES = (5.0, 10.0, 15.0, 20.0, 30.0)


def main() -> int:
    paths = ProjectPaths.from_root(ROOT)
    config = ModelConfig()
    inputs = InputFiles.from_dir(paths.input_dir)
    _utam, df_inputs, _vulnerability, _attention, tmat, _loss, _tm_alt = load_minimal_inputs(paths, inputs, config)
    utam_list = df_inputs["UTAM"].tolist()

    vul = df_inputs[["UTAM", "vulnerability_index"]].copy()
    corte = vul["vulnerability_index"].quantile(1 - config.vulnerability_top_share)
    high_mask = (vul["vulnerability_index"] >= corte).to_numpy()
    print(f"zonas del conjunto vulnerable: {int(high_mask.sum())} de {len(high_mask)}")

    filas = []
    for escenario in ("S0", "S1", "S2", "S3"):
        f = TABLAS / f"scenario_{escenario}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["UTAM"] = d["UTAM"].map(normalize_zone_code)
        orden = pd.DataFrame({"UTAM": [normalize_zone_code(u) for u in utam_list]})
        d = orden.merge(d[["UTAM", "ports_total"]], on="UTAM", how="left")
        puertos = d["ports_total"].to_numpy(dtype=float).reshape(-1, 1)
        for umbral in UMBRALES:
            c = coverage_within_threshold(puertos, tmat, high_mask, threshold=umbral)
            filas.append({
                "scenario": escenario,
                "threshold_min": umbral,
                "coverage_vulnerable": c,
                "zones_covered": int(round(c * high_mask.sum())),
                "zones_total": int(high_mask.sum()),
            })
    out = pd.DataFrame(filas)
    destino = TABLAS / "table_coverage_thresholds.csv"
    out.to_csv(destino, index=False)
    print(out.pivot(index="scenario", columns="threshold_min", values="coverage_vulnerable").to_string())
    print(f"escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
