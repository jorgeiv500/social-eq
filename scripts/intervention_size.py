"""Intensidad de provisión y tamaño de la intervención, por escenario.

Archiva las cantidades que el manuscrito cita en §5.3, §6 y §7: puertos por 10.000 residentes
en el cuartil vulnerable y en el resto, media de la ciudad, Gini per cápita ponderado por
población, y los puertos adicionales necesarios para llevar al cuartil vulnerable a la media
de la ciudad y a la paridad con las zonas restantes, en valor absoluto y como porcentaje de la
cuota. Todas las razones se calculan sobre las zonas con población censal.

Uso: python scripts/intervention_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from social_eq.followup import normalize_zone_code  # noqa: E402

TABLAS = ROOT / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
VULN = ROOT / "data" / "inputs" / "utam_social_vulnerability.csv"
CUOTA_TOTAL = 7352 + 2739 + 779


def gini_ponderado(valores: np.ndarray, pesos: np.ndarray) -> float:
    """Gini de una tasa (puertos por habitante) ponderado por población."""
    orden = np.argsort(valores)
    v, w = valores[orden], pesos[orden]
    acum_w = np.cumsum(w) / w.sum()
    acum_vw = np.cumsum(v * w) / (v * w).sum()
    return float(1.0 - np.sum((acum_vw[1:] + acum_vw[:-1]) * np.diff(acum_w)))


def main() -> int:
    pobl = pd.read_csv(VULN)
    pobl["UTAM"] = pobl["UTAM"].map(normalize_zone_code)
    filas = []
    for escenario in ("S0", "S1", "S2", "S3"):
        f = TABLAS / f"scenario_{escenario}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["UTAM"] = d["UTAM"].map(normalize_zone_code)
        d = d.merge(pobl[["UTAM", "dane_population"]], on="UTAM", how="left")
        corte = d["vulnerability_index"].quantile(0.75)
        vq_all = d["vulnerability_index"] >= corte
        con_pob = d["dane_population"].fillna(0.0) > 0
        d, vq = d.loc[con_pob].copy(), vq_all.loc[con_pob]
        pob_v, por_v = d.loc[vq, "dane_population"].sum(), d.loc[vq, "ports_total"].sum()
        pob_r, por_r = d.loc[~vq, "dane_population"].sum(), d.loc[~vq, "ports_total"].sum()
        tasa_v, tasa_r = 1e4 * por_v / pob_v, 1e4 * por_r / pob_r
        tasa_ciudad = 1e4 * (por_v + por_r) / (pob_v + pob_r)
        d["ports_per_10k"] = 1e4 * d["ports_total"] / d["dane_population"]
        g = gini_ponderado(d["ports_per_10k"].to_numpy(dtype=float), d["dane_population"].to_numpy(dtype=float))
        extra_media = tasa_ciudad * pob_v / 1e4 - por_v
        extra_paridad = tasa_r * pob_v / 1e4 - por_v
        filas.append({
            "scenario": escenario,
            "zones_with_population": int(con_pob.sum()),
            "vulnerable_zones": int(vq.sum()),
            "vulnerable_population": int(pob_v),
            "vulnerable_ports": int(por_v),
            "remaining_population": int(pob_r),
            "remaining_ports": int(por_r),
            "ports_per_10k_vulnerable": tasa_v,
            "ports_per_10k_remaining": tasa_r,
            "ports_per_10k_city": tasa_ciudad,
            "gini_ports_per_capita_population_weighted": g,
            "extra_ports_to_city_mean": extra_media,
            "extra_ports_to_city_mean_pct_quota": 100.0 * extra_media / CUOTA_TOTAL,
            "extra_ports_to_parity": extra_paridad,
            "extra_ports_to_parity_pct_quota": 100.0 * extra_paridad / CUOTA_TOTAL,
            "quota_total": CUOTA_TOTAL,
        })
    out = pd.DataFrame(filas)
    destino = TABLAS / "table_provision_intensity.csv"
    out.to_csv(destino, index=False)
    cols = ["scenario", "ports_per_10k_vulnerable", "ports_per_10k_remaining", "ports_per_10k_city",
            "gini_ports_per_capita_population_weighted", "extra_ports_to_city_mean",
            "extra_ports_to_city_mean_pct_quota", "extra_ports_to_parity", "extra_ports_to_parity_pct_quota"]
    print(out[cols].round(4).to_string(index=False))
    print(f"escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
