"""Desviación realizada frente a la cota del presupuesto de fidelidad.

En el nivel inferior del modelo lexicográfico las variables de holgura $u_{iv}$ solo están
acotadas por debajo por $|n_{iv}-T_{iv}|$ y no entran en el objetivo, así que el solver puede
devolverlas pegadas a la restricción $\\sum u \\le (1+\\varepsilon)U^{\\star}$. La suma de
$u_{iv}$ es, por tanto, la cota del presupuesto y no la desviación efectivamente incurrida.
Este script recalcula ex post $\\sum_{i,v}|n_{iv}-T_{iv}|$ desde las asignaciones archivadas,
para los escenarios centrales y para el barrido de tolerancia, y archiva ambas cantidades.

Uso: python scripts/realized_deviation.py
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
    load_minimal_inputs,
    normalize_zone_code,
)

TABLAS = ROOT / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
A1 = ROOT / "experiments" / "revision_2026" / "A1_barrido_epsilon" / "salidas" / "tablas"
VENUES = ("residential", "workplace", "public")
QUOTAS = {"residential": 7352, "workplace": 2739, "public": 779}


def objetivos(df_inputs: pd.DataFrame) -> np.ndarray:
    """Objetivo T_iv: prioridad normalizada por pesos de sede, escalada a la cuota de cada sede."""
    pr = df_inputs["priority_norm"].to_numpy(dtype=float)
    pr = pr / pr.sum()
    cols = []
    for v in VENUES:
        w = df_inputs[v].to_numpy(dtype=float)
        cols.append(QUOTAS[v] * (pr * w) / (pr * w).sum())
    return np.column_stack(cols)


def main() -> int:
    paths = ProjectPaths.from_root(ROOT)
    inputs = InputFiles.from_dir(paths.input_dir)
    _u, df_inputs, _v, _a, _t, _l, _ta = load_minimal_inputs(paths, inputs, ModelConfig())
    T = objetivos(df_inputs)
    orden = pd.DataFrame({"UTAM": [normalize_zone_code(u) for u in df_inputs["UTAM"]]})

    def desviacion(ruta: Path) -> float | None:
        if not ruta.exists():
            return None
        d = pd.read_csv(ruta)
        d["UTAM"] = d["UTAM"].map(normalize_zone_code)
        cols = [f"ports_{v}" for v in VENUES]
        if not all(c in d.columns for c in cols):
            return None
        n = orden.merge(d[["UTAM"] + cols], on="UTAM", how="left")[cols].to_numpy(dtype=float)
        return float(np.abs(n - T).sum())

    u_star = desviacion(TABLAS / "scenario_S0.csv")
    eq = pd.read_csv(TABLAS / "table_equity_metrics_final.csv").set_index("scenario")
    filas = []
    for esc, eps in (("S0", 0.0), ("S1", 0.03), ("S2", 0.08), ("S3", 0.08)):
        real = desviacion(TABLAS / f"scenario_{esc}.csv")
        filas.append({"case": esc, "epsilon": eps, "realized_abs_deviation": real,
                      "budget_bound": (1.0 + eps) * u_star,
                      "archived_objective_abs_dev": float(eq.loc[esc, "objective_abs_dev"])})
    for f in sorted(A1.glob("asignacion_eps_*.csv")):
        eps = float(f.stem.split("_")[-1])
        real = desviacion(f)
        if real is None:
            continue
        filas.append({"case": f"sweep_eps_{eps:g}", "epsilon": eps, "realized_abs_deviation": real,
                      "budget_bound": (1.0 + eps) * u_star, "archived_objective_abs_dev": np.nan})
    out = pd.DataFrame(filas)
    out["realized_fidelity_loss_pct"] = 100.0 * (out["realized_abs_deviation"] - u_star) / u_star
    out["budget_fidelity_loss_pct"] = 100.0 * out["epsilon"]
    out["slack_ports"] = out["budget_bound"] - out["realized_abs_deviation"]
    destino = TABLAS / "table_realized_deviation.csv"
    out.to_csv(destino, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nU* = {u_star:.6f}\nescrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
