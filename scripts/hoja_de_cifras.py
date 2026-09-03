#!/usr/bin/env python
"""Hoja de cifras: extrae de las tablas oficiales y de los resúmenes de experimentos todos los
números que el manuscrito cita, en un JSON y un Markdown. Sin cifras a mano: cada valor lleva su
archivo de origen. Uso: python scripts/hoja_de_cifras.py [--tablas DIR] [--salida NOMBRE]

Escribe outputs/social_equity_followup_outputs/paper_v6_experiments/<NOMBRE>.json y .md.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in AQUI.parents if (p / "pyproject.toml").exists())
EXP = RAIZ / "experiments" / "revision_2026"
REENTRENO = RAIZ / "outputs" / "stgnn_retraining" / "peso_fexp_miles"
DESTINO = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments"


def frac36(x: float) -> str:
    return f"{x:.4f} ({int(round(x * 36))}/36)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tablas", default=str(RAIZ / "outputs/social_equity_followup_outputs/paper_v6_experiments/tables"))
    ap.add_argument("--salida", default="hoja_de_cifras")
    a = ap.parse_args(); T = Path(a.tablas); H: dict = {"origen_tablas": str(T)}

    eq = pd.read_csv(T / "table_equity_metrics_final.csv").set_index("scenario")
    u_star = float(eq.loc["S0", "objective_abs_dev"])
    H["u_star"] = u_star
    H["escenarios"] = {}
    for s in eq.index:
        r = eq.loc[s]
        H["escenarios"][s] = {
            "objective_abs_dev": float(r["objective_abs_dev"]),
            "perdida_fidelidad_pct": 100.0 * (float(r["objective_abs_dev"]) - u_star) / u_star,
            "gini": round(float(r["accessibility_gini"]), 6), "gini_ponderado": round(float(r["accessibility_weighted_gini"]), 6),
            "share_cuartil_vulnerable": round(float(r["share_access_most_vulnerable_quartile"]), 6),
            "cobertura_vulnerable": frac36(float(r["coverage_within_threshold_vulnerable"])),
            "cobertura_pct": round(100 * float(r["coverage_within_threshold_vulnerable"]), 2),
            "media_acceso_VH": round(float(r["mean_access_high_vulnerability"]), 6),
            "rho_prioridad_puertos": round(float(r["rho_priority_ports"]), 6),
        }
        for c in ("s3_floor_value", "s3_protected_count", "s3_total_shortfall", "s3_protected_share_meeting_floor", "s3_protected_p10_access", "s3_protected_mean_access", "s3_protected_min_access"):
            if c in eq.columns and pd.notna(r.get(c)): H["escenarios"][s][c] = float(r[c])
    # asignaciones y cambios
    asig = {s: pd.read_csv(T / f"scenario_{s}.csv") for s in ("S0", "S1", "S2", "S3")}
    for s, df in asig.items():
        H["escenarios"][s]["zonas_con_puertos"] = int((df["ports_total"] > 0).sum())
        H["escenarios"][s]["total_puertos"] = int(df["ports_total"].sum())
    def cambios(a, b):
        m = asig[a][["UTAM", "ports_total"]].merge(asig[b][["UTAM", "ports_total"]], on="UTAM", suffixes=("_a", "_b"))
        d = m["ports_total_b"] - m["ports_total_a"]
        return {"zonas_cambiadas": int((d != 0).sum()), "puertos_movidos": float(np.abs(d).sum() / 2), "ganan": m.loc[d > 0, "UTAM"].tolist(), "pierden": m.loc[d < 0, "UTAM"].tolist(), "max_abs_cambio": int(np.abs(d).max())}
    H["cambios"] = {"S1_vs_S0": cambios("S0", "S1"), "S2_vs_S1": cambios("S1", "S2"), "S3_vs_S2": cambios("S2", "S3"), "S2_vs_S0": cambios("S0", "S2")}
    # robustez
    rb = pd.read_csv(T / "table_robustness.csv")
    H["robustez"] = {"rank_stability_min": float(rb["rank_stability"].min()), "rank_stability_max": float(rb["rank_stability"].max()),
                     "filas": rb[["alternative_specification", "delta_weighted_gini", "delta_vulnerable_coverage", "rank_stability"]].round(6).to_dict("records")}
    # GNN
    g = T / "table_gnn_performance_available.csv"
    if g.exists(): H["gnn"] = pd.read_csv(g).round(4).to_dict("records")
    # experimentos de la revisión (si existen)
    def leer(p):
        p = EXP / p; return pd.read_csv(p).round(6).to_dict("records") if p.exists() else None
    H["experimentos"] = {
        "A1_barrido_epsilon": leer("A1_barrido_epsilon/salidas/tablas/resumen_barrido_epsilon.csv"),
        "A2_benchmark": leer("A2_benchmark_equidad/salidas/tablas/resumen_benchmark.csv"),
        "A3_crecimiento": leer("A3_escenarios_crecimiento/salidas/tablas/resumen_crecimiento.csv"),
        "A4_topes": leer("A4_topes_capacidad/salidas/tablas/diagnostico_topes.csv"),
        "B1_baselines": leer("B1_baselines/resultados/b1_metricas_cv.csv"),
        "B2_ablacion_omega": leer("B2_ablacion_omega/salidas/tablas/resumen_ablacion_omega.csv"),
        "B3_tenencia": leer("B3_tenencia/salidas/tabla_correlaciones.csv"),
        "B4_contrafactual": leer("B4_contrafactual_prioridad/salidas/tablas/resumen_contrafactual.csv"),
        "C1_piso": leer("C1_piso_vinculante/salidas/tablas/resumen_piso_vinculante.csv"),
    }
    a4 = EXP / "A4_topes_capacidad/salidas/tablas/resumen_diagnostico.json"
    if a4.exists(): H["experimentos"]["A4_resumen"] = json.loads(a4.read_text())
    stg = RAIZ / "data/inputs/priority_utam_scores.csv"
    prot = REENTRENO / "stgnn_run_protocol.json"
    if prot.exists(): H["stgnn_protocolo"] = json.loads(prot.read_text())
    ms = REENTRENO / "model_metrics_split.csv"
    if ms.exists(): H["stgnn_metricas"] = pd.read_csv(ms).round(4).to_dict("records")
    (DESTINO / f"{a.salida}.json").write_text(json.dumps(H, indent=2, default=float), encoding="utf-8")
    # resumen legible
    lines = [f"# Hoja de cifras — tablas: {T}", ""]
    lines.append("| Escenario | pérdida fidelidad % | Gini | Gini pond. | cobertura vulnerable | share cuartil | media V^H | ρ prior-puertos | zonas con puertos |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s, r in H["escenarios"].items():
        lines.append(f"| {s} | {r['perdida_fidelidad_pct']:.1f} | {r['gini']:.6f} | {r['gini_ponderado']:.6f} | {r['cobertura_vulnerable']} | {r['share_cuartil_vulnerable']:.6f} | {r['media_acceso_VH']:.6f} | {r['rho_prioridad_puertos']:.6f} | {r['zonas_con_puertos']} |")
    lines.append(""); lines.append(f"u* = {u_star:.6f}")
    for k, v in H["cambios"].items():
        lines.append(f"- {k}: {v['zonas_cambiadas']} zonas cambian, {v['puertos_movidos']:.0f} puertos movidos, máx |Δ| {v['max_abs_cambio']}; ganan {v['ganan'][:12]}; pierden {v['pierden'][:12]}")
    lines.append(f"- robustez: rank stability {H['robustez']['rank_stability_min']:.4f}–{H['robustez']['rank_stability_max']:.4f}")
    for r in H["robustez"]["filas"]:
        lines.append(f"  - {r['alternative_specification']}: ΔGini_w {r['delta_weighted_gini']:+.6f}, Δcob {r['delta_vulnerable_coverage']:+.4f}, rank {r['rank_stability']:.4f}")
    (DESTINO / f"{a.salida}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12])); print(f"... escrito {a.salida}.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
