"""B6 — Tablas, bootstrap por bloques de localidad y veredicto GO/NO-GO (criterio de preregistro.md)."""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

SAL = Path(__file__).resolve().parent / "salidas"
METODOS = ["media_ciudad", "crudo_A", "EB", "RF_rezago", "IDW", "GCN", "STGNN"]
NB = 2000


def preparar(df):
    df = df.copy()
    om = df[["om_res", "om_tra", "om_pub"]].to_numpy(); ob = df[["omB_res", "omB_tra", "omB_pub"]].to_numpy()
    df["e_om"] = np.abs(om - ob).mean(1)
    df["e_l"] = np.abs(df["logL_hat"] - df["logL_B"])
    df["ok_om"] = df["ventana_B"].astype(bool) & (df["L_B"] > 0)
    df["ok_l"] = df["n_B"] > 0
    return df


def wmae_por_particion(g, col, ok, peso_loc=None):
    g = g[g[ok]]
    w = g["L_B"].to_numpy() * (1.0 if peso_loc is None else peso_loc[g["loc"].to_numpy()])
    num = pd.Series(w * g[col].to_numpy()).groupby(g["particion"].to_numpy()).sum()
    den = pd.Series(w).groupby(g["particion"].to_numpy()).sum()
    return (num / den.replace(0, np.nan))


def spearman_nivel(g):
    out = []
    for k, gg in g[g["ok_l"]].groupby("particion"):
        out.append(gg[["logL_hat", "logL_B"]].corr(method="spearman").iloc[0, 1])
    return np.nanmean(out)


def resumen(df):
    filas = []
    for (esc, modo, reg), g in df.groupby(["escala", "modo", "regimen"]):
        for m in METODOS:
            gm = g[g["metodo"] == m]
            e = wmae_por_particion(gm, "e_om", "ok_om"); el = wmae_por_particion(gm, "e_l", "ok_l")
            filas.append({"escala": esc, "modo": modo, "regimen": reg, "metodo": m, "K": int(e.notna().sum()),
                          "wMAE_omega": e.mean(), "DE_entre_particiones": e.std(ddof=1) if len(e) > 1 else np.nan,
                          "wMAE_logL": el.mean(), "spearman_nivel": spearman_nivel(gm),
                          "zonas_evaluadas_por_particion": gm[gm["ok_om"]].groupby("particion").size().mean()})
    return pd.DataFrame(filas)


def fiabilidad(df):
    """Fiabilidad entre mitades: correlación ponderada de ω entre A (crudo) y B, en zonas con dato en ambas."""
    filas = []
    for (esc, modo, reg), g in df[(df["metodo"] == "crudo_A") & (df["regimen"] != "kriging")].groupby(["escala", "modo", "regimen"]):
        g = g[g["ok_om"] & (g["n_A"] > 0)]
        rs = []
        for k, gg in g.groupby("particion"):
            w = gg["L_B"].to_numpy()
            for a, b in (("om_res", "omB_res"), ("om_tra", "omB_tra"), ("om_pub", "omB_pub")):
                x, y = gg[a].to_numpy(), gg[b].to_numpy()
                mx, my = np.average(x, weights=w), np.average(y, weights=w)
                c = np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
                rs.append((k, a, c))
        r = pd.DataFrame(rs, columns=["k", "v", "r"]).groupby("v")["r"].mean()
        e = wmae_por_particion(g, "e_om", "ok_om").mean()
        filas.append({"escala": esc, "modo": modo, "regimen": reg, "r_res": r.get("om_res"), "r_tra": r.get("om_tra"), "r_pub": r.get("om_pub"),
                      "wMAE_A_vs_B_con_dato": e, "cota_aprox_alcanzable": e / np.sqrt(2)})
    return pd.DataFrame(filas)


def bootstrap(df, tab):
    filas = []
    rng = np.random.default_rng(20261011)
    for (esc, modo, reg), g in df.groupby(["escala", "modo", "regimen"]):
        t = tab[(tab["escala"] == esc) & (tab["modo"] == modo) & (tab["regimen"] == reg)].set_index("metodo")
        mejor = t.loc[["EB", "RF_rezago"], "wMAE_omega"].idxmin()
        nloc = int(df["loc"].max()) + 1
        locs = np.unique(g["loc"])
        gs, gb = g[g["metodo"] == "STGNN"], g[g["metodo"] == mejor]
        dif, rel = [], []
        for _ in range(NB):
            cnt = np.bincount(rng.choice(locs, len(locs), replace=True), minlength=nloc).astype(float)
            a = wmae_por_particion(gs, "e_om", "ok_om", cnt).mean(); b = wmae_por_particion(gb, "e_om", "ok_om", cnt).mean()
            dif.append(a - b); rel.append((b - a) / b)
        a0, b0 = t.loc["STGNN", "wMAE_omega"], t.loc[mejor, "wMAE_omega"]
        filas.append({"escala": esc, "modo": modo, "regimen": reg, "mejor_clasico": mejor, "wMAE_STGNN": a0, "wMAE_mejor": b0,
                      "dif_STGNN_menos_mejor": a0 - b0, "IC95_inf": np.percentile(dif, 2.5), "IC95_sup": np.percentile(dif, 97.5),
                      "mejora_relativa": (b0 - a0) / b0, "mejora_rel_IC95_inf": np.percentile(rel, 2.5), "mejora_rel_IC95_sup": np.percentile(rel, 97.5),
                      "n_localidades": len(locs)})
    return pd.DataFrame(filas)


def main():
    dfs = [pd.read_csv(p) for p in sorted(SAL.glob("pred_*.csv.gz"))]
    df = preparar(pd.concat(dfs, ignore_index=True))
    df.loc[df["regimen"] == "tiempo", "particion"] = df.loc[df["regimen"] == "tiempo", "particion"].map({0: 0, 1: 1})
    # en "tiempo" la partición 0 es junio y la 1 es octubre: se reportan por separado
    df.loc[(df["regimen"] == "tiempo") & (df["particion"] == 1), "regimen"] = "tiempo_octubre"
    df.loc[df["regimen"] == "tiempo", "regimen"] = "tiempo_junio"
    tab = resumen(df); tab.to_csv(SAL / "tabla_resultados.csv", index=False)
    fia = fiabilidad(df); fia.to_csv(SAL / "fiabilidad_entre_mitades.csv", index=False)
    bs = bootstrap(df, tab); bs.to_csv(SAL / "bootstrap_stgnn_vs_mejor.csv", index=False)
    s0 = pd.concat([pd.read_csv(p) for p in sorted(SAL.glob("s0_*.csv"))])
    s0t = s0.groupby(["escala", "modo", "regimen", "metodo"])["puertos_mal_ubicados"].agg(["mean", "std", "count"]).reset_index()
    s0t.to_csv(SAL / "s0_puertos_mal_ubicados.csv", index=False)
    d = bs[(bs["escala"] == "zat") & (bs["modo"] == "auto") & (bs["regimen"] == "principal")].iloc[0]
    go = bool(d["IC95_sup"] < 0 and d["mejora_relativa"] >= 0.10)
    ver = {"veredicto": "GO" if go else "NO-GO", "criterio": "ZAT, principal, AUTO: IC95 de (STGNN - mejor de EB/RF) excluye 0 y mejora relativa >= 10 %",
           **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}}
    (SAL / "veredicto.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False, default=float))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(tab.round(4).to_string(index=False)); print(fia.round(3).to_string(index=False)); print(bs.round(4).to_string(index=False))
    print(s0t.round(1).to_string(index=False)); print(json.dumps(ver, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
