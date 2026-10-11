"""C3 — Nivel de llegadas de conductores de auto por ZAT, suavizado con los estimadores de B6.

Las predicciones archivadas de B6 (`pred_principal.csv.gz`) son del nivel de TODOS los extremos de
viaje del modo AUTO agrupado (conductores y pasajeros, salidas y llegadas), estimado con la mitad A
de los hogares. El preregistro de C3 (A5) pide el nivel de LLEGADAS DE CONDUCTORES. Por eso aquí se
reajustan los mismos estimadores de B6 (EB, RF con rezago espacial, GCN, ST-GNN), con las mismas
covariables de hogares, el mismo vecindario (k = 8, IDW) y las mismas rejillas de configuración,
importando su código de `B6_stgnn_honesto/run_b6.py`. Solo cambian:

  * el objetivo: viajes con `modo_principal_desagrupado` = "Vehículo privado como conductor"
    (los 8.110 de C2), contados solo en su ZAT de destino (las salidas no entran);
  * el criterio de selección de configuración: error absoluto del log-nivel ponderado por el nivel
    (en B6 era el error de ω para RF, IDW y redes; aquí ω no se usa);
  * el esquema de estimación final (ajuste cruzado): K = 5 particiones 50/50 de hogares; en cada
    una, cada mitad hace de A (ajuste interno A1→A2 / A2→A1, validación A2→A1, igual que B6) y la
    predicción con entrada A es una estimación del nivel poblacional (B6 reescala cada submuestra
    a la población). Nivel final = media de las 2K predicciones (en nivel, no en log).

Salida: `salidas/demanda_zat_estimadores.csv` (ZAT × estimador, nivel suavizado y nivel final
según la regla A5: en ZAT con ≥ 30 llegadas en muestra manda el valor observado de C2).
Uso: python demanda_c3.py [--K 5] [--njobs 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
REV = AQUI.parent
RAIZ = REV.parents[1]
sys.path.insert(0, str(REV / "B6_stgnn_honesto"))
sys.path.insert(0, str(RAIZ / "scripts"))
import run_b6 as B6  # noqa: E402

SAL = AQUI / "salidas"
CONDUCTOR = "Vehículo privado como conductor"
METODOS = ["EB", "RF_rezago", "GCN", "STGNN"]


def preparar_datos():
    """Caché de B6 con la máscara de conductores y sin salidas (solo cuentan las llegadas)."""
    from retrain_stgnn import leer_modulo
    D = B6.datos()
    hog = leer_modulo("a. Modulo hogares.csv").drop_duplicates("key_hg")
    via = leer_modulo("d. Modulo viajes.csv")
    via = via[via["key_hg"].isin(set(hog["key_hg"]))]
    assert len(via) == len(D["trips"]["hh"]), "los viajes no se alinean con la caché de B6"
    cond = (via["modo_principal_desagrupado"].astype(str).str.strip() == CONDUCTOR).to_numpy()
    t = dict(D["trips"])
    t["auto"] = cond                        # el 'modo auto' de B6 pasa a ser 'conductor de auto'
    t["h_sal"] = np.full_like(t["h_sal"], -1)  # sin salidas: el perfil y el nivel son solo llegadas
    D2 = dict(D); D2["trips"] = t
    B6._D = D2
    return D2, int(cond.sum())


def err_l(l, par):
    tg, m = par[1], par[2] & (par[1].n > 0)
    return float((tg.L[m] * np.abs(l[m] - tg.logL[m])).sum() / tg.L[m].sum())


def trabajo(k: int, mitad: int, D2):
    B6._D = D2
    t0 = time.time()
    H = D2["H"]; Z = D2["zat"]; N = len(Z["cod"]); loc = Z["loc"]
    rng = np.random.default_rng(500 + k)
    H1 = rng.random(H) < 0.5
    A = H1 if mitad == 0 else ~H1
    r2 = np.random.default_rng(600 + 10 * k + mitad)
    A1 = A & (r2.random(H) < 0.5); A2 = A & ~A1
    esc, modo = "zat", "auto"
    o_A1 = B6.Obs(esc, modo, A1); o_A2 = B6.Obs(esc, modo, A2)
    todos = np.ones(N, bool)
    pares_tr = [(o_A1, o_A2, todos), (o_A2, o_A1, todos)]
    val = (o_A2, o_A1, todos)
    tr_dir = [(o_A1, o_A2, todos)]
    inp = B6.Obs(esc, modo, A)
    out, eleg = {}, {}
    rej = [5.0, 20.0, 50.0, 200.0]
    kLb = min(rej, key=lambda kk: err_l(B6.pred_eb(val[0], 20.0, kk, loc)[1], val))
    out["EB"] = B6.pred_eb(inp, 20.0, kLb, loc)[1]; eleg["EB"] = kLb
    cfg_rf = [(None, 3), (None, 10), (12, 3), (12, 10)]
    best = min(cfg_rf, key=lambda c: err_l(B6.ajustar_rf(tr_dir, c, True)(val[0])[1], val))
    out["RF_rezago"] = B6.ajustar_rf(pares_tr, best, True)(inp)[1]; eleg["RF_rezago"] = best
    cfg_nn = [(16, 1e-4), (32, 1e-4), (16, 1e-3), (32, 1e-3)]
    for tipo in ("GCN", "STGNN"):
        res = []
        for c in cfg_nn:
            f, ep = B6.ajustar_red(tipo, tr_dir, val, c, semilla=k)
            res.append((err_l(f(val[0])[1], val), c, ep))
        _, c, ep = min(res, key=lambda r: r[0])
        f, _ = B6.ajustar_red(tipo, pares_tr, None, c, semilla=k, epocas_fijas=ep)
        out[tipo] = f(inp)[1]; eleg[tipo] = (c, ep)
    filas = pd.DataFrame({"ZAT": Z["cod"], "particion": k, "mitad": mitad, "n_A": inp.n, "logL_A": inp.logL,
                          **{m: out[m] for m in METODOS}})
    return filas, {"particion": k, "mitad": mitad, "seg": round(time.time() - t0, 1), **{m: str(v) for m, v in eleg.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--njobs", type=int, default=4)
    a = ap.parse_args()
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from joblib import Parallel, delayed
    SAL.mkdir(exist_ok=True)
    t0 = time.time()
    D2, ncond = preparar_datos()
    print(f"viajes de conductor de auto: {ncond}", flush=True)
    tareas = [(k, m) for k in range(a.K) for m in (0, 1)]
    res = Parallel(n_jobs=a.njobs, verbose=5)(delayed(trabajo)(k, m, D2) for k, m in tareas)
    pr = pd.concat([r[0] for r in res])
    pr.to_csv(SAL / "demanda_predicciones_crudas.csv.gz", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(SAL / "demanda_meta.csv", index=False)
    # nivel suavizado = media en nivel de las 2K predicciones
    niv = pr.groupby("ZAT")[METODOS].agg(lambda s: float(np.mean(np.expm1(np.maximum(s, 0))))).reset_index()
    obs = pd.read_csv(REV / "C2_parqueaderos_permanencias" / "salidas" / "llegadas_permanencia_por_zat.csv")
    niv = niv.merge(obs[["ZAT", "llegadas_muestra", "llegadas_expandidas"]], on="ZAT", how="outer")
    niv["llegadas_muestra"] = niv["llegadas_muestra"].fillna(0)
    niv["llegadas_expandidas"] = niv["llegadas_expandidas"].fillna(0)
    for m in METODOS:
        sm = niv[m]
        niv[f"demanda_{m}"] = np.where(niv["llegadas_muestra"] >= 30, niv["llegadas_expandidas"], sm)
    niv.to_csv(SAL / "demanda_zat_estimadores.csv", index=False)
    print(niv[[f"demanda_{m}" for m in METODOS]].describe().round(1).to_string())
    print(f"correlación Spearman entre estimadores (ZAT con dato):\n{niv[METODOS].corr('spearman').round(3)}")
    print(f"listo en {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
