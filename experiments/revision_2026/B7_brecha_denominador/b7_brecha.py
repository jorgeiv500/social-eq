"""B7 — Brecha de provisión: denominador, índice, geografía e incertidumbre por muestreo.

Revisión JTRG-D-26-00692 (R1.B/R1.D, R2 «¿es pública la brecha?», R3.1–R3.2). Cálculos locales,
sin NEOS, sobre resultados ya publicados. No toca `src/`, `latex/` ni resultados del paper.

Uso:  .venv/bin/python b7_brecha.py [--B 200]

Insumos (todos de solo lectura):
- asignaciones publicadas S0/S2: outputs/.../paper_v6_experiments/tables/scenario_S*.csv
- población DANE e indicadores del índice: data/inputs/utam_social_vulnerability.csv
- tenencia en arriendo (B3): ../B3_tenencia/salidas/utam_tenencia.csv
- estrato medio de la encuesta: data/inputs/priority_utam_scores.csv (avg_stratum)
- localidad: shapefile UTAM2023 (LOCNombre)
- viajes EM-2023 (módulo de viajes, carpeta EM2023_DIR) para los viajes atraídos por propósito
- caché de hogares y viajes de B6 (../B6_stgnn_honesto/salidas/cache_b6.pkl) para el bootstrap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
SAL = AQUI / "salidas"
TABLAS = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(AQUI.parent / "B6_stgnn_honesto"))
from retrain_stgnn import canon, leer_modulo, to_num  # noqa: E402

SEDES = ["residential", "workplace", "public"]
CUOTAS = np.array([7352, 2739, 779])
RES_H = [22, 23, 0, 1, 2, 3, 4, 5]
WORK_H = list(range(8, 18))
PUB_H = [18, 19, 20, 21]


# ------------------------------------------------------------------ insumos comunes
def zonas_base() -> pd.DataFrame:
    """Una fila por UTAM (142) con asignaciones S0/S2, índice, población, localidad."""
    s0 = pd.read_csv(TABLAS / "scenario_S0.csv")
    s2 = pd.read_csv(TABLAS / "scenario_S2.csv")
    for d in (s0, s2):
        d["UTAM"] = d["UTAM"].map(canon)
    cols = ["UTAM", *[f"ports_{s}" for s in SEDES], "ports_total"]
    z = s0[[*cols, "vulnerability_index", "high_vulnerability", "priority_norm"]].rename(
        columns={c: f"{c}_S0" for c in cols if c != "UTAM"})
    z = z.merge(s2[cols].rename(columns={c: f"{c}_S2" for c in cols if c != "UTAM"}), on="UTAM", how="left")
    v = pd.read_csv(RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    v["UTAM"] = v["UTAM"].map(canon)
    z = z.merge(v, on="UTAM", how="left", suffixes=("", "_crudo"))
    z["dane_population"] = z["dane_population"].fillna(0.0)
    u = gpd.read_file(RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023")
    u["UTAM"] = u["UTAM"].map(canon)
    z = z.merge(u[["UTAM", "LOCNombre", "UTAMNombre"]], on="UTAM", how="left")
    ten = pd.read_csv(AQUI.parent / "B3_tenencia" / "salidas" / "utam_tenencia.csv")
    ten["UTAM"] = ten["UTAM"].map(canon)
    z = z.merge(ten[["UTAM", "pct_arriendo_subarriendo"]], on="UTAM", how="left")
    pr = pd.read_csv(RAIZ / "data" / "inputs" / "priority_utam_scores.csv")
    pr["UTAM"] = pr["UTAM"].map(canon)
    z = z.merge(pr[["UTAM", "avg_stratum"]], on="UTAM", how="left")
    assert len(z) == 142 and z["UTAM"].is_unique
    # cuartil del paper: cuantil 0,75 del índice normalizado sobre las 142 zonas
    vq = z["vulnerability_index"] >= z["vulnerability_index"].quantile(0.75)
    assert (vq.astype(int) == z["high_vulnerability"]).all(), "el cuartil reconstruido no coincide con el publicado"
    z["VQ"] = vq
    return z


def viajes_atraidos(codigos: list[str]) -> pd.DataFrame:
    """Viajes expandidos (fexp_vj) por UTAM de destino y propósito, todos los modos y solo AUTO."""
    via = leer_modulo("d. Modulo viajes.csv")
    via["d"] = via["utam_des"].map(canon)
    via["w"] = to_num(via["fexp_vj"]).fillna(0.0)
    mot = via["motivo_viaje"].astype(str).str.strip()
    via["trabajo"] = mot.eq("A trabajar")
    via["hogar"] = mot.eq("A regresar al hogar")
    via["auto"] = via["modo_principal_agrupado"].astype(str).str.upper().str.strip().eq("AUTO")
    filas = []
    for modo, m in (("todos", np.ones(len(via), bool)), ("auto", via["auto"].to_numpy())):
        g = via[m]
        agg = pd.DataFrame({
            f"atraidos_total_{modo}": g.groupby("d")["w"].sum(),
            f"atraidos_trabajo_{modo}": g[g["trabajo"]].groupby("d")["w"].sum(),
            f"atraidos_no_hogar_{modo}": g[~g["hogar"]].groupby("d")["w"].sum(),
            f"atraidos_no_hogar_no_trabajo_{modo}": g[~g["hogar"] & ~g["trabajo"]].groupby("d")["w"].sum(),
        })
        filas.append(agg)
    out = pd.concat(filas, axis=1).reindex(codigos).fillna(0.0)
    out.index.name = "UTAM"
    diag = {
        "viajes_registro": int(len(via)),
        "viajes_expandidos": float(via["w"].sum()),
        "pct_expandido_con_destino_en_UTAM": float(100 * via.loc[via["d"].isin(codigos), "w"].sum() / via["w"].sum()),
    }
    return out.reset_index(), diag


def tasa(num: pd.Series, den: pd.Series, mask: pd.Series) -> float:
    m = mask & (den > 0)
    return float(1e4 * num[m].sum() / den[m].sum())


def puertos_media_paridad(ports: pd.Series, pop: pd.Series, vq: pd.Series) -> dict:
    """Misma aritmética que table_provision_intensity.csv (zonas con población > 0)."""
    m = pop > 0
    p_v, p_r = ports[m & vq].sum(), ports[m & ~vq].sum()
    n_v, n_r = pop[m & vq].sum(), pop[m & ~vq].sum()
    r_v, r_r, r_c = 1e4 * p_v / n_v, 1e4 * p_r / n_r, 1e4 * (p_v + p_r) / (n_v + n_r)
    return {"vulnerable": r_v, "resto": r_r, "ciudad": r_c, "diferencia": r_r - r_v, "razon_resto_vulnerable": r_r / r_v,
            "puertos_a_media": n_v * (r_c - r_v) / 1e4, "puertos_a_paridad": n_v * (r_r - r_v) / 1e4}


# ------------------------------------------------------------------ 1. brecha por sede
def punto1(z: pd.DataFrame, viajes: pd.DataFrame) -> pd.DataFrame:
    z = z.merge(viajes, on="UTAM", how="left")
    filas = []
    defs = [
        ("residential", "residentes (DANE)", "dane_population", "todos"),
        ("workplace", "viajes atraídos, propósito trabajo", "atraidos_trabajo_todos", "todos"),
        ("workplace", "viajes atraídos, propósito trabajo", "atraidos_trabajo_auto", "auto"),
        ("public", "viajes atraídos sin regreso al hogar ni trabajo", "atraidos_no_hogar_no_trabajo_todos", "todos"),
        ("public", "viajes atraídos sin regreso al hogar ni trabajo", "atraidos_no_hogar_no_trabajo_auto", "auto"),
        ("public", "viajes atraídos sin regreso al hogar (incl. trabajo)", "atraidos_no_hogar_todos", "todos"),
        ("total", "residentes (DANE) [titular publicado]", "dane_population", "todos"),
    ]
    for esc in ("S0", "S2"):
        for sede, den_txt, den, modo in defs:
            num = z[f"ports_{sede}_{esc}"] if sede != "total" else z[f"ports_total_{esc}"]
            for universo, mu in (("zonas con población DANE", z["dane_population"] > 0), ("todas las zonas con denominador", pd.Series(True, index=z.index))):
                if den == "dane_population" and universo != "zonas con población DANE":
                    continue
                rv, rr = tasa(num, z[den], mu & z["VQ"]), tasa(num, z[den], mu & ~z["VQ"])
                filas.append({"escenario": esc, "sede": sede, "denominador": den_txt, "modo_viajes": modo, "universo": universo,
                              "puertos_vulnerable": int(num[mu & z["VQ"] & (z[den] > 0)].sum()),
                              "puertos_resto": int(num[mu & ~z["VQ"] & (z[den] > 0)].sum()),
                              "denominador_vulnerable": float(z.loc[mu & z["VQ"], den].sum()),
                              "denominador_resto": float(z.loc[mu & ~z["VQ"], den].sum()),
                              "por_10k_vulnerable": rv, "por_10k_resto": rr, "razon_resto_vulnerable": rr / rv})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ 2. sensibilidad al índice
def pca_bateria(z: pd.DataFrame) -> np.ndarray:
    """Réplica exacta del índice PCA de la batería de robustez (followup.py, 'PCA weights')."""
    v = pd.read_csv(RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    v["UTAM"] = v["UTAM"].map(canon)
    cols = [c for c in v.columns if c not in ["UTAM", "vulnerability_index"] and pd.api.types.is_numeric_dtype(v[c])]
    x = v.set_index("UTAM").reindex(z["UTAM"])[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median())
    xs = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    uu, ss, _ = np.linalg.svd(xs.to_numpy(float), full_matrices=False)
    pc1 = uu[:, 0] * ss[0]
    if np.corrcoef(pc1, z["vulnerability_index"].to_numpy(float))[0, 1] < 0:
        pc1 = -pc1
    return (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9), cols


def punto2(z: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pca, cols_pca = pca_bateria(z)
    med = lambda s: s.fillna(s.median())  # noqa: E731  (misma imputación por mediana que el índice publicado)
    indices = {
        "publicado (personas/vivienda + densidad)": (z["vulnerability_index"], "142 zonas"),
        "(a) personas por vivienda": (med(z["persons_per_dwelling"]), "142 zonas (mediana imputada)"),
        "(b) densidad de población": (med(z["population_density_ha"]), "142 zonas (mediana imputada)"),
        "(c) % hogares en arriendo (B3)": (z["pct_arriendo_subarriendo"], "80 zonas con dato"),
        "(d) PCA de la batería de robustez": (pd.Series(pca, index=z.index), "142 zonas"),
        "(e) estrato medio invertido (EM-2023)": (-z["avg_stratum"], "zonas con hogares encuestados"),
    }
    filas = []
    for nombre, (idx, univ) in indices.items():
        ok = idx.notna()
        vq = pd.Series(False, index=z.index)
        vq[ok] = idx[ok] >= idx[ok].quantile(0.75)
        n_v = int(vq.sum())
        orig = z["VQ"]
        en_univ = int((orig & ok).sum())
        siguen = int((orig & vq).sum())
        fila = {"indice": nombre, "universo": univ, "zonas_en_universo": int(ok.sum()), "zonas_cuartil": n_v,
                "vq_original_en_universo": en_univ, "vq_original_que_siguen": siguen,
                "vq_original_que_salen": en_univ - siguen,
                "spearman_con_publicado": float(pd.Series(idx[ok]).rank().corr(z.loc[ok, "vulnerability_index"].rank()))}
        pop = z["dane_population"].where(ok, 0.0)
        fila["poblacion_cuartil"] = float(pop[vq].sum())
        for esc in ("S0", "S2"):
            r = puertos_media_paridad(z[f"ports_total_{esc}"].where(ok, 0), pop, vq)
            rres = puertos_media_paridad(z[f"ports_residential_{esc}"].where(ok, 0), pop, vq)
            fila.update({f"{esc}_por10k_vulnerable": r["vulnerable"], f"{esc}_por10k_resto": r["resto"],
                         f"{esc}_razon": r["razon_resto_vulnerable"], f"{esc}_puertos_a_media": r["puertos_a_media"],
                         f"{esc}_puertos_a_paridad": r["puertos_a_paridad"],
                         f"{esc}_resid_por10k_vulnerable": rres["vulnerable"], f"{esc}_resid_por10k_resto": rres["resto"]})
        filas.append(fila)
    # correlación de Spearman entre provisión per cápita S2 y cada indicador (zonas con población)
    m = z["dane_population"] > 0
    pc = z.loc[m, "ports_total_S2"] / z.loc[m, "dane_population"]
    rho = {n: float(idx[m].rank().corr(pc.rank())) for n, (idx, _) in indices.items()}
    return pd.DataFrame(filas), {"columnas_pca": cols_pca, "spearman_indice_vs_puertos_por_hab_S2": rho}


# ------------------------------------------------------------------ 3. geografía
punto3_gradiente: dict = {}


def punto3(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = z["dane_population"] > 0
    d = z.loc[m].copy()
    d["por10k_S2"] = 1e4 * d["ports_total_S2"] / d["dane_population"]
    d["resid_por10k_S2"] = 1e4 * d["ports_residential_S2"] / d["dane_population"]
    # distancia del centroide de cada UTAM al centroide de La Candelaria (centro histórico), EPSG:3116
    u = gpd.read_file(RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023").to_crs(3116)
    u["UTAM"] = u["UTAM"].map(canon)
    centro = u.loc[u["LOCNombre"].eq("CANDELARIA")].dissolve().centroid.iloc[0]
    dist = pd.Series(u.centroid.distance(centro).to_numpy() / 1000, index=u["UTAM"])
    d["dist_centro_km"] = d["UTAM"].map(dist)
    cols = ["UTAM", "UTAMNombre", "LOCNombre", "dane_population", "ports_total_S2", "ports_residential_S2",
            "por10k_S2", "resid_por10k_S2", "dist_centro_km", "VQ"]
    p_todas = d.sort_values("por10k_S2").head(10)[cols].assign(filtro="población > 0")
    p_10k = d[d["dane_population"] >= 10000].sort_values("por10k_S2").head(10)[cols].assign(filtro="población >= 10.000")
    peores = pd.concat([p_todas, p_10k], ignore_index=True)
    g = d[d["dane_population"] >= 10000]
    punto3_gradiente["spearman_por10k_S2_vs_dist_centro_pob10k"] = float(g["por10k_S2"].rank().corr(g["dist_centro_km"].rank()))
    punto3_gradiente["spearman_resid_por10k_S2_vs_dist_centro_pob10k"] = float(g["resid_por10k_S2"].rank().corr(g["dist_centro_km"].rank()))
    punto3_gradiente["zonas_pob10k"] = int(len(g))
    for a, b_ in ((0, 5), (5, 10), (10, 15), (15, 1e9)):
        gg = g[(g["dist_centro_km"] >= a) & (g["dist_centro_km"] < b_)]
        punto3_gradiente[f"por10k_S2_anillo_{a}_{int(min(b_, 99))}km"] = float(1e4 * gg["ports_total_S2"].sum() / gg["dane_population"].sum())
        punto3_gradiente[f"zonas_anillo_{a}_{int(min(b_, 99))}km"] = int(len(gg))
    loc = d.groupby("LOCNombre").agg(zonas=("UTAM", "size"), zonas_VQ=("VQ", "sum"), poblacion=("dane_population", "sum"),
                                     puertos_S2=("ports_total_S2", "sum"), puertos_resid_S2=("ports_residential_S2", "sum"))
    loc["por10k_S2"] = 1e4 * loc["puertos_S2"] / loc["poblacion"]
    loc["resid_por10k_S2"] = 1e4 * loc["puertos_resid_S2"] / loc["poblacion"]
    loc["zonas_en_10_peores"] = loc.index.map(peores["LOCNombre"].value_counts()).fillna(0).astype(int)
    return peores, loc.sort_values("por10k_S2").reset_index()


# ------------------------------------------------------------------ 4. bootstrap
def s0_exacto(t: np.ndarray) -> np.ndarray:
    """Igual que B5/B6: S0 es separable por sede; floor + remanente a las fracciones mayores."""
    n = np.zeros_like(t)
    for v, q in enumerate(CUOTAS):
        b = np.floor(t[:, v]); r = int(round(q - b.sum())); nv = b.copy()
        if r > 0:
            nv[np.argsort(-(t[:, v] - b), kind="stable")[:r]] += 1
        n[:, v] = nv
    return n


def omega_encuesta(C: np.ndarray) -> np.ndarray:
    """ω observado: participación de extremos de viaje (salidas por hora de inicio, llegadas por hora
    de fin) en las ventanas residencial 22–5, trabajo 8–17 y público 18–21 (como B6)."""
    tot = C[:, :24] + C[:, 24:]
    W = np.c_[tot[:, RES_H].sum(1), tot[:, WORK_H].sum(1), tot[:, PUB_H].sum(1)]
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.maximum(s, 1e-12), 1 / 3)


def omega_atencion(z: pd.DataFrame) -> np.ndarray:
    """ω del pipeline publicado: atención temporal del ST-GNN sumada por ventana (build_venue_weights)."""
    att = pd.read_csv(RAIZ / "data" / "inputs" / "temporal_attention_by_utam.csv")
    att["UTAM"] = att["UTAM"].map(canon)
    att = att.set_index("UTAM").reindex(z["UTAM"])
    W = np.c_[att[[f"hour_{h}" for h in RES_H]].sum(1), att[[f"hour_{h}" for h in WORK_H]].sum(1),
              att[[f"hour_{h}" for h in PUB_H]].sum(1)]
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.maximum(s, 1e-12), 1 / 3)


def minmax(x: np.ndarray) -> np.ndarray:
    return np.maximum((x - x.min()) / (x.max() - x.min() + 1e-12), 1e-9)


def metas(prio: np.ndarray, om: np.ndarray) -> np.ndarray:
    """make_targets del pipeline: t_iv = Q_v · s_i ω_iv / Σ_j s_j ω_jv."""
    sh = prio[:, None] * om
    return sh / sh.sum(0, keepdims=True) * CUOTAS


def punto4(z: pd.DataFrame, B: int, semilla: int = 20261011) -> tuple[pd.DataFrame, dict]:
    import pickle
    with open(AQUI.parent / "B6_stgnn_honesto" / "salidas" / "cache_b6.pkl", "rb") as f:
        D = pickle.load(f)
    cod = D["utam"]["cod"]; N = len(cod)
    orden = pd.Series(range(N), index=cod).reindex(z["UTAM"]).to_numpy()
    assert not np.isnan(orden.astype(float)).any()
    t, hh = D["trips"], D["hh"]
    z = z.reset_index(drop=True)
    pop = z["dane_population"]; vq = z["VQ"].reset_index(drop=True)
    sel = [(t["u_ori"] >= 0) & (t["h_sal"] >= 0), (t["u_des"] >= 0) & (t["h_lle"] >= 0)]
    estr = hh["utam"].copy()  # estrato = UTAM del hogar (-1 = fuera de las 142, estrato propio)
    grupos = [np.where(estr == g)[0] for g in np.unique(estr)]

    om_att = omega_atencion(z)

    def una(conteo_hh: np.ndarray, variante: str) -> dict:
        w = t["w"] * conteo_hh[t["hh"]]
        C = np.zeros((N, 48))
        np.add.at(C, (t["u_ori"][sel[0]], t["h_sal"][sel[0]]), w[sel[0]])
        np.add.at(C, (t["u_des"][sel[1]], t["h_lle"][sel[1]] + 24), w[sel[1]])
        C = C[orden]  # al orden de las tablas publicadas
        om = omega_encuesta(C) if variante == "directa" else om_att
        n = s0_exacto(metas(minmax(C.sum(1)), om))
        tot = pd.Series(n.sum(1)); res = pd.Series(n[:, 0])
        r = puertos_media_paridad(tot, pop, vq)
        rr = puertos_media_paridad(res, pop, vq)
        return {**r, "resid_vulnerable": rr["vulnerable"], "resid_resto": rr["resto"], "_n": n}

    pub = np.c_[[z[f"ports_{s}_S0"].to_numpy() for s in SEDES]].T
    variantes = ("directa", "omega_atencion_fija")
    base = {v: una(np.ones(len(hh["w"])), v) for v in variantes}
    rng = np.random.default_rng(semilla)
    filas = []
    for b in range(B):
        c = np.zeros(len(hh["w"]))
        for g in grupos:
            if len(g):
                np.add.at(c, rng.choice(g, size=len(g), replace=True), 1.0)
        for v in variantes:  # mismas réplicas de hogares para las dos variantes
            r = una(c, v)
            r["puertos_movidos_vs_muestra_completa"] = float(np.abs(r["_n"] - base[v]["_n"]).sum() / 2)
            filas.append({"variante": v, "replica": b, **{k: x for k, x in r.items() if k != "_n"}})
    df = pd.DataFrame(filas)
    resumen = {}
    for v in variantes:
        dv = df[df["variante"] == v]
        resumen[v] = {}
        for col in ["vulnerable", "resto", "diferencia", "razon_resto_vulnerable", "puertos_a_media", "puertos_a_paridad",
                    "resid_vulnerable", "resid_resto", "puertos_movidos_vs_muestra_completa"]:
            q = dv[col].quantile([0.025, 0.5, 0.975])
            resumen[v][col] = {"muestra_completa": None if col.startswith("puertos_movidos") else float(base[v][col]),
                               "mediana": float(q[0.5]), "ic95_inf": float(q[0.025]), "ic95_sup": float(q[0.975])}
        resumen[v]["puertos_movidos_muestra_completa_vs_S0_publicado"] = float(np.abs(base[v]["_n"] - pub).sum() / 2)
    resumen["_contexto"] = {"B": B, "semilla": semilla, "estratos_utam_hogar": len(grupos), "hogares": int(len(hh["w"]))}
    return df, resumen


def descomposicion_meta(z: pd.DataFrame) -> pd.DataFrame:
    """S0 exacto con cada combinación de prioridad s_i y reparto por sede ω (muestra completa)."""
    import pickle
    with open(AQUI.parent / "B6_stgnn_honesto" / "salidas" / "cache_b6.pkl", "rb") as f:
        D = pickle.load(f)
    cod = D["utam"]["cod"]; t = D["trips"]
    orden = pd.Series(range(len(cod)), index=cod).reindex(z["UTAM"]).to_numpy()
    C = np.zeros((len(cod), 48))
    m = (t["u_ori"] >= 0) & (t["h_sal"] >= 0); np.add.at(C, (t["u_ori"][m], t["h_sal"][m]), t["w"][m])
    m = (t["u_des"] >= 0) & (t["h_lle"] >= 0); np.add.at(C, (t["u_des"][m], t["h_lle"][m] + 24), t["w"][m])
    C = C[orden]
    pub = np.c_[[z[f"ports_{s}_S0"].to_numpy() for s in SEDES]].T
    prios = {"ST-GNN publicada (priority_norm)": z["priority_norm"].to_numpy(float), "encuesta directa (nivel de viajes)": minmax(C.sum(1))}
    oms = {"atención ST-GNN publicada": omega_atencion(z), "encuesta observada": omega_encuesta(C)}
    filas = []
    for pn, p in prios.items():
        for on, o in oms.items():
            n = s0_exacto(metas(p, o))
            r = puertos_media_paridad(pd.Series(n.sum(1)), z["dane_population"].reset_index(drop=True), z["VQ"].reset_index(drop=True))
            rr = puertos_media_paridad(pd.Series(n[:, 0]), z["dane_population"].reset_index(drop=True), z["VQ"].reset_index(drop=True))
            filas.append({"prioridad": pn, "reparto_por_sede": on, "por10k_vulnerable": r["vulnerable"], "por10k_resto": r["resto"],
                          "razon": r["razon_resto_vulnerable"], "puertos_a_media": r["puertos_a_media"], "puertos_a_paridad": r["puertos_a_paridad"],
                          "resid_por10k_vulnerable": rr["vulnerable"], "resid_por10k_resto": rr["resto"],
                          "puertos_movidos_vs_S0_publicado": float(np.abs(n - pub).sum() / 2)})
    return pd.DataFrame(filas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    SAL.mkdir(exist_ok=True)
    z = zonas_base()
    viajes, diag_v = viajes_atraidos(z["UTAM"].tolist())
    viajes.to_csv(SAL / "b7_viajes_atraidos_por_utam.csv", index=False)

    p1 = punto1(z, viajes)
    p1.to_csv(SAL / "b7_1_brecha_por_sede.csv", index=False)

    p2, extra2 = punto2(z)
    p2.to_csv(SAL / "b7_2_sensibilidad_indice.csv", index=False)

    peores, loc = punto3(z)
    peores.to_csv(SAL / "b7_3_diez_zonas_menor_provision_S2.csv", index=False)
    loc.to_csv(SAL / "b7_3_provision_por_localidad_S2.csv", index=False)

    rep, res4 = punto4(z, a.B)
    rep.to_csv(SAL / "b7_4_bootstrap_replicas.csv", index=False)
    descomposicion_meta(z).to_csv(SAL / "b7_4_descomposicion_meta_S0.csv", index=False)

    # verificación contra la tabla publicada
    pub = pd.read_csv(TABLAS / "table_provision_intensity.csv").set_index("scenario")
    verif = {}
    for esc in ("S0", "S2"):
        r = puertos_media_paridad(z[f"ports_total_{esc}"], z["dane_population"], z["VQ"])
        verif[esc] = {"recalculado": r, "publicado": {
            "vulnerable": float(pub.loc[esc, "ports_per_10k_vulnerable"]), "resto": float(pub.loc[esc, "ports_per_10k_remaining"]),
            "puertos_a_media": float(pub.loc[esc, "extra_ports_to_city_mean"]), "puertos_a_paridad": float(pub.loc[esc, "extra_ports_to_parity"])}}
    with open(SAL / "b7_resumen.json", "w") as f:
        json.dump({"viajes": diag_v, "verificacion_tabla_publicada": verif, "punto2": extra2, "punto3_gradiente_centro": punto3_gradiente, "punto4_bootstrap": res4},
                  f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({"boot": res4}, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
