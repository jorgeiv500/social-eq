"""C3 — Asignación «de la zona al sitio» (nivel 2), según preregistro.md y su Adenda 1.

Nivel 1: puertos por UTAM y sede de la asignación S2 publicada (sin cambios).
Nivel 2: dentro de cada UTAM, los puertos de trabajo y públicos (A4) se reparten entre los
parqueaderos candidatos elegibles en proporción a w_s = D_z(s) / (nº de sitios elegibles en la ZAT
z(s)), con D_z = nivel de llegadas de conductores de auto (demanda_c3.py, regla A5), respetando la
capacidad por sitio c_s = máx(⌊f·cupos_s⌋, 1 si cupos_s > 0) con reparto proporcional con tope y
redistribución del excedente (water-filling) dentro de la UTAM. Los puertos que no caben son la
brecha de la UTAM.

Escenarios (cada uno con candidatos «ambas», «OSM» y «Catastro», como fija el control de sesgo):
principal; capacidad 5 % y 20 %; imputación de cupos p25 y p75 (A1); + ZPP y OSM en vía;
residenciales del cuartil vulnerable en parqueaderos (A4); reubicación de la brecha a la UTAM
vecina (decisión 3); estimadores EB, GCN y ST-GNN (decisión 4).

No lanza NEOS ni toca resultados publicados. Uso: python asignar_c3.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
REV = AQUI.parent
RAIZ = REV.parents[1]
sys.path.insert(0, str(REV))
import comun  # noqa: E402
from social_eq.followup import normalize_zone_code, weighted_gini  # noqa: E402

SAL = AQUI / "salidas"
C2 = REV / "C2_parqueaderos_permanencias"
EM = Path(os.environ.get("EM2023_DIR", "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes"))
SHP_UTAM = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023" / "UTAM2023.shp"
CRS = 9377
EN_VIA_OSM = {"street_side", "on_kerb", "lane"}
ACCESO_EXCLUIDO = {"private", "no"}   # A2
SEDES_SITIO = ("workplace", "public")  # A4
FUENTES = ("ambas", "OSM", "Catastro")


# ---------------------------------------------------------------- insumos
def cargar():
    cand = gpd.read_file(C2 / "salidas" / "parqueaderos_candidatos.gpkg", layer="candidatos").to_crs(CRS)
    cand["ZAT"] = cand["ZAT"].astype(int)
    man = gpd.read_file(C2 / "datos" / "catastro_manzanas_depositos_parqueaderos_2026.geojson").to_crs(CRS)
    s2 = comun.leer_asignacion_paper("S2")
    s2["UTAM"] = s2["UTAM"].map(normalize_zone_code)
    b5 = pd.read_csv(REV / "B5_nucleo_robusto" / "salidas" / "b5_zonas_S2.csv")
    dem = pd.read_csv(os.environ.get("C3_DEMANDA_PRUEBA", SAL / "demanda_zat_estimadores.csv"))
    ut = gpd.read_file(SHP_UTAM).to_crs(CRS)
    ut["UTAM"] = ut["UTAM"].map(normalize_zone_code)
    ut = ut.dissolve(by="UTAM", as_index=False)[["UTAM", "MUNCodigo", "geometry"]]
    return cand, man, s2, b5, dem, ut


def zat_poblacion(ut, b5):
    """Centroides de las 940 ZAT de Bogotá y su población (DANE de la UTAM repartida por personas EM-2023)."""
    z = gpd.read_file(EM / "ZAT2023.shp").to_crs(CRS)
    z = z[z["MUNCod"] == 11001][["ZAT", "geometry"]].copy()
    z["ZAT"] = z["ZAT"].astype(int)
    z = z.dissolve(by="ZAT", as_index=False)
    z["area"] = z.geometry.area
    cz = gpd.GeoDataFrame(z[["ZAT", "area"]], geometry=z.geometry.representative_point(), crs=CRS)
    j = gpd.sjoin(cz, ut, predicate="within", how="left").drop(columns="index_right").drop_duplicates("ZAT")
    falt = j["UTAM"].isna()
    if falt.any():
        jn = gpd.sjoin_nearest(cz[cz["ZAT"].isin(j.loc[falt, "ZAT"])], ut, how="left").drop_duplicates("ZAT")
        j.loc[falt, "UTAM"] = j.loc[falt, "ZAT"].map(jn.set_index("ZAT")["UTAM"]).to_numpy()
    with open(REV / "B6_stgnn_honesto" / "salidas" / "cache_b6.pkl", "rb") as f:
        D = pickle.load(f)
    h = D["hh"]; cods = np.array(D["zat"]["cod"])
    m = h["zat"] >= 0
    pers = pd.Series(h["w"][m] * np.nan_to_num(h["pers"][m]), index=cods[h["zat"][m]]).groupby(level=0).sum()
    j["pers_em"] = j["ZAT"].map(pers).fillna(0.0)
    pob_u = b5.set_index("UTAM")["poblacion_dane"]
    j["pob_utam"] = j["UTAM"].map(pob_u).fillna(0.0)
    tot_p = j.groupby("UTAM")["pers_em"].transform("sum")
    tot_a = j.groupby("UTAM")["area"].transform("sum")
    j["poblacion"] = np.where(tot_p > 0, j["pob_utam"] * j["pers_em"] / tot_p.where(tot_p > 0, 1),
                              j["pob_utam"] * j["area"] / tot_a)
    j["cuartil"] = j["UTAM"].map(b5.set_index("UTAM")["cuartil_vulnerabilidad"])
    j["x"], j["y"] = j.geometry.x, j.geometry.y
    return pd.DataFrame(j.drop(columns="geometry"))


# ---------------------------------------------------------------- candidatos por escenario
def candidatos(cand, man, fuente="ambas", en_via=False):
    c = cand.copy()
    acc = c["acceso"].astype(str).str.lower()
    osm = c["fuente"].eq("OSM") & ~acc.isin(ACCESO_EXCLUIDO)
    osm_fuera = osm & ~c["tipo"].isin(EN_VIA_OSM)
    osm_via = osm & c["tipo"].isin(EN_VIA_OSM)
    cat = c["fuente"].eq("Catastro_UAECD_2026")
    zpp = c["fuente"].eq("ZPP_TerminalTransporte")
    if fuente in ("ambas", "Catastro"):
        if fuente == "ambas":   # A3: manzana con un punto OSM elegible cuenta una sola vez (como OSM)
            pts = c[osm_fuera | (osm_via if en_via else False)]
            dup = gpd.sjoin(pts[["geometry"]], man[["MANCODIGO", "geometry"]], predicate="within")
            cat = cat & ~c["id"].isin("cat_" + dup["MANCODIGO"].astype(str))
    sel = {"ambas": osm_fuera | cat, "OSM": osm_fuera, "Catastro": cat}[fuente]
    if en_via:
        sel = sel | zpp | (osm_via if fuente != "Catastro" else False)
    return c[sel].copy()


def capacidad(c, mediana_ref, f):
    cup = pd.to_numeric(c["capacidad"], errors="coerce")
    imput = cup.isna()
    cup = cup.fillna(mediana_ref)  # A1 (OSM y Catastro sin cupos); ZPP siempre trae cupos de auto
    cap = np.floor(f * cup + 1e-9)
    cap = np.where((cup > 0) & (cap < 1), 1, cap)
    return cup.to_numpy(), imput.to_numpy(), cap.astype(int)


# ---------------------------------------------------------------- reparto con tope (water-filling)
def waterfill(T: int, cap: np.ndarray, w: np.ndarray) -> np.ndarray:
    cap = np.asarray(cap, float); w = np.asarray(w, float)
    total = int(min(T, cap.sum()))
    x = np.zeros(len(cap))
    if total <= 0:
        return x.astype(int)
    for _ in range(len(cap) + 2):
        rem = total - x.sum()
        if rem <= 1e-9:
            break
        libre = x < cap - 1e-12
        act = libre & (w > 0)
        ww = w.copy()
        if not act.any():           # solo quedan sitios de ZAT sin demanda: reparto uniforme
            act = libre; ww = np.ones(len(cap))
        prop = rem * ww[act] / ww[act].sum()
        nuevo = x[act] + prop
        idx = np.where(act)[0]
        sobre = nuevo > cap[idx]
        if not sobre.any():
            x[idx] = nuevo; break
        x[idx[sobre]] = cap[idx[sobre]]
    base = np.floor(x + 1e-9)
    r = int(round(total - base.sum()))
    if r > 0:
        frac = np.where(base < cap, x - base, -1.0)
        base[np.argsort(-frac, kind="stable")[:r]] += 1
    return base.astype(int)


def repartir_sedes(tot_sitio: np.ndarray, por_sede: dict) -> dict:
    """Etiqueta de sede de los puertos de cada sitio (redondeo de mayor resto, filas y columnas exactas)."""
    I = tot_sitio.sum(); out = {}
    restante = tot_sitio.astype(float).copy()
    sedes = [s for s in por_sede if por_sede[s] > 0]
    for k, s in enumerate(sedes):
        if k == len(sedes) - 1:
            out[s] = restante.astype(int); break
        meta = tot_sitio * por_sede[s] / max(I, 1)
        b = np.minimum(np.floor(meta), restante)
        r = int(por_sede[s] - b.sum())
        if r > 0:
            frac = np.where(b < restante, meta - b, -1.0)
            b[np.argsort(-frac, kind="stable")[:r]] += 1
        out[s] = b.astype(int); restante -= b
    for s in por_sede:
        out.setdefault(s, np.zeros(len(tot_sitio), int))
    return out


def asignar(sit, s2, sedes_por_utam):
    """Nivel 2 en cada UTAM. sedes_por_utam: UTAM -> lista de sedes que van a sitio."""
    sit = sit.copy()
    for s in ("residential", *SEDES_SITIO):
        sit[f"puertos_{s}"] = 0
    filas = []
    g = sit.groupby("UTAM").indices
    for _, r in s2.iterrows():
        u = r["UTAM"]; sedes = sedes_por_utam(u)
        cuota = {s: int(r[f"ports_{s}"]) for s in sedes}
        T = sum(cuota.values())
        idx = g.get(u, np.array([], int))
        cap = sit["capacidad"].to_numpy()[idx]; w = sit["peso"].to_numpy()[idx]
        x = waterfill(T, cap, w) if len(idx) else np.zeros(0, int)
        I = int(x.sum())
        # puertos instalados por sede: mayor resto de I según la cuota de cada sede
        if T > 0:
            meta = {s: I * cuota[s] / T for s in sedes}
            inst = {s: int(np.floor(meta[s])) for s in sedes}
            for s in sorted(sedes, key=lambda s: -(meta[s] - inst[s]))[: I - sum(inst.values())]:
                inst[s] += 1
        else:
            inst = {s: 0 for s in sedes}
        if len(idx):
            lab = repartir_sedes(x, inst)
            for s in sedes:
                sit.iloc[idx, sit.columns.get_loc(f"puertos_{s}")] = lab[s]
        filas.append({"UTAM": u, "n_sitios": len(idx), "capacidad_utam": int(cap.sum()), "cuota_sitio": T,
                      "instalados": I, "brecha": T - I,
                      **{f"cuota_{s}": cuota.get(s, 0) for s in ("residential", *SEDES_SITIO)},
                      **{f"brecha_{s}": cuota.get(s, 0) - inst.get(s, 0) for s in ("residential", *SEDES_SITIO)}})
    sit["puertos"] = sit[[f"puertos_{s}" for s in ("residential", *SEDES_SITIO)]].sum(1)
    return sit, pd.DataFrame(filas)


# ---------------------------------------------------------------- reubicación de la brecha (decisión 3)
def contiguidad(ut):
    g = ut.copy(); g["geometry"] = g.buffer(1.0)
    j = gpd.sjoin(g, g, predicate="intersects")
    j = j[j["UTAM_left"] != j["UTAM_right"]]
    vec = j.groupby("UTAM_left")["UTAM_right"].apply(set).to_dict()
    cen = ut.set_index("UTAM").geometry.centroid
    return vec, cen


def reubicar(sit, bu, vec, cen, origenes=None):
    """Mueve la brecha de cada UTAM a la vecina contigua con capacidad libre más cercana (por anillos
    de contigüidad si las vecinas de primer orden se llenan). Procesa las UTAM de mayor a menor brecha."""
    sit = sit.copy(); bu = bu.copy()
    libre = (sit["capacidad"] - sit["puertos"]).to_numpy().astype(int)
    gi = sit.groupby("UTAM").indices
    mov = []
    for _, r in bu.sort_values("brecha", ascending=False).iterrows():
        u = r["UTAM"]
        if origenes is not None and u not in origenes:
            bu.loc[bu["UTAM"] == u, "sin_reubicar"] = int(r["brecha"])
            continue
        pend = {s: int(r[f"brecha_{s}"]) for s in ("residential", *SEDES_SITIO) if r[f"brecha_{s}"] > 0}
        if not pend:
            continue
        visto, frontera, anillo = {u}, {u}, 0
        while sum(pend.values()) > 0:
            nueva = set().union(*[vec.get(x, set()) for x in frontera]) - visto
            if not nueva:
                break
            anillo += 1; visto |= nueva; frontera = nueva
            for v in sorted(nueva, key=lambda v: cen[u].distance(cen[v])):
                idx = gi.get(v)
                if idx is None or libre[idx].sum() == 0:
                    continue
                for s in list(pend):
                    if pend[s] <= 0 or libre[idx].sum() == 0:
                        continue
                    x = waterfill(pend[s], libre[idx], sit["peso"].to_numpy()[idx])
                    k = int(x.sum())
                    if k == 0:
                        continue
                    col = sit.columns.get_loc(f"puertos_{s}")
                    sit.iloc[idx, col] = sit.iloc[idx, col].to_numpy() + x
                    libre[idx] -= x; pend[s] -= k
                    mov.append({"origen": u, "destino": v, "sede": s, "puertos": k, "anillo": anillo,
                                "dist_centroides_km": cen[u].distance(cen[v]) / 1000})
        bu.loc[bu["UTAM"] == u, "sin_reubicar"] = sum(pend.values())
    bu["sin_reubicar"] = bu.get("sin_reubicar", 0)
    bu["sin_reubicar"] = bu["sin_reubicar"].fillna(0).astype(int)
    sit["puertos"] = sit[[f"puertos_{s}" for s in ("residential", *SEDES_SITIO)]].sum(1)
    return sit, bu, pd.DataFrame(mov)


# ---------------------------------------------------------------- métricas
def frac_instalable(bu, cuart, bog=None, suf=""):
    b = bu.assign(cuartil=bu["UTAM"].map(cuart))
    if bog is not None:
        b = b[b["UTAM"].isin(bog)]
    out = {}
    for q in (1, 2, 3, 4):
        m = b["cuartil"] == q
        out[f"frac_Q{q}"] = b.loc[m, "instalados"].sum() / max(b.loc[m, "cuota_sitio"].sum(), 1)
    out["frac_resto"] = b.loc[b["cuartil"] != 4, "instalados"].sum() / max(b.loc[b["cuartil"] != 4, "cuota_sitio"].sum(), 1)
    out["frac_ciudad"] = b["instalados"].sum() / b["cuota_sitio"].sum()
    out["brecha_total"] = int(b["brecha"].sum()); out["cuota_total"] = int(b["cuota_sitio"].sum())
    out["utam_con_brecha"] = int((b["brecha"] > 0).sum())
    return {k + suf: v for k, v in out.items()}


def distancias(zp, xy_sitios, pref="dist"):
    z = zp[zp["poblacion"] > 0]
    d, _ = cKDTree(xy_sitios).query(np.c_[z["x"], z["y"]])
    out = {}
    for q in (1, 2, 3, 4):
        m = (z["cuartil"] == q).to_numpy()
        out[f"{pref}_Q{q}_m"] = float(np.average(d[m], weights=z["poblacion"].to_numpy()[m]))
    m = (z["cuartil"] == 4).to_numpy()
    out[f"{pref}_resto_m"] = float(np.average(d[~m], weights=z["poblacion"].to_numpy()[~m]))
    out[f"{pref}_ciudad_m"] = float(np.average(d, weights=z["poblacion"]))
    out[f"{pref}_brecha_Q4_menos_resto_m"] = out[f"{pref}_Q4_m"] - out[f"{pref}_resto_m"]
    out[f"{pref}_razon_Q4_resto"] = out[f"{pref}_Q4_m"] / out[f"{pref}_resto_m"]
    return out


def provision(n_utam: pd.DataFrame, s2, pobl):
    """Igual que scripts/intervention_size.py (puertos por 10.000 hab., zonas con población censal)."""
    d = s2[["UTAM", "vulnerability_index"]].merge(n_utam, on="UTAM").merge(pobl, on="UTAM", how="left")
    vq = d["vulnerability_index"] >= d["vulnerability_index"].quantile(0.75)
    con = d["dane_population"].fillna(0) > 0
    d, vq = d[con], vq[con]
    tv = 1e4 * d.loc[vq, "ports_total"].sum() / d.loc[vq, "dane_population"].sum()
    tr = 1e4 * d.loc[~vq, "ports_total"].sum() / d.loc[~vq, "dane_population"].sum()
    return tv, tr


# ---------------------------------------------------------------- principal
def main() -> int:
    SAL.mkdir(exist_ok=True)
    cand, man, s2, b5, dem, ut = cargar()
    cuart = b5.set_index("UTAM")["cuartil_vulnerabilidad"]
    vuln_utam = set(cuart[cuart == 4].index)
    zp = zat_poblacion(ut, b5)
    zp.to_csv(SAL / "zat_poblacion_cuartil.csv", index=False)
    pobl = pd.read_csv(RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    pobl["UTAM"] = pobl["UTAM"].map(normalize_zone_code); pobl = pobl[["UTAM", "dane_population"]]
    ctx = comun.preparar_contexto(SAL / "_ctx")
    orden = ctx.df_inputs["UTAM"].map(normalize_zone_code).tolist()
    nu = ctx.df_inputs["vulnerability_index"].to_numpy(float)
    vec, cen = contiguidad(ut)
    # la capa de candidatos de C2 solo cubre Bogotá (ZAT con MUNCod 11001): las 26 UTAM de municipios
    # vecinos no tienen candidatos por construcción; se reportan las fracciones con y sin ellas
    bog = set(ut.loc[ut["MUNCodigo"] == 11001, "UTAM"])

    def gini_de(n_utam):
        m = n_utam.set_index("UTAM").loc[orden][[f"ports_{s}" for s in comun.SEDES]].to_numpy()
        return weighted_gini(comun.calcular_accesibilidad(m, ctx.coeffs, ctx.config), 1.0 + nu)

    g_s2 = gini_de(s2); pv_s2, pr_s2 = provision(s2[["UTAM", "ports_total"]], s2, pobl)
    print(f"S2 publicado: Gini ponderado {g_s2:.6f} | provisión vulnerable {pv_s2:.2f} vs resto {pr_s2:.2f}")

    osm_fuera_cap = pd.to_numeric(cand.loc[cand["fuente"].eq("OSM") & ~cand["tipo"].isin(EN_VIA_OSM), "capacidad"], errors="coerce")
    osm_fuera_cap = osm_fuera_cap[osm_fuera_cap > 0]
    imp = {"mediana": float(osm_fuera_cap.median()), "p25": float(osm_fuera_cap.quantile(0.25)), "p75": float(osm_fuera_cap.quantile(0.75))}
    print(f"cupos imputados (OSM fuera de vía con cupos > 0, n={len(osm_fuera_cap)}): {imp}")

    estim = {"RF_rezago": "demanda_RF_rezago", "EB": "demanda_EB", "GCN": "demanda_GCN", "STGNN": "demanda_STGNN"}
    escenarios = [
        ("principal", dict()), ("capacidad_5", dict(f=0.05)), ("capacidad_20", dict(f=0.20)),
        ("imputacion_p25", dict(imp="p25")), ("imputacion_p75", dict(imp="p75")),
        ("con_en_via", dict(en_via=True)), ("A4_residencial_vulnerable", dict(res_vuln=True)),
        ("reubicacion", dict(reubicar=True)),
        # variante no preregistrada: la brecha de los municipios vecinos (sin capa de candidatos) no se mueve
        ("reubicacion_solo_bogota", dict(reubicar=True, solo_bog=True)),
        ("estimador_EB", dict(est="EB")), ("estimador_GCN", dict(est="GCN")), ("estimador_STGNN", dict(est="STGNN")),
    ]
    filas, sitios_largo, brechas_largo, movs = [], [], [], []
    base_sitios = {}
    for fuente in FUENTES:
        for nombre, p in escenarios:
            f = p.get("f", 0.10); est = p.get("est", "RF_rezago")
            c = candidatos(cand, man, fuente, en_via=p.get("en_via", False))
            cup, imput, cap = capacidad(c, imp[p.get("imp", "mediana")], f)
            c["cupos"] = cup; c["cupos_imputados"] = imput; c["capacidad"] = cap
            D = dem.set_index("ZAT")[estim[est]]
            c["demanda_zat"] = c["ZAT"].map(D).fillna(0.0)
            c["sitios_en_zat"] = c.groupby("ZAT")["id"].transform("count")
            c["peso"] = c["demanda_zat"] / c["sitios_en_zat"]
            c["x"], c["y"] = c.geometry.x, c.geometry.y
            c = pd.DataFrame(c.drop(columns="geometry")).reset_index(drop=True)
            if p.get("res_vuln"):
                sedes_fn = lambda u: (["residential", *SEDES_SITIO] if u in vuln_utam else list(SEDES_SITIO))
            else:
                sedes_fn = lambda u: list(SEDES_SITIO)
            sit, bu = asignar(c, s2, sedes_fn)
            fila = {"fuente": fuente, "escenario": nombre, "estimador": est, "fraccion_capacidad": f,
                    "cupos_imputados": imp[p.get("imp", "mediana")], "n_candidatos": len(c),
                    "capacidad_total": int(c["capacidad"].sum()), **frac_instalable(bu, cuart),
                    **frac_instalable(bu, cuart, bog, "_bogota")}
            if p.get("res_vuln"):
                bv = bu[bu["UTAM"].isin(vuln_utam)]
                fila["brecha_residencial_Q4"] = int(bv["brecha_residential"].sum())
                fila["cuota_residencial_Q4"] = int(bv["cuota_residential"].sum())
                fila["frac_residencial_Q4_instalable"] = 1 - bv["brecha_residential"].sum() / max(bv["cuota_residential"].sum(), 1)
            if p.get("reubicar"):
                sit, bu, mv = reubicar(sit, bu, vec, cen, bog if p.get("solo_bog") else None)
                mv["fuente"] = fuente; mv["escenario"] = nombre; movs.append(mv)
                rec = sit.groupby("UTAM")[[f"puertos_{s}" for s in SEDES_SITIO]].sum()
                n_utam = s2[["UTAM", "ports_residential"]].copy()
                sin = bu.set_index("UTAM")
                # sedes en sitio: lo instalado en la UTAM (propio + recibido) + lo que no se pudo reubicar
                # (se deja en su UTAM de origen; declarado en el README)
                for s in SEDES_SITIO:
                    rest = (sin[f"brecha_{s}"]).clip(lower=0)
                    n_utam[f"ports_{s}"] = n_utam["UTAM"].map(rec[f"puertos_{s}"]).fillna(0).astype(int)
                fila["puertos_reubicados"] = int(mv["puertos"].sum()) if len(mv) else 0
                fila["puertos_sin_reubicar"] = int(bu["sin_reubicar"].sum())
                if fila["puertos_sin_reubicar"] > 0:
                    # reparto de lo no reubicado a su UTAM de origen, por sede en proporción a la brecha
                    for u, rb in bu[bu["sin_reubicar"] > 0].set_index("UTAM").iterrows():
                        tot_b = sum(rb[f"brecha_{s}"] for s in SEDES_SITIO)
                        k = int(rb["sin_reubicar"]); asign = 0
                        for i, s in enumerate(SEDES_SITIO):
                            v = k - asign if i == len(SEDES_SITIO) - 1 else int(round(k * rb[f"brecha_{s}"] / max(tot_b, 1)))
                            n_utam.loc[n_utam["UTAM"] == u, f"ports_{s}"] += v; asign += v
                n_utam["ports_total"] = n_utam[[f"ports_{s}" for s in comun.SEDES]].sum(1)
                assert n_utam["ports_total"].sum() == 10870, n_utam["ports_total"].sum()
                g = gini_de(n_utam); pv, pr = provision(n_utam[["UTAM", "ports_total"]], s2, pobl)
                fila.update({"gini_ponderado": g, "delta_gini_ponderado": g - g_s2, "provision_Q4_por10k": pv,
                             "provision_resto_por10k": pr, "delta_provision_Q4": pv - pv_s2,
                             "utam_con_puertos_movidos": int(mv["origen"].nunique()) if len(mv) else 0,
                             "dist_media_reubicacion_km": float(np.average(mv["dist_centroides_km"], weights=mv["puertos"])) if len(mv) else 0.0,
                             "anillo_max": int(mv["anillo"].max()) if len(mv) else 0})
                n_utam.assign(fuente=fuente).to_csv(SAL / f"asignacion_utam_{nombre}_{fuente}.csv", index=False)
            con = sit[sit["puertos"] > 0]
            fila["sitios_con_puertos"] = len(con)
            fila["puertos_en_sitios"] = int(sit["puertos"].sum())
            fila.update(distancias(zp, con[["x", "y"]].to_numpy()) if len(con) else {})
            fila.update(distancias(zp, c[["x", "y"]].to_numpy(), pref="dist_cualquier_candidato"))
            if nombre == "principal":
                base_sitios[fuente] = sit.set_index("id")["puertos"]
            if nombre.startswith("estimador_"):
                b = base_sitios[fuente]; a = sit.set_index("id")["puertos"].reindex(b.index).fillna(0)
                fila["puertos_que_cambian_de_sitio"] = 0.5 * float(np.abs(a - b).sum())
                fila["pct_cambian_de_sitio_vs_RF"] = 100 * fila["puertos_que_cambian_de_sitio"] / b.sum()
            filas.append(fila)
            cols = ["id", "fuente", "tipo", "acceso", "ZAT", "UTAM", "cupos", "cupos_imputados", "capacidad", "demanda_zat",
                    "sitios_en_zat", "peso", "puertos", "puertos_workplace", "puertos_public", "puertos_residential", "x", "y"]
            sl = sit[cols].copy(); sl.insert(0, "escenario", nombre); sl.insert(0, "conjunto_fuentes", fuente)
            sitios_largo.append(sl[sl["puertos"] > 0] if nombre != "principal" else sl)
            bl = bu.copy(); bl.insert(0, "escenario", nombre); bl.insert(0, "conjunto_fuentes", fuente)
            bl["cuartil"] = bl["UTAM"].map(cuart); bl["en_bogota"] = bl["UTAM"].isin(bog); brechas_largo.append(bl)
            print(f"[{fuente:8s}] {nombre:28s} frac Q1..Q4 (Bogotá) = "
                  f"{fila['frac_Q1_bogota']:.3f} {fila['frac_Q2_bogota']:.3f} {fila['frac_Q3_bogota']:.3f} {fila['frac_Q4_bogota']:.3f} | "
                  f"dist Q4/resto = {fila.get('dist_Q4_m', np.nan):.0f}/{fila.get('dist_resto_m', np.nan):.0f} m", flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(SAL / "resultados_escenarios.csv", index=False)
    sl = pd.concat(sitios_largo)
    sl[(sl["escenario"] == "principal") & (sl["conjunto_fuentes"] == "ambas")].drop(columns=["escenario", "conjunto_fuentes"]) \
        .to_csv(SAL / "asignacion_por_sitio_principal.csv", index=False)
    sl.to_csv(SAL / "asignacion_por_sitio_todos_escenarios.csv.gz", index=False)
    pd.concat(brechas_largo).to_csv(SAL / "brecha_por_utam.csv", index=False)
    if movs:
        pd.concat(movs).to_csv(SAL / "reubicacion_movimientos.csv", index=False)
    json.dump({"gini_ponderado_S2_recalculado": g_s2, "provision_Q4_S2": pv_s2, "provision_resto_S2": pr_s2,
               "cupos_imputados": imp, "n_osm_fuera_con_cupos": int(len(osm_fuera_cap))},
              open(SAL / "referencias_c3.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
