#!/usr/bin/env python
"""Reconstrucción de insumos y reentrenamiento del ST-GNN con el cruce de códigos corregido.

Réplica local de la celda 18 de `modelado/bogota_ev_charging_full_experiment_colab_enriched_utam.ipynb`
(misma arquitectura HybridSTGNN, mismo protocolo: semilla 42, particiones 70/15/15 con la misma
permutación, Adam 1e-3, wd 1e-4, L1, 1200 épocas, early stopping 120, contigüidad con respaldo k-NN 6,
semillas de estabilidad 42/52/62) con TRES correcciones de datos, todas documentadas en
`../README.md`:

  1. códigos de zona canónicos en ambos lados (`UTAM043` -> `UTAM43`, `UPR0002` -> `UPR2`);
  2. pesos con coma decimal leídos como número (`fexp_per>5años`, `fexp_eta`, `fexp_hg`);
  3. viajes ponderados por su factor de expansión `fexp_vj` (el notebook usó peso 1 al no hallarlo).

Todo se escribe en `salidas/`; no toca `data/`, `outputs/` ni `latex/`. Uso:
    python reentrenar_stgnn.py            # completo (principal + 3 semillas + baselines)
    python reentrenar_stgnn.py --rapido   # solo la corrida principal
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
import os
DRIVE = Path(os.environ.get("EM2023_DIR", "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes"))
SHAPEFILE = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"
SALIDAS = RAIZ / "outputs" / "stgnn_retraining"
PUBLICADO = RAIZ / "data" / "inputs" / "priority_utam_scores.csv"

HOURS = list(range(24))
KNN_K, RANDOM_SEED, EPOCHS, LR, WD = 6, 42, 1200, 1e-3, 1e-4
DYN_G_HID = DYN_T_HID = STATIC_HID = FUSION_HID = 32
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15
EARLY_STOPPING_PATIENCE, MIN_DELTA = 120, 1e-5
STABILITY_SEEDS, STABILITY_EPOCHS, STABILITY_PATIENCE = [42, 52, 62], 400, 80


# ------------------------------------------------------------------ helpers (del notebook)
def norm_text(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii").lower().strip()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


def canon(v) -> str | None:
    """Corrección 1: código canónico sin ceros a la izquierda."""
    s = str(v).strip().upper()
    m = re.match(r"^(UTAM|UPR)0*(\d+)$", s)
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def to_num(s: pd.Series) -> pd.Series:
    """Corrección 2: números con coma decimal."""
    direct = pd.to_numeric(s, errors="coerce")
    if direct.notna().mean() >= 0.8:
        return direct
    txt = s.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    txt = txt.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan}), errors="coerce")


def leer_modulo(nombre: str) -> pd.DataFrame:
    df = pd.read_csv(DRIVE / nombre, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [norm_text(c) for c in df.columns]
    return df


def find_col(df: pd.DataFrame, aliases: list[str], required: bool = False) -> str | None:
    cols = list(df.columns)
    for a in aliases:
        if norm_text(a) in cols:
            return norm_text(a)
    for c in cols:
        for a in aliases:
            aa = norm_text(a)
            if aa in c or c in aa:
                return c
    if required:
        raise KeyError(f"No encontré columna para {aliases}")
    return None


def weighted_mean(x, w) -> float:
    x = pd.to_numeric(x, errors="coerce"); w = pd.to_numeric(w, errors="coerce")
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.sum() and float(w[m].sum()) else np.nan


def weighted_share(mask, w) -> float:
    m = pd.Series(mask).fillna(False).astype(bool); ww = pd.to_numeric(w, errors="coerce").fillna(0.0)
    den = float(ww.sum())
    return float(ww[m].sum() / den) if den else np.nan


def parse_hour(s: pd.Series) -> pd.Series:
    num = to_num(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    frac = num.between(0, 1, inclusive="left"); out[frac] = np.floor(num[frac] * 24)
    rest = out.isna() & num.between(0, 23.999); out[rest] = np.floor(num[rest])
    return out


def agrupar(df: pd.DataFrame, calc) -> pd.DataFrame:
    filas = []
    for z, g in df.groupby("utam"):
        fila = calc(g); fila["utam"] = z; filas.append(fila)
    return pd.DataFrame(filas)


def spear(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(spearmanr(a, b, nan_policy="omit").statistic) if np.nanstd(a) > 0 and np.nanstd(b) > 0 else np.nan


def topk_overlap(a, b, frac=0.10) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float); k = max(1, int(np.ceil(frac * len(a))))
    return float(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)


def metricas(y_true, y_pred, split, model) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "model": model, "split": split, "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.std(y_true) > 0 else np.nan,
        "medae": float(median_absolute_error(y_true, y_pred)), "spearman": spear(y_true, y_pred),
        "kendall": float(kendalltau(y_true, y_pred).statistic) if np.std(y_true) > 0 else np.nan,
        "top10_overlap": topk_overlap(y_true, y_pred, 0.10), "top20_overlap": topk_overlap(y_true, y_pred, 0.20),
    }


# ------------------------------------------------------------------ features (portadas del notebook)
def build_hogares(h: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    w = to_num(h["fexp_hg"]).fillna(0.0)
    est = to_num(h["estrato_hg"]); viv = h["tipo_viv"].astype(str)
    df = h.assign(_w=w, _est=est, _viv=viv)
    out = agrupar(df, lambda g: {
        "hh_total_w": float(g["_w"].sum()),
        "avg_stratum": weighted_mean(g["_est"], g["_w"]),
        "low_stratum_share": weighted_share(g["_est"].isin([1, 2]), g["_w"]),
        "apartment_share": weighted_share(g["_viv"].str.contains("apart", case=False, na=False), g["_w"]),
        "house_share": weighted_share(g["_viv"].str.contains("casa", case=False, na=False), g["_w"]),
    })
    if "key_hg" in p.columns:
        ppl = p.groupby("key_hg").agg(utam=("utam", "first"), n=("utam", "size"))
        out = out.merge(ppl.groupby("utam")["n"].mean().rename("household_size_avg").reset_index(), on="utam", how="left")
    age = to_num(p["edad"]); pw = to_num(p[find_col(p, ["fexp_per_5anos", "fexp_per"], required=True)]).fillna(0.0)
    demo = agrupar(p.assign(_c=age.between(0, 14), _o=age >= 60, _w=pw), lambda g: {
        "children_in_hh_proxy_share": weighted_share(g["_c"], g["_w"]),
        "older_adult_in_hh_proxy_share": weighted_share(g["_o"], g["_w"]),
    })
    return out.merge(demo, on="utam", how="left")


def build_personas(p: pd.DataFrame) -> pd.DataFrame:
    pw = to_num(p[find_col(p, ["fexp_per_5anos", "fexp_per"], required=True)]).fillna(0.0)
    age = to_num(p["edad"])
    low = lambda c: p[c].astype(str).str.lower() if c in p.columns else pd.Series("", index=p.index)
    sex, edu, occ = low("sexo"), low("max_nivel_edu"), low("ocupacion_principal")
    dis, lic, cell = low("condicion_discapacidad"), low("licencia_cond_vigente"), low("posee_celular")
    mov, madre = low("realiza_desplazamientos"), low("madre_cab_familia")
    care1 = low(find_col(p, ["quien_lleva_recoge_establecimiento"]) or "")
    care2, diff = low("cuidado_despues_regresar"), low("dific_princ_medios_transp_discap")
    d = p.assign(
        _w=pw, is_woman=sex.str.contains("mujer|femen", na=False),
        a0=age.between(0, 14), a15=age.between(15, 29), a30=age.between(30, 59), a60=age >= 60,
        disab=~dis.str.contains("ninguna|no aplica|sin informacion|ns/nr", na=False),
        hedu=edu.str.contains("univers|tecnico|tecnolog|posgrad", na=False),
        emp=occ.str.contains("empleado|trabajador independiente|vendedor informal|empleado domestico|ocupad", na=False),
        inf=occ.str.contains("independiente|informal|vendedor informal|domestico", na=False),
        lic=~lic.str.contains("no tiene|no aplica|ns/nr|sin informacion", na=False),
        cel=cell.str.contains("voz y datos|datos", na=False), mov=mov.str.contains(r"\bsi\b", na=False),
        mh=madre.str.contains(r"\bsi\b", na=False),
        care=(~care1.str.contains("no aplica|nadie|sin informacion|ns/nr", na=False)) | (~care2.str.contains("no aplica|nadie|sin informacion|ns/nr", na=False)),
        mdiff=~diff.str.contains("no aplica|ninguna|sin informacion|ns/nr", na=False),
    )
    return agrupar(d, lambda g: {
        "pop_total_w": float(g["_w"].sum()), "women_share": weighted_share(g["is_woman"], g["_w"]),
        "age_0_14_share": weighted_share(g["a0"], g["_w"]), "age_15_29_share": weighted_share(g["a15"], g["_w"]),
        "age_30_59_share": weighted_share(g["a30"], g["_w"]), "age_60plus_share": weighted_share(g["a60"], g["_w"]),
        "disability_share": weighted_share(g["disab"], g["_w"]), "higher_edu_share": weighted_share(g["hedu"], g["_w"]),
        "employed_share": weighted_share(g["emp"], g["_w"]), "informal_or_independent_share": weighted_share(g["inf"], g["_w"]),
        "license_share": weighted_share(g["lic"], g["_w"]), "cellphone_data_share": weighted_share(g["cel"], g["_w"]),
        "moves_share": weighted_share(g["mov"], g["_w"]), "mother_head_share": weighted_share(g["mh"], g["_w"]),
        "caregiver_proxy_share": weighted_share(g["care"], g["_w"]), "mobility_difficulty_share": weighted_share(g["mdiff"], g["_w"]),
    })


def build_viajes(v: pd.DataFrame) -> pd.DataFrame:
    w = to_num(v["fexp_vj"]).fillna(0.0)  # corrección 3
    mode = v["modo_principal_agrupado"].astype(str).str.lower(); purp = v["motivo_viaje"].astype(str).str.lower()
    hour = parse_hour(v["hora_ini"]); dur = to_num(v["duracion_min"])
    d = v.assign(
        _w=w, work=purp.str.contains("trab", na=False), study=purp.str.contains("estud", na=False),
        care=purp.str.contains("cuidado|llevar|recoger", na=False), shop=purp.str.contains("compr", na=False),
        health=purp.str.contains("salud|medic", na=False),
        pt=mode.str.contains("transmilenio|sitp|bus|buseta|microbus|public", na=False), walk=mode.str.contains("a pie|peat", na=False),
        bike=mode.str.contains("bici", na=False), car=mode.str.contains("vehiculo particular|auto|carro", na=False),
        moto=mode.str.contains("motocic", na=False), pam=hour.between(6, 8), ppm=hour.between(17, 19), _dur=dur,
    )
    return agrupar(d, lambda g: {
        "trips_total_w": float(g["_w"].sum()), "work_trip_share": weighted_share(g["work"], g["_w"]),
        "study_trip_share": weighted_share(g["study"], g["_w"]), "care_trip_share": weighted_share(g["care"], g["_w"]),
        "shopping_trip_share": weighted_share(g["shop"], g["_w"]), "health_trip_share": weighted_share(g["health"], g["_w"]),
        "pt_trip_share": weighted_share(g["pt"], g["_w"]), "walk_trip_share": weighted_share(g["walk"], g["_w"]),
        "bike_trip_share": weighted_share(g["bike"], g["_w"]), "car_trip_share": weighted_share(g["car"], g["_w"]),
        "moto_trip_share": weighted_share(g["moto"], g["_w"]), "peak_am_share": weighted_share(g["pam"], g["_w"]),
        "peak_pm_share": weighted_share(g["ppm"], g["_w"]), "mean_trip_duration_min": weighted_mean(g["_dur"], g["_w"]),
    })


def build_etapas(e: pd.DataFrame) -> pd.DataFrame:
    w = to_num(e["fexp_eta"]).fillna(0.0); mode = e["modo_etapa"].astype(str).str.lower()
    wait = to_num(e["t_espera_min"])
    d = e.assign(_w=w, pt=mode.str.contains("transmilenio|sitp|bus|buseta|microbus|public", na=False),
                 walk=mode.str.contains("a pie|peat", na=False), bike=mode.str.contains("bici", na=False), _wait=wait)
    out = agrupar(d, lambda g: {
        "stages_total_w": float(g["_w"].sum()), "mean_wait_min": weighted_mean(g["_wait"], g["_w"]),
        "pt_stage_share": weighted_share(g["pt"], g["_w"]), "walk_stage_share": weighted_share(g["walk"], g["_w"]),
        "bike_stage_share": weighted_share(g["bike"], g["_w"]),
    })
    if "key_viaje" in e.columns:
        n = e.groupby(["utam", "key_viaje"]).size().rename("n").reset_index()
        extra = n.groupby("utam").agg(avg_stages_per_trip=("n", "mean"), multimodal_trip_share=("n", lambda x: float((x > 1).mean()))).reset_index()
        out = out.merge(extra, on="utam", how="left")
    return out


# ------------------------------------------------------------------ modelo (idéntico al notebook)
class HybridSTGNN(nn.Module):
    def __init__(self, in_dyn=3, in_static=8, g_hid=32, t_hid=32, static_hid=32, fusion_hid=32, att_hid=16):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn1, self.gcn2 = GCNConv(in_dyn, g_hid), GCNConv(g_hid, g_hid)
        self.lstm = nn.LSTM(input_size=g_hid, hidden_size=t_hid, num_layers=1, batch_first=True)
        self.att_w, self.att_u = nn.Linear(t_hid, att_hid), nn.Linear(att_hid, 1, bias=False)
        self.static_mlp = nn.Sequential(nn.Linear(in_static, static_hid), nn.ReLU(), nn.Linear(static_hid, static_hid), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(t_hid + static_hid, fusion_hid), nn.ReLU(), nn.Linear(fusion_hid, 1))

    def forward(self, X_seq, X_static, edge_index):
        g_list = []
        for h in range(X_seq.shape[1]):
            xh = self.gcn2(self.gcn1(X_seq[:, h, :], edge_index).relu(), edge_index).relu()
            g_list.append(xh.unsqueeze(1))
        Y, _ = self.lstm(torch.cat(g_list, dim=1))
        alpha = torch.softmax(self.att_u(torch.tanh(self.att_w(Y))).squeeze(-1), dim=1)
        z = torch.cat([(Y * alpha.unsqueeze(-1)).sum(dim=1), self.static_mlp(X_static)], dim=1)
        return self.out(z).squeeze(-1), alpha


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--peso-viajes", choices=["uno", "fexp_miles", "fexp"], default="uno",
                    help="peso de cada viaje en el tensor: 1 (como el notebook de facto), fexp_vj/1000 (miles de viajes expandidos) o fexp_vj")
    args = ap.parse_args()
    global SALIDAS
    SALIDAS = AQUI / "salidas" / f"peso_{args.peso_viajes}"
    SALIDAS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    print("Leyendo módulos de la EM-2023...", flush=True)
    hog, per, via, eta = (leer_modulo(f) for f in ("a. Modulo hogares.csv", "c. Modulo personas.csv", "d. Modulo viajes.csv", "e. Modulo etapas.csv"))
    for df in (hog, per, eta):
        df["utam"] = df["cod_utam_hg"].map(canon)
    via["utam"] = via["cod_utam_hg"].map(canon)
    utam = gpd.read_file(SHAPEFILE)
    utam["utam"] = utam["UTAM"].astype(str).map(canon)
    zone_ids = utam["utam"].tolist(); id2pos = {z: i for i, z in enumerate(zone_ids)}; N = len(zone_ids)
    cent = gpd.GeoSeries(utam.to_crs(3857).geometry.centroid, crs=3857).to_crs(4326)
    utam["lon"], utam["lat"] = cent.x, cent.y

    print("Construyendo features estáticas...", flush=True)
    feats = build_hogares(hog, per)
    for extra in (build_personas(per), build_viajes(via), build_etapas(eta)):
        feats = feats.merge(extra, on="utam", how="outer")
    feats["trips_per_person"] = feats["trips_total_w"] / feats["pop_total_w"].replace(0, np.nan)
    vuln_cols = [c for c in ["low_stratum_share", "female_head_hh_share", "age_60plus_share", "disability_share", "mother_head_share", "mobility_difficulty_share", "mean_wait_min"] if c in feats.columns]
    tmp = feats[vuln_cols].apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0)
    feats["vulnerability_index_v1"] = tmp.mean(axis=1, skipna=True)

    print("Construyendo tensor horario (códigos canónicos, fexp_vj)...", flush=True)
    v = via.copy(); v["o"] = v["utam_ori"].map(canon); v["d"] = v["utam_des"].map(canon)
    v["hour"] = parse_hour(v["hora_ini"])
    if args.peso_viajes == "uno":
        v["w"] = 1.0
    elif args.peso_viajes == "fexp_miles":
        v["w"] = to_num(v["fexp_vj"]).fillna(0.0) / 1000.0
    else:
        v["w"] = to_num(v["fexp_vj"]).fillna(0.0)
    v = v[v["hour"].notna() & v["hour"].between(0, 23)].copy(); v["hour"] = v["hour"].astype(int)
    X_dyn = np.zeros((N, 24, 3), dtype=np.float32)
    for col, k in (("o", 0), ("d", 1)):
        for (z, h), w in v.groupby([col, "hour"])["w"].sum().items():
            i = id2pos.get(z)
            if i is not None:
                X_dyn[i, h, k] = float(w)
    X_dyn[:, :, 2] = X_dyn[:, :, 0] + X_dyn[:, :, 1]
    y = X_dyn[:, :, 2].mean(axis=1).astype(np.float32)
    print(f"  zonas con y > 0: {int((y > 0).sum())} de {N} | y media {y.mean():.1f} | max {y.max():.1f}")

    hourly = pd.DataFrame({"utam": zone_ids, **{f"trip_hour_{h:02d}": X_dyn[:, h, 2] for h in range(24)}})
    feats = pd.DataFrame({"utam": zone_ids}).merge(feats, on="utam", how="left").merge(hourly, on="utam", how="left")
    num_cols = [c for c in feats.select_dtypes(include=[np.number]).columns if not c.startswith("trip_hour_")]
    n_imputadas = {c: int(feats[c].isna().sum()) for c in num_cols if feats[c].isna().any()}
    for c in num_cols:
        med = feats[c].median(); feats[c] = feats[c].fillna(0.0 if pd.isna(med) else med)
    static_cols = [c for c in num_cols if feats[c].nunique(dropna=False) > 1]
    X_static = StandardScaler().fit_transform(feats[static_cols].values).astype(np.float32)
    print(f"  features estáticas válidas: {len(static_cols)} | celdas imputadas por mediana: {sum(n_imputadas.values())}")

    # grafo: contigüidad con respaldo k-NN (igual que el notebook)
    edges = set(); sidx = utam.sindex
    for i, geom in enumerate(utam.geometry):
        for j in sidx.query(geom, predicate="intersects"):
            if i < j and (geom.touches(utam.geometry.iloc[j]) or geom.intersects(utam.geometry.iloc[j])):
                edges.add((i, j)); edges.add((j, i))
    tipo_grafo = "contiguidad"
    if len(edges) < 2 * N:
        coords = np.c_[utam["lon"].values, utam["lat"].values]
        _, idx = NearestNeighbors(n_neighbors=min(KNN_K + 1, N)).fit(coords).kneighbors(coords)
        edges = {(i, j) for i, nb in enumerate(idx) for j in nb[1:]} | {(j, i) for i, nb in enumerate(idx) for j in nb[1:]}
        tipo_grafo = f"knn{KNN_K}"
    edge_index = np.array(sorted(edges), dtype=np.int64).T
    print(f"  grafo: {tipo_grafo}, {edge_index.shape[1]} aristas dirigidas")

    device = torch.device("cpu")
    Xd, Xs, yt, ei = (torch.tensor(a, dtype=torch.float32) for a in (X_dyn, X_static, y, edge_index)) if False else (
        torch.tensor(X_dyn, dtype=torch.float32), torch.tensor(X_static, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long))
    rng = np.random.default_rng(RANDOM_SEED); idx_all = rng.permutation(N)
    n_train, n_val = max(1, int(TRAIN_SPLIT * N)), max(1, int(VAL_SPLIT * N))
    train_idx, val_idx, test_idx = np.sort(idx_all[:n_train]), np.sort(idx_all[n_train:n_train + n_val]), np.sort(idx_all[n_train + n_val:])
    masks = {}
    for nombre, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        m = torch.zeros(N, dtype=torch.bool); m[ix] = True; masks[nombre] = m

    def entrenar(seed, max_epochs, patience, verbose):
        np.random.seed(seed); torch.manual_seed(seed)
        model = HybridSTGNN(in_dyn=3, in_static=X_static.shape[1], g_hid=DYN_G_HID, t_hid=DYN_T_HID, static_hid=STATIC_HID, fusion_hid=FUSION_HID)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD); loss_fn = nn.L1Loss()
        hist, best_val, best_epoch, best_state, wait = [], np.inf, 0, None, 0
        for ep in range(max_epochs):
            model.train(); opt.zero_grad()
            pred, _ = model(Xd, Xs, ei); loss = loss_fn(pred[masks["train"]], yt[masks["train"]]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pe, _ = model(Xd, Xs, ei); val_loss = loss_fn(pe[masks["val"]], yt[masks["val"]]).item()
            hist.append((ep + 1, float(loss.item()), float(val_loss)))
            if (best_val - val_loss) > MIN_DELTA:
                best_val, best_epoch, wait = val_loss, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if verbose and ep % 100 == 0:
                print(f"    [seed={seed}] época {ep:04d} | MAE train {loss.item():.2f} | MAE val {val_loss:.2f}", flush=True)
            if wait >= patience:
                if verbose: print(f"    [seed={seed}] early stopping en {ep + 1} | mejor época {best_epoch} | mejor val {best_val:.2f}")
                break
        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred, alpha = model(Xd, Xs, ei)
        return pred.numpy(), alpha.numpy(), pd.DataFrame(hist, columns=["epoch", "loss_mae_train", "loss_mae_val"]), float(best_val), int(best_epoch)

    print("Entrenando ST-GNN (corrida principal)...", flush=True)
    score, alpha, loss_df, best_val, best_epoch = entrenar(RANDOM_SEED, EPOCHS, EARLY_STOPPING_PATIENCE, True)
    met = pd.DataFrame([metricas(y[ix], score[ix], s, "hybrid_stgnn") for s, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx))] + [metricas(y, score, "all", "hybrid_stgnn")])
    print(met[["split", "n", "mae", "r2", "spearman"]].to_string(index=False))

    base_df = pd.DataFrame(); seed_df = pd.DataFrame(); stab_df = pd.DataFrame()
    if not args.rapido:
        dyn_summary = np.column_stack([X_dyn[:, :, k].mean(axis=1) for k in range(3)] + [X_dyn[:, :, 2].std(axis=1), X_dyn[:, :, 2].max(axis=1)]).astype(np.float32)
        filas = []
        for bname, Xb in {"baseline_static_linear": X_static, "baseline_dynamic_linear": dyn_summary, "baseline_combined_linear": np.concatenate([dyn_summary, X_static], axis=1)}.items():
            pb = LinearRegression().fit(Xb[train_idx], y[train_idx]).predict(Xb)
            filas += [metricas(y[ix], pb[ix], s, bname) for s, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx))] + [metricas(y, pb, "all", bname)]
        base_df = pd.DataFrame(filas)
        print("Estabilidad entre semillas...", flush=True)
        preds = {}; filas = []
        for sd in STABILITY_SEEDS:
            p_sd, _, _, bv, be = entrenar(sd, STABILITY_EPOCHS, STABILITY_PATIENCE, False); preds[sd] = p_sd
            for s, ix in (("all", np.arange(N)), ("test", test_idx)):
                r = metricas(y[ix], p_sd[ix], s, f"hybrid_stgnn_seed_{sd}"); r.update(seed=sd, best_epoch=be, best_val=bv); filas.append(r)
        seed_df = pd.DataFrame(filas)
        stab_df = pd.DataFrame([{"seed_a": a, "seed_b": b, "spearman_priority": spear(preds[a], preds[b]), "top10_overlap": topk_overlap(preds[a], preds[b], 0.10), "top20_overlap": topk_overlap(preds[a], preds[b], 0.20)} for i, a in enumerate(STABILITY_SEEDS) for b in STABILITY_SEEDS[i + 1:]])

    # exports en el formato que consume src/social_eq/followup.py
    smin, smax = float(score.min()), float(score.max())
    out = pd.DataFrame({"UTAM": zone_ids, "lon": utam["lon"].values, "lat": utam["lat"].values, "priority_score": score, "priority_norm": (score - smin) / (smax - smin + 1e-8)})
    out = out.merge(feats, left_on="UTAM", right_on="utam", how="left")
    out.to_csv(SALIDAS / "priority_utam_scores.csv", index=False)
    att = pd.DataFrame(alpha, columns=[f"hour_{h}" for h in HOURS]); att.insert(0, "UTAM", zone_ids)
    att.to_csv(SALIDAS / "temporal_attention_by_utam.csv", index=False)
    loss_df.to_csv(SALIDAS / "learning_loss_history.csv", index=False)
    met.to_csv(SALIDAS / "model_metrics_split.csv", index=False)
    feats.to_csv(SALIDAS / "features_utam_full.csv", index=False)
    for df, nombre in ((base_df, "baseline_metrics.csv"), (seed_df, "model_metrics_by_seed.csv"), (stab_df, "priority_stability_across_seeds.csv")):
        if len(df): df.to_csv(SALIDAS / nombre, index=False)

    # comparación con la prioridad publicada
    pub = pd.read_csv(PUBLICADO); pub["UTAM"] = pub["UTAM"].astype(str).str.upper()
    cmpd = out[["UTAM", "priority_score", "priority_norm"]].merge(pub[["UTAM", "priority_score", "priority_norm"]], on="UTAM", suffixes=("_nuevo", "_publicado"))
    comparacion = {
        "spearman_prioridad_nueva_vs_publicada": spear(cmpd["priority_score_nuevo"], cmpd["priority_score_publicado"]),
        "spearman_prioridad_nueva_vs_y_corregida": spear(score, y),
        "spearman_prioridad_publicada_vs_y_corregida": spear(cmpd["priority_score_publicado"], y),
        "top20_overlap_nueva_vs_publicada": topk_overlap(cmpd["priority_score_nuevo"], cmpd["priority_score_publicado"], 0.20),
        "zonas_prioridad_norm_mayor_0_05_nueva": int((cmpd["priority_norm_nuevo"] > 0.05).sum()),
        "zonas_prioridad_norm_mayor_0_05_publicada": int((cmpd["priority_norm_publicado"] > 0.05).sum()),
    }
    protocolo = {
        "random_seed": RANDOM_SEED, "epochs_max": EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE, "min_delta": MIN_DELTA,
        "learning_rate": LR, "weight_decay": WD, "knn_k": KNN_K, "grafo": tipo_grafo, "aristas_dirigidas": int(edge_index.shape[1]),
        "train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT, "test_split": TEST_SPLIT, "train_n": int(len(train_idx)), "val_n": int(len(val_idx)), "test_n": int(len(test_idx)),
        "stability_seeds": STABILITY_SEEDS, "stability_epochs": STABILITY_EPOCHS, "n_utam": N, "best_epoch": best_epoch, "best_val_mae": best_val,
        "zonas_y_positiva": int((y > 0).sum()), "n_features_estaticas": len(static_cols), "celdas_imputadas_mediana": n_imputadas,
        "correcciones": ["codigos canonicos (UTAM043->UTAM43, UPR0002->UPR2) en los cuatro modulos y en origen/destino de viajes",
                         "pesos con coma decimal parseados como numero (fexp_hg, fexp_per>5anos, fexp_eta)",
                         f"peso de cada viaje en el tensor: {args.peso_viajes} (el notebook uso peso 1 al no encontrar fexp_vj)"],
        "comparacion_con_publicado": comparacion,
    }
    (SALIDAS / "stgnn_run_protocol.json").write_text(json.dumps(protocolo, indent=2, default=float), encoding="utf-8")
    print("\n=== Comparación con la prioridad publicada ===")
    for k, v in comparacion.items(): print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSalidas en {SALIDAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
