"""B6 — Preparación de datos: zonas (UTAM, ZAT), viajes por extremo y covariables de hogares.

Reutiliza la lectura de la EM-2023 de `scripts/retrain_stgnn.py` (carpeta EM2023_DIR, separador
`;`, latin-1, canon de códigos UTAM, números con coma decimal, `fexp_vj`). Escribe una caché
`salidas/cache_b6.pkl` con arreglos numpy que leen los trabajadores de `run_b6.py`.
No usa nada de `src/` ni toca resultados publicados.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
from retrain_stgnn import canon, leer_modulo, parse_hour, to_num  # noqa: E402

DRIVE = Path(os.environ.get("EM2023_DIR", "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes"))
SHP_UTAM = RAIZ / "outputs" / "social_equity_followup_outputs" / "utam_shp" / "UTAM2023"
CACHE = AQUI / "salidas" / "cache_b6.pkl"


def zonas():
    u = gpd.read_file(SHP_UTAM).to_crs(3116)
    u["cod"] = u["UTAM"].astype(str).map(canon)
    u = u[u["cod"].notna()].reset_index(drop=True)
    locs = sorted(u["LOCNombre"].astype(str).unique())
    u["loc"] = u["LOCNombre"].astype(str).map({l: i for i, l in enumerate(locs)})
    cu = u.geometry.centroid
    z = gpd.read_file(DRIVE / "ZAT2023.shp").to_crs(3116)
    z = z[z["ZAT"].notna()].copy()
    z["cod"] = z["ZAT"].astype(float).astype(int)
    z = z.dissolve(by="cod", as_index=False)
    cz = gpd.GeoDataFrame({"cod": z["cod"]}, geometry=z.geometry.centroid, crs=3116)
    j = gpd.sjoin(cz, u[["cod", "loc", "geometry"]].rename(columns={"cod": "utam"}), predicate="within", how="inner")
    j = j.drop_duplicates("cod").sort_values("cod").reset_index(drop=True)
    utam = {"cod": u["cod"].tolist(), "xy": np.c_[cu.x, cu.y], "loc": u["loc"].to_numpy(), "locs": locs}
    zat = {"cod": j["cod"].tolist(), "xy": np.c_[j.geometry.x, j.geometry.y], "loc": j["loc"].to_numpy(),
           "utam_idx": j["utam"].map({c: i for i, c in enumerate(utam["cod"])}).to_numpy(), "locs": locs}
    return utam, zat


def main() -> int:
    utam, zat = zonas()
    print(f"UTAM: {len(utam['cod'])} | ZAT dentro de UTAM: {len(zat['cod'])} | localidades: {len(utam['locs'])}")
    hog = leer_modulo("a. Modulo hogares.csv")
    via = leer_modulo("d. Modulo viajes.csv")
    veh = leer_modulo("b. Modulo vehiculos.csv")
    hog = hog.drop_duplicates("key_hg").reset_index(drop=True)
    hh_id = {k: i for i, k in enumerate(hog["key_hg"])}
    H = len(hog)
    u_pos = {c: i for i, c in enumerate(utam["cod"])}
    z_pos = {c: i for i, c in enumerate(zat["cod"])}

    def zat_idx(s):
        x = pd.to_numeric(s, errors="coerce")
        return np.array([z_pos.get(int(v), -1) if np.isfinite(v) else -1 for v in x])

    # ---------------- hogares
    fecha = pd.to_datetime(hog["fecha"], dayfirst=True, errors="coerce")
    tipo_viv = hog["tipo_viv"].astype(str).str.lower()
    veh["n"] = to_num(veh["cantidad"]).fillna(1.0)
    autos = veh[veh["tipo_vehiculo"].astype(str).str.contains("autom|camioneta|campero", case=False, na=False)]
    autos_hh = autos.groupby("key_hg")["n"].sum()
    hh = {
        "w": to_num(hog["fexp_hg"]).fillna(0.0).to_numpy(),
        "utam": hog["cod_utam_hg"].map(canon).map(u_pos).fillna(-1).astype(int).to_numpy(),
        "zat": zat_idx(hog["zat_hg"]),
        "estrato": to_num(hog["estrato_hg"]).to_numpy(),
        "apto": tipo_viv.str.contains("apart", na=False).to_numpy().astype(float),
        "pers": to_num(hog["perstotal_hg"]).to_numpy(),
        "autos": hog["key_hg"].map(autos_hh).fillna(0.0).to_numpy(),
        "mes": fecha.dt.month.fillna(0).astype(int).to_numpy(),
        "finde": (fecha.dt.dayofweek >= 5).to_numpy(),
    }
    print(f"hogares: {H} | fecha: {fecha.min()} -> {fecha.max()} | por mes: {pd.Series(hh['mes']).value_counts().sort_index().to_dict()}")

    # ---------------- viajes (un registro por viaje; el perfil usa ambos extremos)
    via = via[via["key_hg"].isin(hh_id)].copy()
    hi = parse_hour(via[[c for c in via.columns if c.strip("_") == "hora_ini"][0]])
    hf = parse_hour(via["hora_fin"])
    trips = {
        "hh": via["key_hg"].map(hh_id).to_numpy(),
        "w": to_num(via["fexp_vj"]).fillna(0.0).to_numpy(),
        "auto": (via["modo_principal_agrupado"].astype(str).str.upper().str.strip() == "AUTO").to_numpy(),
        "h_sal": hi.fillna(-1).astype(int).to_numpy(), "h_lle": hf.fillna(-1).astype(int).to_numpy(),
        "u_ori": via["utam_ori"].map(canon).map(u_pos).fillna(-1).astype(int).to_numpy(),
        "u_des": via["utam_des"].map(canon).map(u_pos).fillna(-1).astype(int).to_numpy(),
        "z_ori": zat_idx(via["zat_ori"]), "z_des": zat_idx(via["zat_des"]),
    }
    for k in ("h_sal", "h_lle"):
        trips[k] = np.where((trips[k] >= 0) & (trips[k] <= 23), trips[k], -1)
    print(f"viajes: {len(via)} | AUTO: {trips['auto'].sum()} | con hora fin válida: {(trips['h_lle'] >= 0).mean():.3f}")
    print(f"extremos en zona UTAM: ori {(trips['u_ori'] >= 0).mean():.3f} des {(trips['u_des'] >= 0).mean():.3f} | "
          f"ZAT: ori {(trips['z_ori'] >= 0).mean():.3f} des {(trips['z_des'] >= 0).mean():.3f}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"utam": utam, "zat": zat, "hh": hh, "trips": trips, "H": H}, f)
    print(f"caché: {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
