#!/usr/bin/env python
"""Parte 2 — Llegadas y permanencias de conductores de auto por ZAT y hora (EM-2023).

Definiciones (todas declaradas en el README):

* Viaje de conductor de auto: `modo_principal_desagrupado` == "Vehículo privado como conductor".
  La EM-2023 separa conductor y pasajero en el modo principal, así que no hace falta cruzar con
  personas ni etapas. Los demás modos del grupo AUTO ("como pasajero", "Auto compartido",
  "Auto alquilado", "Auto eléctrico") no declaran si la persona conducía y quedan fuera; se
  reportan sus conteos.
* Horas: `hora_ini` y `hora_fin` son horas decimales (7,50 = 7:30). Se comprobó contra
  `duracion_min`: (hora_fin − hora_ini)·60 coincide a ±2 min en el 99,97 % de los viajes.
* Persona-día: cada persona reporta un único día de viajes; el orden es `orden_vj`.
* Permanencia observada del viaje k = hora_ini(k+1) − hora_fin(k) de la misma persona (siguiente
  salida en cualquier modo; es el tiempo que la PERSONA pasa en el destino).
  Variante de vehículo (sensibilidad): hasta la siguiente salida de la persona como conductora
  de auto, solo si sale de la misma ZAT a la que llegó; si no, queda sin dato.
* Regla para el último viaje del día (sin salida siguiente): si el destino es el hogar
  (motivo "A regresar al hogar" o ZAT de destino = ZAT del hogar) → PERNOCTA imputada:
  permanencia = (24 − hora_fin) + hora_ini del primer viaje del día de esa persona (supone que el
  día siguiente repite la primera salida). Si el destino no es el hogar → se EXCLUYE (sin dato).
* Inconsistencias: permanencia observada negativa (la siguiente salida es anterior a la llegada)
  → se excluye y se cuenta.
* Ocupación esperada O_zh: suma de fexp_vj de los vehículos llegados a la ZAT z que siguen
  estacionados en el instante h:30 (intervalo [llegada, llegada + permanencia) sobre un reloj
  circular de 24 h, de modo que la pernocta ocupa la madrugada). Solo entran llegadas con
  permanencia conocida o imputada; las excluidas no ocupan (O_zh queda por debajo).
* Ponderación: fexp_vj (coma decimal). Tamaños de muestra sin expandir se reportan aparte.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = next(p for p in AQUI.parents if (p / "pyproject.toml").exists())
SALIDAS = AQUI / "salidas"
EM2023_DIR = Path(os.environ.get(
    "EM2023_DIR",
    "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes"))
CONDUCTOR = "Vehículo privado como conductor"
HORAS = np.arange(24)


def num(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t.replace({"-": np.nan, "": np.nan, "No aplica": np.nan}), errors="coerce")


def canon(v):
    m = re.match(r"^(UTAM|UPR)0*(\d+)$", str(v).strip().upper())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def leer(nombre: str) -> pd.DataFrame:
    df = pd.read_csv(EM2023_DIR / nombre, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def wq(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    if not m.any():
        return np.nan
    o = np.argsort(x[m]); x, w = x[m][o], w[m][o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def resumen_perm(g: pd.DataFrame, col: str = "perm_h") -> dict:
    k = g[g[col].notna()]
    return {
        "llegadas_muestra": int(len(g)),
        "llegadas_expandidas": round(float(g.fexp_vj.sum()), 1),
        "con_permanencia_muestra": int(len(k)),
        "pct_imputada_pernocta": round(100 * float((g.estado_perm == "imputada_pernocta").mean()), 1),
        "pct_excluida": round(100 * float(g[col].isna().mean()), 1),
        "media_h": round(float(np.average(k[col], weights=k.fexp_vj)), 2) if len(k) else np.nan,
        "p25_h": round(wq(k[col], k.fexp_vj, .25), 2),
        "mediana_h": round(wq(k[col], k.fexp_vj, .5), 2),
        "p75_h": round(wq(k[col], k.fexp_vj, .75), 2),
        "media_h_solo_observada": round(float(np.average(k.loc[k.estado_perm == "observada", col],
                                                         weights=k.loc[k.estado_perm == "observada", "fexp_vj"])), 2)
        if (k.estado_perm == "observada").any() else np.nan,
    }


def main():
    SALIDAS.mkdir(exist_ok=True)
    d: dict = {}
    hog, v, eta = leer("a. Modulo hogares.csv"), leer("d. Modulo viajes.csv"), leer("e. Modulo etapas.csv")

    # ---------------- fecha de hogares: ¿día de viaje o entrevista?
    f = pd.to_datetime(hog["fecha"], format="%d/%m/%Y %H:%M")
    hog["dia_semana_fecha"] = f.dt.day_name()
    vv = v.merge(hog[["key_hg", "dia_semana_fecha"]], on="key_hg", how="left")
    vv["trab_est"] = vv.motivo_viaje.isin(["A trabajar", "A Estudiar"])
    vv["sale_5a8"] = num(vv.hora_ini).between(5, 8)
    perfil = vv.groupby("dia_semana_fecha").agg(viajes=("trab_est", "size"), hogares=("key_hg", "nunique"),
                                                 pct_trabajo_estudio=("trab_est", "mean"), pct_salida_5a8=("sale_5a8", "mean"))
    perfil.round(3).to_csv(SALIDAS / "verificacion_fecha_hogares.csv")
    d["fecha_hogares"] = {
        "rango": [str(f.min()), str(f.max())],
        "pct_con_hora_entre_15_y_23": round(100 * float(f.dt.hour.between(15, 23).mean()), 1),
        "hogares_por_dia_semana": f.dt.day_name().value_counts().to_dict(),
        "perfil_viajes_por_dia_de_fecha": perfil.round(3).reset_index().to_dict(orient="records"),
    }

    # ---------------- preparación de viajes
    v["hi"], v["hf"], v["dur"] = num(v.hora_ini), num(v.hora_fin), num(v.duracion_min)
    v["fexp_vj"] = num(v.fexp_vj)
    v["orden"] = pd.to_numeric(v.orden_vj, errors="coerce")
    v["zat_des_n"] = pd.to_numeric(v.zat_des, errors="coerce")
    v["zat_hg_n"] = pd.to_numeric(v.zat_hg, errors="coerce")
    v["utam_des_c"] = v.utam_des.map(canon)
    coh = (((v.hf - v.hi) % 24) * 60 - v.dur).abs() <= 2
    d["horas_decimales_coherentes_con_duracion_pct"] = round(100 * float(coh[v.dur.notna()].mean()), 2)
    v = v.sort_values(["key_pers", "orden", "hi"]).reset_index(drop=True)
    g = v.groupby("key_pers")
    v["hi_sig"] = g.hi.shift(-1)
    v["primer_hi"] = g.hi.transform("first")
    v["es_ultimo"] = v.hi_sig.isna() & (g.cumcount(ascending=False) == 0)
    v["cond"] = v.modo_principal_desagrupado.eq(CONDUCTOR)
    # siguiente salida como conductor (variante vehículo)
    v["hi_cond_sig"], v["zat_ori_cond_sig"] = np.nan, np.nan
    for _, idx in v[v.cond].groupby("key_pers").groups.items():
        idx = list(idx)
        for a, b in zip(idx[:-1], idx[1:]):
            v.at[a, "hi_cond_sig"] = v.at[b, "hi"]
            v.at[a, "zat_ori_cond_sig"] = pd.to_numeric(v.at[b, "zat_ori"], errors="coerce")

    grupo_auto = v[v.modo_principal_agrupado.eq("AUTO")]
    d["modos_grupo_AUTO_muestra"] = grupo_auto.modo_principal_desagrupado.value_counts().to_dict()
    c = v[v.cond].copy()
    d["viajes_conductor_auto_muestra"] = int(len(c))
    d["viajes_conductor_auto_expandidos"] = round(float(c.fexp_vj.sum()), 0)
    d["conductores_personas_muestra"] = int(c.key_pers.nunique())
    d["viajes_conductor_sin_fexp"] = int(c.fexp_vj.isna().sum())

    # ---------------- permanencia
    c["a_casa"] = c.motivo_viaje.eq("A regresar al hogar") | (c.zat_des_n == c.zat_hg_n)
    obs = c.hi_sig - c.hf
    c["perm_h"], c["estado_perm"] = np.nan, "excluida_ultimo_no_hogar"
    m_obs = c.hi_sig.notna() & (obs >= 0)
    c.loc[m_obs, "perm_h"], c.loc[m_obs, "estado_perm"] = obs[m_obs], "observada"
    m_neg = c.hi_sig.notna() & (obs < 0)
    c.loc[m_neg, "estado_perm"] = "excluida_inconsistente"
    m_pern = c.hi_sig.isna() & c.a_casa & c.hf.notna() & c.primer_hi.notna()
    c.loc[m_pern, "perm_h"] = (24 - c.hf[m_pern]) + c.primer_hi[m_pern]
    c.loc[m_pern, "estado_perm"] = "imputada_pernocta"
    c.loc[c.hi_sig.isna() & ~c.a_casa, "estado_perm"] = "excluida_ultimo_no_hogar"
    vo = c.hi_cond_sig - c.hf
    ok = c.hi_cond_sig.notna() & (vo >= 0) & (c.zat_ori_cond_sig == c.zat_des_n)
    c["perm_vehiculo_h"] = np.where(ok, vo, np.where(c.estado_perm.eq("imputada_pernocta"), c.perm_h, np.nan))

    est = c.estado_perm.value_counts()
    estw = c.groupby("estado_perm").fexp_vj.sum()
    d["estado_permanencia_muestra"] = est.to_dict()
    d["estado_permanencia_pct_muestra"] = (100 * est / est.sum()).round(1).to_dict()
    d["estado_permanencia_pct_expandido"] = (100 * estw / estw.sum()).round(1).to_dict()
    d["variante_vehiculo_con_dato_pct_muestra"] = round(100 * float(c.perm_vehiculo_h.notna().mean()), 1)

    # contraste con el tiempo de parqueo declarado en etapas (t_parqueo_min)
    e = eta[eta.modo_etapa.eq(CONDUCTOR)][["key_viaje", "lugar_parqueo", "t_parqueo_min"]].copy()
    e["t_parq_h"] = num(e.t_parqueo_min) / 60
    e = e.sort_values("t_parq_h", ascending=False).drop_duplicates("key_viaje")
    c = c.merge(e[["key_viaje", "lugar_parqueo", "t_parq_h"]], on="key_viaje", how="left")
    k = c[(c.estado_perm == "observada") & c.t_parq_h.notna()]
    d["contraste_t_parqueo_declarado"] = {
        "viajes_con_ambos": int(len(k)),
        "spearman": round(float(k[["perm_h", "t_parq_h"]].corr(method="spearman").iloc[0, 1]), 3),
        "mediana_perm_calculada_h": round(float(k.perm_h.median()), 2),
        "mediana_t_parqueo_declarado_h": round(float(k.t_parq_h.median()), 2),
        "pct_dif_abs_menor_1h": round(100 * float(((k.perm_h - k.t_parq_h).abs() < 1).mean()), 1),
    }
    lp = c.groupby("lugar_parqueo").fexp_vj.sum()
    d["lugar_parqueo_destino_pct_expandido"] = (100 * lp / lp.sum()).round(1).sort_values(ascending=False).to_dict()

    # ---------------- restringir a destinos en Bogotá (ZAT del shapefile)
    zat = gpd.read_file(EM2023_DIR / "ZAT2023.shp")
    zat_bog = set(zat.loc[zat.MUNCod == 11001, "ZAT"].astype(int))
    c["en_bogota"] = c.zat_des_n.isin(zat_bog)
    d["llegadas_conductor_destino_bogota_muestra"] = int(c.en_bogota.sum())
    d["llegadas_conductor_destino_fuera_bogota_muestra"] = int((~c.en_bogota).sum())
    c["hora_llegada"] = np.floor(c.hf).clip(0, 23)
    COLS = ["key_viaje", "key_pers", "orden", "zat_ori", "zat_des_n", "utam_des_c", "zat_hg_n", "nom_mun_des",
            "hi", "hf", "hora_llegada", "motivo_viaje", "fexp_vj", "a_casa", "estado_perm", "perm_h",
            "perm_vehiculo_h", "lugar_parqueo", "t_parq_h", "en_bogota"]
    c[COLS].to_csv(SALIDAS / "viajes_conductor_auto_permanencia.csv", index=False)
    b = c[c.en_bogota].copy()
    b["ZAT"] = b.zat_des_n.astype(int)

    # ---------------- permanencia por propósito
    filas = [{"motivo_viaje": m, **resumen_perm(gg)} for m, gg in b.groupby("motivo_viaje")]
    filas.append({"motivo_viaje": "TODOS", **resumen_perm(b)})
    pp = pd.DataFrame(filas).sort_values("llegadas_expandidas", ascending=False)
    pp.to_csv(SALIDAS / "permanencia_por_proposito.csv", index=False)
    pv = pd.DataFrame([{"motivo_viaje": m, **resumen_perm(gg, "perm_vehiculo_h")} for m, gg in b.groupby("motivo_viaje")]
                      + [{"motivo_viaje": "TODOS", **resumen_perm(b, "perm_vehiculo_h")}])
    pv.to_csv(SALIDAS / "permanencia_vehiculo_por_proposito_sensibilidad.csv", index=False)

    # ---------------- llegadas por zona y hora; ocupación
    def ocupacion(df, zona):
        out = []
        k = df[df.perm_h.notna()]
        t0, dur = k.hf.to_numpy(), k.perm_h.to_numpy()
        for h in HORAS:
            inst = h + 0.5
            dentro = ((inst - t0) % 24) < np.minimum(dur, 24)
            s = pd.Series(np.where(dentro, k.fexp_vj, 0.0), index=k.index).groupby(k[zona]).sum()
            n = pd.Series(dentro.astype(int), index=k.index).groupby(k[zona]).sum()
            out.append(pd.DataFrame({zona: s.index, "hora": h, "ocupacion_esperada": s.values.round(1),
                                     "ocupacion_muestra": n.values}))
        return pd.concat(out, ignore_index=True)

    def tablas(df, zona, universo, sufijo):
        lh = df.groupby([zona, "hora_llegada"]).agg(llegadas_expandidas=("fexp_vj", "sum"),
                                                     llegadas_muestra=("fexp_vj", "size")).reset_index()
        lh["llegadas_expandidas"] = lh.llegadas_expandidas.round(1)
        lh.rename(columns={"hora_llegada": "hora"}).to_csv(SALIDAS / f"llegadas_{sufijo}_hora.csv", index=False)
        oc = ocupacion(df, zona)
        oc.to_csv(SALIDAS / f"ocupacion_{sufijo}_hora.csv", index=False)
        filas = []
        for z, gg in df.groupby(zona):
            r = {zona: z, **resumen_perm(gg)}
            ocz = oc[oc[zona] == z]
            r["ocupacion_max_expandida"] = float(ocz.ocupacion_esperada.max()) if len(ocz) else 0.0
            r["hora_ocupacion_max"] = int(ocz.loc[ocz.ocupacion_esperada.idxmax(), "hora"]) if len(ocz) else np.nan
            r["pct_llegadas_trabajo"] = round(100 * float(gg.motivo_viaje.eq("A trabajar").mean()), 1)
            r["pct_llegadas_hogar"] = round(100 * float(gg.motivo_viaje.eq("A regresar al hogar").mean()), 1)
            filas.append(r)
        t = pd.DataFrame(filas)
        t = pd.DataFrame({zona: sorted(universo)}).merge(t, on=zona, how="left")
        t["llegadas_muestra"] = t.llegadas_muestra.fillna(0).astype(int)
        t["llegadas_expandidas"] = t.llegadas_expandidas.fillna(0.0)
        t.to_csv(SALIDAS / f"llegadas_permanencia_por_{sufijo}.csv", index=False)
        return t

    tz = tablas(b, "ZAT", zat_bog, "zat")
    utam = gpd.read_file(RAIZ / "outputs" / "social_equity_followup_outputs"
                         / "utam_shp" / "UTAM2023" / "UTAM2023.shp")
    u_univ = set(utam.UTAM.map(canon))
    b["UTAM"] = b.utam_des_c
    d["llegadas_bogota_con_utam_fuera_del_shapefile"] = int((~b.UTAM.isin(u_univ)).sum())
    tu = tablas(b[b.UTAM.isin(u_univ)], "UTAM", u_univ, "utam")

    # perfil horario ciudad
    ph = b.groupby("hora_llegada").fexp_vj.sum()
    oc_c = ocupacion(b.assign(ciudad="Bogota"), "ciudad")
    pd.DataFrame({"hora": HORAS, "llegadas_expandidas": ph.reindex(HORAS).fillna(0).round(0).values,
                  "ocupacion_esperada": oc_c.set_index("hora").ocupacion_esperada.reindex(HORAS).values}).to_csv(
        SALIDAS / "perfil_horario_bogota.csv", index=False)

    # ---------------- escasez
    for nom, t, n in [("zat", tz, len(zat_bog)), ("utam", tu, len(u_univ))]:
        d[f"escasez_{nom}"] = {
            "universo": n,
            "sin_llegadas": int((t.llegadas_muestra == 0).sum()),
            "menos_de_10": int((t.llegadas_muestra < 10).sum()),
            "menos_de_30": int((t.llegadas_muestra < 30).sum()),
            "mediana_llegadas_muestra": float(t.llegadas_muestra.median()),
            "pct_llegadas_en_zonas_con_30_o_mas": round(100 * float(t.loc[t.llegadas_muestra >= 30, "llegadas_muestra"].sum()
                                                                     / t.llegadas_muestra.sum()), 1),
        }
    d["permanencia_todos_bogota"] = resumen_perm(b)
    d["permanencia_vehiculo_todos_bogota"] = resumen_perm(b, "perm_vehiculo_h")
    (SALIDAS / "diagnostico_parte2.json").write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    pd.set_option("display.width", 250)
    print(pp.to_string(index=False))


if __name__ == "__main__":
    main()
