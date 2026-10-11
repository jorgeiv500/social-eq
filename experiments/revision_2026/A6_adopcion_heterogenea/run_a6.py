#!/usr/bin/env python
"""A6 — Escenarios 2030 con adopción espacialmente diferenciada (JTRG-D-26-00692).

Responde a la objeción previsible sobre §5.6: en A3 las cuotas 2030 se escalan
proporcionalmente y la adopción se supone espacialmente uniforme, de modo que la señal
de demanda zonal no cambia y la estabilidad del Gini sale en parte por construcción.

Aquí se mantienen las cuotas por sede de A3-2030 (derivadas en tiempo de ejecución con
`run_a3.derivar_cuotas`) y se cambia la DISTRIBUCIÓN ESPACIAL de la demanda:

    T_iv(2030) = Q_v(2030) * [ s * pi_iv + (1 - s) * g_iv^H ]

- pi_iv: participación del paper (`make_targets`: prioridad ST-GNN x peso de sede).
- s = VE(2024) / VE(2030): fracción de la flota 2030 que ya existía en 2024 (stock que se
  supone distribuido como la señal observada).
- g_iv^H: distribución espacial del INCREMENTO de flota 2024->2030 bajo la hipótesis H:
    * U  (control):     g = pi                          -> reproduce exactamente A3-2030.
    * H1 (concentrada): g ∝ pi_iv * c_i, con c_i = proporción de hogares con automóvil
                        particular propio (EM-2023, módulo de vehículos, expandido).
    * H2 (convergencia): g ∝ h_i * w_iv, con h_i = hogares expandidos (EM-2023) y w_iv el
                        peso de sede del paper; equivale a crecimiento inversamente
                        proporcional a la intensidad actual de la señal por hogar (pi_i/h_i),
                        es decir, la adopción por hogar converge entre zonas.
  Las tres conservan el total por sede: sum_i T_iv = Q_v(2030).

Re-corre S0 -> S1 -> S2 por hipótesis (H1, H2) con el modelo EXISTENTE de
src/social_eq/followup.py, sin modificarlo.

SEGURIDAD: sin `--lanzar` solo se construye el proxy, se calculan los objetivos y se
imprimen las comprobaciones; no se envía nada a NEOS.

Uso:
    python run_a6.py                         # corrida en seco
    python run_a6.py --lanzar                # 6 corridas NEOS (S0/S1/S2 x H1/H2)
    python run_a6.py --lanzar --hipotesis H1 # 3 corridas
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

import comun  # noqa: E402
from social_eq.followup import (  # noqa: E402
    extract_solution,
    extract_solution_refined,
    make_targets,
    normalize_zone_code,
    solve_equity,
    solve_equity_refined,
    solve_s0,
    solve_safe,
    summarize_scenario,
)


def _cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Reutilización literal de la derivación de cuotas de A3 y del Gini per cápita del repo.
run_a3 = _cargar_modulo("run_a3", AQUI.parent / "A3_escenarios_crecimiento" / "run_a3.py")
intervencion = _cargar_modulo("intervention_size", comun.RAIZ / "scripts" / "intervention_size.py")

DIR_SALIDAS = AQUI / "salidas"
DIR_TABLAS = DIR_SALIDAS / "tablas"
DIR_A3 = AQUI.parent / "A3_escenarios_crecimiento" / "salidas" / "tablas"
HORIZONTE = 2030
EM2023_DIR = Path(os.environ.get(
    "EM2023_DIR",
    "/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes",
))
MODULO_HOGARES = "a. Modulo hogares.csv"
MODULO_VEHICULOS = "b. Modulo vehiculos.csv"
MODULO_VIAJES = "d. Modulo viajes.csv"
N_MIN_MUESTRA = 30  # hogares (residencia) o viajes (destino) mínimos para usar el valor de la UTAM
RUTA_PROXY = DIR_TABLAS / "proxy_adopcion_utam.csv"
TIPOS_AUTO = ("Automóvil", "Campero/Camioneta")
COMBUSTIBLES_VE = ("Eléctrico", "Híbrido (eléctrico- gasolina / diesel)")
HIPOTESIS = ("H1", "H2")


# ------------------------------------------------------------------ proxy zonal (EM-2023)
def _a_numero(serie: pd.Series) -> pd.Series:
    """Números con coma decimal y punto de miles (formato de los módulos EM-2023).

    Misma regla que `to_num` de scripts/retrain_stgnn.py.
    """
    txt = serie.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    txt = txt.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def _md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def construir_proxy() -> pd.DataFrame:
    """Tenencia de automóvil por hogar y por UTAM desde la Encuesta de Movilidad 2023.

    Si los módulos originales no están disponibles, se usa la tabla derivada archivada.
    """
    ruta_h, ruta_v = EM2023_DIR / MODULO_HOGARES, EM2023_DIR / MODULO_VEHICULOS
    if not (ruta_h.exists() and ruta_v.exists() and (EM2023_DIR / MODULO_VIAJES).exists()):
        if RUTA_PROXY.exists():
            print(f"Módulos EM-2023 no disponibles; se usa la tabla archivada {RUTA_PROXY}")
            return pd.read_csv(RUTA_PROXY)
        raise SystemExit(f"Faltan {ruta_h} / {ruta_v} y no hay tabla archivada en {RUTA_PROXY}.")

    ruta_vj = EM2023_DIR / MODULO_VIAJES
    hog = pd.read_csv(ruta_h, sep=";", encoding="latin-1", low_memory=False)
    veh = pd.read_csv(ruta_v, sep=";", encoding="latin-1", low_memory=False)
    vj = pd.read_csv(ruta_vj, sep=";", encoding="latin-1", low_memory=False,
                     usecols=["key_hg", "utam_des", "localidad_des", "nom_mun_des", "fexp_vj"])
    hog["peso"] = _a_numero(hog["fexp_hg"])
    hog["UTAM"] = hog["cod_utam_hg"].map(normalize_zone_code)
    if hog["peso"].isna().any() or hog["UTAM"].isna().any():
        raise SystemExit("Hogares sin factor de expansión o sin UTAM en EM-2023.")
    # Unidad de respaldo para zonas con muestra pequeña: localidad (Bogotá) o municipio.
    hog["respaldo"] = np.where(hog["nom_loc_hg"].astype(str).str.strip().eq("No aplica"),
                               hog["nom_mpio_hg"].astype(str).str.strip(), hog["nom_loc_hg"].astype(str).str.strip())

    autos = veh[
        veh["tipo_vehículo"].isin(TIPOS_AUTO)
        & veh["tipo_placa"].eq("Privada")
        & veh["propiedad_vh"].eq("Este hogar")
    ]
    ve = veh[veh["tipo_vehículo"].isin(TIPOS_AUTO) & veh["tipo_combustible"].isin(COMBUSTIBLES_VE)]
    hog["autos"] = hog["key_hg"].map(autos.groupby("key_hg")["cantidad"].sum()).fillna(0.0)
    hog["ve_hibridos"] = hog["key_hg"].map(ve.groupby("key_hg")["cantidad"].sum()).fillna(0.0)
    hog["con_auto"] = (hog["autos"] > 0).astype(float)
    hog["estrato"] = pd.to_numeric(hog["estrato_hg"], errors="coerce")

    def agregar(g: pd.DataFrame) -> pd.Series:
        w = g["peso"]
        est = g["estrato"].notna()
        return pd.Series({
            "respaldo_residencia": g["respaldo"].mode().iloc[0],
            "hogares_muestra": int(len(g)),
            "hogares_expandidos": float(w.sum()),
            "share_hogares_con_auto_bruto": float(np.average(g["con_auto"], weights=w)),
            "autos_por_hogar": float(np.average(g["autos"], weights=w)),
            "ve_hibridos_por_hogar": float(np.average(g["ve_hibridos"], weights=w)),
            "estrato_medio": float(np.average(g.loc[est, "estrato"], weights=w[est])) if est.any() else np.nan,
        })

    proxy = hog.groupby("UTAM").apply(agregar).reset_index()
    resp_res = hog.groupby("respaldo").apply(lambda g: np.average(g["con_auto"], weights=g["peso"]))
    pequena = proxy["hogares_muestra"] < N_MIN_MUESTRA
    proxy["c_residencia"] = np.where(pequena, proxy["respaldo_residencia"].map(resp_res),
                                     proxy["share_hogares_con_auto_bruto"])
    proxy["c_residencia_fuente"] = np.where(pequena, "respaldo_localidad_municipio", "utam")

    # Extremo de destino: proporción de viajes que llegan a la zona hechos por miembros de
    # hogares con automóvil (para las sedes workplace y public, cuya demanda no es residencial).
    vj["peso"] = _a_numero(vj["fexp_vj"])
    vj["UTAM"] = vj["utam_des"].map(normalize_zone_code)
    vj = vj[vj["UTAM"].str.match(r"^(UTAM|UPR)\d+$", na=False) & vj["peso"].notna()].copy()
    vj["con_auto"] = vj["key_hg"].map(hog.set_index("key_hg")["con_auto"])
    vj = vj[vj["con_auto"].notna()]
    loc = vj["localidad_des"].astype(str).str.strip()
    vj["respaldo"] = np.where(loc.isin(["", "nan", "No aplica", "-"]), vj["nom_mun_des"].astype(str).str.strip(), loc)
    des = vj.groupby("UTAM").apply(lambda g: pd.Series({
        "respaldo_destino": g["respaldo"].mode().iloc[0],
        "viajes_destino_muestra": int(len(g)),
        "viajes_destino_expandidos": float(g["peso"].sum()),
        "share_viajes_destino_con_auto_bruto": float(np.average(g["con_auto"], weights=g["peso"])),
    })).reset_index()
    resp_des = vj.groupby("respaldo").apply(lambda g: np.average(g["con_auto"], weights=g["peso"]))
    proxy = proxy.merge(des, on="UTAM", how="left")
    pequena_d = proxy["viajes_destino_muestra"].fillna(0) < N_MIN_MUESTRA
    proxy["c_destino"] = np.where(pequena_d, proxy["respaldo_destino"].map(resp_des),
                                  proxy["share_viajes_destino_con_auto_bruto"])
    proxy["c_destino"] = proxy["c_destino"].fillna(proxy["c_residencia"])  # sin viajes de destino: residencia
    proxy["c_destino_fuente"] = np.where(proxy["viajes_destino_muestra"].isna(), "residencia",
                                         np.where(pequena_d, "respaldo_localidad_municipio", "utam"))

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    proxy.to_csv(RUTA_PROXY, index=False)
    (DIR_TABLAS / "fuente_proxy_em2023.json").write_text(json.dumps({
        "fuente": "Encuesta de Movilidad de Bogotá 2023 (Secretaría Distrital de Movilidad), "
                  "módulos de hogares, vehículos y viajes; microdatos con factores de expansión fexp_hg y fexp_vj",
        "archivos": {
            MODULO_HOGARES: {"md5": _md5(ruta_h), "filas": int(len(hog))},
            MODULO_VEHICULOS: {"md5": _md5(ruta_v), "filas": int(len(veh))},
            MODULO_VIAJES: {"md5": _md5(ruta_vj), "filas_validas_destino": int(len(vj))},
        },
        "definicion_automovil": f"tipo_vehículo en {list(TIPOS_AUTO)}, tipo_placa = Privada, propiedad_vh = Este hogar",
        "definicion_ve": f"tipo_vehículo en {list(TIPOS_AUTO)}, tipo_combustible en {list(COMBUSTIBLES_VE)}",
        "umbral_muestra_minima": N_MIN_MUESTRA,
        "utam_con_respaldo_residencia": proxy.loc[pequena, "UTAM"].tolist(),
        "utam_con_respaldo_destino": proxy.loc[proxy["c_destino_fuente"].ne("utam"), "UTAM"].tolist(),
        "control_total_hogares_expandidos": float(hog["peso"].sum()),
        "control_share_hogares_con_auto_total": float(np.average(hog["con_auto"], weights=hog["peso"])),
        "control_ve_hibridos_expandidos_total": float((hog["ve_hibridos"] * hog["peso"]).sum()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return proxy


# ------------------------------------------------------------------ objetivos 2030
def objetivos(ctx, config_h, proxy: pd.DataFrame, s: float) -> dict[str, np.ndarray]:
    """Objetivos T_iv(2030) para U (control), H1 y H2, en el orden de zonas del modelo."""
    p = ctx.df_inputs[["UTAM"]].merge(proxy, on="UTAM", how="left")
    if p[["c_residencia", "c_destino", "hogares_expandidos"]].isna().any().any():
        faltan = p.loc[p[["c_residencia", "c_destino"]].isna().any(axis=1), "UTAM"].tolist()
        raise SystemExit(f"UTAM sin proxy EM-2023: {faltan}")
    # Sede residencial: tenencia en la zona de residencia; workplace/public: en los viajes que llegan.
    c = np.column_stack([p["c_residencia" if v == "residential" else "c_destino"].to_numpy(dtype=float)
                         for v in config_h.venues])
    vol = np.column_stack([p["hogares_expandidos" if v == "residential" else "viajes_destino_expandidos"]
                           .fillna(0.0).to_numpy(dtype=float) for v in config_h.venues])
    pi, _ = make_targets(ctx.df_inputs, config_h)
    q = np.array([config_h.quotas[v] for v in config_h.venues], dtype=float)
    pesos_sede = np.column_stack([ctx.df_inputs[v].to_numpy(dtype=float) for v in config_h.venues])

    def normalizar(m: np.ndarray) -> np.ndarray:
        return m / m.sum(axis=0, keepdims=True)

    g = {
        "U": pi,
        "H1": normalizar(pi * c),
        "H2": normalizar(vol * pesos_sede),
    }
    return {k: q[None, :] * (s * pi + (1.0 - s) * gk) for k, gk in g.items()}


# ------------------------------------------------------------------ métricas
def provision_cuartil(df_asig: pd.DataFrame, poblacion: pd.DataFrame) -> dict[str, float]:
    """Misma lógica que scripts/intervention_size.py (puertos por 10.000 hab., cuartil vulnerable)."""
    d = df_asig[["UTAM", "ports_total", "vulnerability_index"]].copy()
    d["UTAM"] = d["UTAM"].map(normalize_zone_code)
    d = d.merge(poblacion, on="UTAM", how="left")
    vq_all = d["vulnerability_index"] >= d["vulnerability_index"].quantile(0.75)
    con_pob = d["dane_population"].fillna(0.0) > 0
    d, vq = d.loc[con_pob].copy(), vq_all.loc[con_pob]
    pob_v, por_v = d.loc[vq, "dane_population"].sum(), d.loc[vq, "ports_total"].sum()
    pob_r, por_r = d.loc[~vq, "dane_population"].sum(), d.loc[~vq, "ports_total"].sum()
    d["ports_per_10k"] = 1e4 * d["ports_total"] / d["dane_population"]
    return {
        "puertos_cuartil_vulnerable": int(por_v),
        "puertos_10k_cuartil_vulnerable": float(1e4 * por_v / pob_v),
        "puertos_10k_resto": float(1e4 * por_r / pob_r),
        "puertos_10k_ciudad": float(1e4 * (por_v + por_r) / (pob_v + pob_r)),
        "gini_per_capita_pond_poblacion": float(intervencion.gini_ponderado(
            d["ports_per_10k"].to_numpy(dtype=float), d["dane_population"].to_numpy(dtype=float))),
    }


def spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    j = a[["UTAM", "ports_total"]].merge(b[["UTAM", "ports_total"]], on="UTAM", suffixes=("_a", "_b"))
    return float(j["ports_total_a"].corr(j["ports_total_b"], method="spearman"))


def metricas_referencia(ctx, poblacion: pd.DataFrame, config_h) -> list[dict]:
    """Filas de referencia sin NEOS: A3-2030 uniforme (S0/S1/S2) y S2 del paper."""
    filas = []
    a3 = {e: pd.read_csv(DIR_A3 / f"asignacion_{e}_{HORIZONTE}.csv") for e in ("S0", "S1", "S2")}
    resumen_a3 = pd.read_csv(DIR_A3 / "resumen_crecimiento.csv")
    for e, df in a3.items():
        r = resumen_a3[(resumen_a3["horizonte"] == HORIZONTE) & (resumen_a3["escenario"] == e)].iloc[0]
        fila = {
            "hipotesis": "U_A3", "escenario": e, "termination": r["termination"],
            "objective_abs_dev": float(r["objective_abs_dev"]),
            "u_star": float(r["u_star_horizonte"]),
            "perdida_fidelidad_pct": float(r["perdida_fidelidad_pct"]),
            "gini_no_ponderado_6d": float(r["gini_no_ponderado_6d"]),
            "gini_ponderado_6d": float(r["gini_ponderado_6d"]),
            "cobertura_15min_vulnerables": float(r["cobertura_15min_vulnerables"]),
            **provision_cuartil(df, poblacion),
            "spearman_vs_A3_mismo_escenario": 1.0,
            "spearman_vs_S2_paper": spearman(df, comun.leer_asignacion_paper("S2")),
        }
        if e == "S2":
            fila.update({f"{k}_S2_vs_S1": v for k, v in comun.comparar_asignaciones(df, a3["S1"]).items()})
        filas.append(fila)
    met = comun.leer_metricas_paper()
    s2 = comun.leer_asignacion_paper("S2")
    fila_m = met.loc[met["scenario"].astype(str).eq("S2")].iloc[0]
    filas.append({
        "hipotesis": "paper_2024", "escenario": "S2", "termination": "archivada",
        "gini_no_ponderado_6d": comun.redondear6(fila_m["accessibility_gini"]),
        "gini_ponderado_6d": comun.redondear6(fila_m["accessibility_weighted_gini"]),
        "cobertura_15min_vulnerables": float(fila_m["coverage_within_threshold_vulnerable"]),
        **provision_cuartil(s2, poblacion),
        "spearman_vs_S2_paper": 1.0,
        **{f"{k}_S2_vs_S1": v for k, v in comun.comparar_asignaciones(s2, comun.leer_asignacion_paper("S1")).items()},
    })
    return filas


# ------------------------------------------------------------------ principal
def main() -> int:
    parser = argparse.ArgumentParser(description="A6: adopción espacialmente diferenciada en 2030.")
    parser.add_argument("--lanzar", action="store_true", help="Enviar realmente los trabajos a NEOS.")
    parser.add_argument("--hipotesis", default="H1,H2", help="Hipótesis a correr, separadas por comas.")
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    args = parser.parse_args()
    hipotesis = [x.strip() for x in args.hipotesis.split(",") if x.strip()]
    if any(x not in HIPOTESIS for x in hipotesis):
        raise SystemExit(f"Hipótesis válidas: {HIPOTESIS}")

    print("Preparando insumos (mismo camino de código que el pipeline oficial)...", flush=True)
    ctx = comun.preparar_contexto(DIR_SALIDAS, config=None)
    config_base = ctx.config
    n_zonas = len(ctx.df_inputs)
    deriv = run_a3.derivar_cuotas(config_base, [HORIZONTE]).iloc[0]
    cuotas = {v: int(deriv[f"cuota_{v}"]) for v in config_base.venues}
    config_h = replace(config_base, quotas=cuotas)
    s = float(deriv["ve_2024"]) / float(deriv["flota_proyectada"])
    time_limit = args.time_limit if args.time_limit is not None else config_base.solver_time_limit
    mip_gap = args.mip_gap if args.mip_gap is not None else config_base.solver_mip_gap

    proxy = construir_proxy()
    T = objetivos(ctx, config_h, proxy, s)
    poblacion = pd.read_csv(comun.RAIZ / "data" / "inputs" / "utam_social_vulnerability.csv")
    poblacion["UTAM"] = poblacion["UTAM"].map(normalize_zone_code)
    poblacion = poblacion[["UTAM", "dane_population"]]

    # ---------------- comprobaciones (corrida en seco)
    _pi, target_a3 = make_targets(ctx.df_inputs, config_h)
    nu = ctx.vector_vulnerabilidad
    vq = ctx.mascara_alta_vulnerabilidad
    print(f"\nCuotas 2030 (A3): {cuotas} | VE2024={deriv['ve_2024']:.0f}, VE2030={deriv['flota_proyectada']:.0f}, s={s:.6f}")
    chequeos = []
    for k, t in T.items():
        tot = t.sum(axis=1)
        a_i = tot / target_a3.sum(axis=1)
        chequeos.append({
            "hipotesis": k,
            "max_err_total_por_sede": float(np.max(np.abs(t.sum(axis=0) - np.array(list(cuotas.values()))))),
            "max_dif_vs_A3": float(np.max(np.abs(t - target_a3))),
            "masa_reubicada_vs_A3_pct": float(100.0 * 0.5 * np.abs(t - target_a3).sum() / t.sum()),
            "multiplicador_p05": float(np.percentile(a_i, 5)),
            "multiplicador_mediana": float(np.median(a_i)),
            "multiplicador_p95": float(np.percentile(a_i, 95)),
            "spearman_multiplicador_vs_nu": float(pd.Series(a_i).corr(pd.Series(nu), method="spearman")),
            "share_objetivo_cuartil_vulnerable": float(t[vq].sum() / t.sum()),
            "spearman_objetivo_vs_A3": float(pd.Series(tot).corr(pd.Series(target_a3.sum(axis=1)), method="spearman")),
        })
    df_chk = pd.DataFrame(chequeos)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df_chk.to_csv(DIR_TABLAS / "comprobaciones_objetivos.csv", index=False)
    p = ctx.df_inputs[["UTAM", "vulnerability_index"]].merge(proxy, on="UTAM", how="left")
    validacion = {
        "spearman_c_residencia_vs_ve_hibridos_por_hogar": float(p["c_residencia"].corr(p["ve_hibridos_por_hogar"], method="spearman")),
        "spearman_c_residencia_vs_estrato_medio": float(p["c_residencia"].corr(p["estrato_medio"], method="spearman")),
        "spearman_c_residencia_vs_nu": float(p["c_residencia"].corr(p["vulnerability_index"], method="spearman")),
        "spearman_c_destino_vs_nu": float(p["c_destino"].corr(p["vulnerability_index"], method="spearman")),
        "spearman_c_residencia_vs_c_destino": float(p["c_residencia"].corr(p["c_destino"], method="spearman")),
        "spearman_ve_hibridos_vs_nu": float(p["ve_hibridos_por_hogar"].corr(p["vulnerability_index"], method="spearman")),
        "hogares_muestra_min": int(p["hogares_muestra"].min()),
        "hogares_muestra_mediana": float(p["hogares_muestra"].median()),
        "utam_con_algun_ve_hibrido": int((p["ve_hibridos_por_hogar"] > 0).sum()),
    }
    (DIR_TABLAS / "validacion_proxy.json").write_text(json.dumps(validacion, indent=2), encoding="utf-8")
    np.savez(DIR_TABLAS / "objetivos_2030.npz", **T, UTAM=ctx.df_inputs["UTAM"].to_numpy(dtype=str))
    pd.set_option("display.width", 200)
    print(df_chk.round(6).to_string(index=False))
    print("Validación del proxy:", json.dumps(validacion, indent=2))

    ref = metricas_referencia(ctx, poblacion, config_h)
    ruta_resumen = DIR_TABLAS / "resumen_adopcion_heterogenea.csv"
    print(f"\nCorridas NEOS previstas: {3 * len(hipotesis)} ({hipotesis}) | timelimit={time_limit}s, mipgap={mip_gap}")
    print(f"Correo NEOS: {comun.email_neos(config_base)}")
    if not args.lanzar:
        pd.DataFrame(ref).to_csv(DIR_TABLAS / "referencias_sin_neos.csv", index=False)
        print("Corrida en seco: NO se envió nada a NEOS.")
        return 0

    a3 = {e: pd.read_csv(DIR_A3 / f"asignacion_{e}_{HORIZONTE}.csv") for e in ("S0", "S1", "S2")}
    s2_paper = comun.leer_asignacion_paper("S2")
    filas: list[dict] = list(ref)

    for hip in hipotesis:
        target = T[hip]
        asign: dict[str, pd.DataFrame] = {}

        def registrar(esc, n_sol, u_sol, a_sol, u_star, terminacion, extra=None):
            df, m = summarize_scenario(f"{esc}_{hip}_{HORIZONTE}", ctx.df_inputs, n_sol, u_sol, a_sol,
                                       ctx.tmat, ctx.mascara_alta_vulnerabilidad, config=config_h)
            df.to_csv(DIR_TABLAS / f"asignacion_{esc}_{hip}.csv", index=False)
            fila = {
                "hipotesis": hip, "escenario": esc, "termination": terminacion,
                "objective_abs_dev": float(m["objective_abs_dev"]), "u_star": u_star,
                "perdida_fidelidad_pct": 100.0 * (float(u_sol.sum()) - u_star) / u_star,
                "gini_no_ponderado_6d": comun.redondear6(m["accessibility_gini"]),
                "gini_ponderado_6d": comun.redondear6(m["accessibility_weighted_gini"]),
                "cobertura_15min_vulnerables": float(m["coverage_within_threshold_vulnerable"]),
                **provision_cuartil(df, poblacion),
                "spearman_vs_A3_mismo_escenario": spearman(df, a3[esc]),
                "spearman_vs_S2_paper": spearman(df, s2_paper),
                **{f"{k}_vs_A3_mismo_escenario": v for k, v in comun.comparar_asignaciones(df, a3[esc]).items()},
                **(extra or {}),
            }
            filas.append(fila)
            pd.DataFrame(filas).to_csv(ruta_resumen, index=False)
            (DIR_TABLAS / f"manifiesto_{esc}_{hip}.json").write_text(json.dumps({
                **fila, "time_limit_s": time_limit, "mip_gap": mip_gap, "cuotas": cuotas, "s_stock_2024": s,
                "metricas_precision_completa": {k: v for k, v in m.items() if k != "scenario"},
            }, indent=2, default=float), encoding="utf-8")
            asign[esc] = df
            return fila

        print(f"[{hip}] S0 vía NEOS/CPLEX...", flush=True)
        m0, u_star = solve_s0(target, config=config_h)
        if not solve_safe(m0):
            raise SystemExit(f"S0_{hip} no resolvió: {getattr(m0, '_termination_condition', None)}")
        n0, u0, _ = extract_solution(m0, n_zonas, len(config_h.venues))
        a0 = comun.calcular_accesibilidad(n0, ctx.coeffs, config_h)
        registrar("S0", n0, u0, a0, u_star, str(getattr(m0, "_termination_condition", "desconocida")))

        print(f"[{hip}] S1 vía NEOS/CPLEX...", flush=True)
        eps1 = max(config_h.epsilon_relax, config_h.epsilon_s1)
        m1 = solve_equity(target, ctx.coeffs, u_star, config=config_h, nu=None, epsilon=eps1)
        if not solve_safe(m1, accept_feasible=True):
            raise SystemExit(f"S1_{hip} no resolvió: {getattr(m1, '_termination_condition', None)}")
        n1, u1, a1 = extract_solution(m1, n_zonas, len(config_h.venues))
        registrar("S1", n1, u1, a1, u_star, str(getattr(m1, "_termination_condition", "desconocida")), {"epsilon": eps1})

        print(f"[{hip}] S2 vía NEOS/CPLEX...", flush=True)
        eps2 = max(config_h.epsilon_relax, config_h.epsilon_s2)
        m2, _ = solve_equity_refined(
            target=target, coeffs=ctx.coeffs, u_star=u_star, config=config_h,
            nu=ctx.vector_vulnerabilidad, epsilon=eps2, kappa=config_h.s2_kappa,
            floor_mask=None, floor_reference=None, use_hard_floor=False,
            caps=config_h.max_ports_matrix, time_limit=time_limit, mip_gap=mip_gap,
        )
        if not solve_safe(m2, accept_feasible=True):
            raise SystemExit(f"S2_{hip} no resolvió: {getattr(m2, '_termination_condition', None)}")
        n2, u2, a2, _ = extract_solution_refined(m2, n_zonas, len(config_h.venues))
        fila = registrar("S2", n2, u2, a2, u_star, str(getattr(m2, "_termination_condition", "desconocida")),
                         {"epsilon": eps2})
        fila.update({f"{k}_S2_vs_S1": v for k, v in comun.comparar_asignaciones(asign["S2"], asign["S1"]).items()})
        pd.DataFrame(filas).to_csv(ruta_resumen, index=False)
        print(f"[{hip}] terminado: {fila}", flush=True)

    print(f"\nResumen: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
