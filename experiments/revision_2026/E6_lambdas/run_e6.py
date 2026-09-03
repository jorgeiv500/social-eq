#!/usr/bin/env python
"""E6 — Anclaje empírico de los parámetros de decaimiento λ (R1.7).

Calcula la distribución ponderada de duración de viaje por propósito con el módulo
de viajes de la Encuesta de Movilidad (Drive sincronizado) y la compara con
λ_res=15, λ_work=25, λ_pub=10 del manuscrito. Diseño registrado ex ante en README.md.

Determinista, sin NEOS. Salidas: salidas/tablas/duracion_por_proposito.csv y
salidas/tablas/manifiesto_e6.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Archivo de viajes de la EM2023 (no se redistribuye: descargar de la SDM, ver README).
# Se toma de la variable de entorno EM2023_VIAJES; por defecto, data/em2023/viajes.csv.
RUTA_VIAJES = Path(os.environ.get("EM2023_VIAJES", str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()) / "data" / "em2023" / "viajes.csv")))
DIR_SALIDAS = Path(__file__).resolve().parent / "salidas" / "tablas"

# Mapeo propósito -> grupo (registrado ex ante en README.md)
GRUPOS = {
    "workplace": ["A trabajar"],
    "residential": ["A regresar al hogar"],
    "public_nucleo": [
        "A realizar algún trámite personal",
        "A realizar compras",
        "A asuntos médicos personales",
    ],
    "public_extendido_extra": [
        "A realizar actividades recreativas y culturales",
        "A realizar actividades físicas y/o deportivas",
        "A asistir actividad de tipo religioso y/o culto",
    ],
}
LAMBDAS_PAPER = {"workplace": 25.0, "residential": 15.0, "public": 10.0}

FILTROS_MODO = {
    "auto": ["AUTO"],
    "motorizado_privado": ["AUTO", "MOTO", "TAXI OCUPADO"],
    "todos_los_modos": None,
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def stat_ponderado(valores: np.ndarray, pesos: np.ndarray) -> dict:
    orden = np.argsort(valores)
    v, w = valores[orden], pesos[orden]
    acum = np.cumsum(w) / w.sum()

    def cuantil(q: float) -> float:
        return float(v[np.searchsorted(acum, q)])

    return {
        "n_muestral": int(len(v)),
        "suma_pesos": float(w.sum()),
        "media_ponderada_min": float(np.average(v, weights=w)),
        "p25_ponderado_min": cuantil(0.25),
        "mediana_ponderada_min": cuantil(0.50),
        "p75_ponderado_min": cuantil(0.75),
    }


def main() -> None:
    DIR_SALIDAS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RUTA_VIAJES, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n_bruto = len(df)

    for col in ("duracion_min", "fexp_vj"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    df["motivo_viaje"] = df["motivo_viaje"].astype(str).str.strip()
    df["modo_principal_agrupado"] = df["modo_principal_agrupado"].astype(str).str.strip()
    df["nom_mun_des"] = df["nom_mun_des"].astype(str).str.strip()

    base = df[
        (df["nom_mun_des"] == "Bogotá")
        & (df["duracion_min"] > 0)
        & (df["fexp_vj"] > 0)
    ].copy()
    n_filtrado = len(base)

    grupos_eval = {
        "workplace": GRUPOS["workplace"],
        "residential": GRUPOS["residential"],
        "public_nucleo": GRUPOS["public_nucleo"],
        "public_extendido": GRUPOS["public_nucleo"] + GRUPOS["public_extendido_extra"],
    }

    filas = []
    for nombre_filtro, modos in FILTROS_MODO.items():
        sub_modo = base if modos is None else base[base["modo_principal_agrupado"].isin(modos)]
        for grupo, motivos in grupos_eval.items():
            sel = sub_modo[sub_modo["motivo_viaje"].isin(motivos)]
            if sel.empty:
                continue
            fila = {
                "filtro_modo": nombre_filtro,
                "grupo": grupo,
                "lambda_paper_min": LAMBDAS_PAPER.get(
                    grupo, LAMBDAS_PAPER["public"] if grupo.startswith("public") else np.nan
                ),
            }
            fila.update(
                stat_ponderado(sel["duracion_min"].to_numpy(), sel["fexp_vj"].to_numpy())
            )
            filas.append(fila)

    tabla = pd.DataFrame(filas)
    ruta_tabla = DIR_SALIDAS / "duracion_por_proposito.csv"
    tabla.to_csv(ruta_tabla, index=False)

    motivos_universo = sorted(df["motivo_viaje"].dropna().unique().tolist())
    incluidos = sorted({m for lst in grupos_eval.values() for m in lst})
    manifiesto = {
        "experimento": "E6_lambdas",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "insumo": str(RUTA_VIAJES),
        "insumo_md5": md5(RUTA_VIAJES),
        "n_viajes_bruto": n_bruto,
        "n_viajes_tras_filtros_base": n_filtrado,
        "filtros_base": "nom_mun_des == 'Bogotá' & duracion_min > 0 & fexp_vj > 0",
        "filtros_modo": {k: (v if v else "todos") for k, v in FILTROS_MODO.items()},
        "mapeo_grupos": grupos_eval,
        "motivos_excluidos": sorted(set(motivos_universo) - set(incluidos)),
        "lambdas_paper_min": LAMBDAS_PAPER,
        "nota": (
            "duracion_min es puerta a puerta autorreportada; la comparación con "
            "lambda_v es de orden y magnitud, no una estimación de deterrencia."
        ),
    }
    ruta_manifiesto = DIR_SALIDAS / "manifiesto_e6.json"
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Viajes brutos: {n_bruto:,}; tras filtros base: {n_filtrado:,}")
    print(tabla.to_string(index=False))
    print(f"\nEscrito: {ruta_tabla}\nEscrito: {ruta_manifiesto}")


if __name__ == "__main__":
    main()
