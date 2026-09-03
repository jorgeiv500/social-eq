"""Utilidades compartidas del bloque experimental A — revisión JTRG-D-26-00692.

Este módulo NO define modelos nuevos: reutiliza vía import las funciones del
pipeline oficial (`src/social_eq/followup.py`) para cargar insumos, construir
objetivos y coeficientes de accesibilidad, y comparar asignaciones. Ningún
código de este módulo envía trabajos a NEOS por sí mismo; eso solo ocurre
dentro de los runners y únicamente con la bandera `--lanzar`.

Convenciones:
- Todas las rutas se resuelven desde la raíz del repositorio (RAIZ).
- Las referencias del paper (u*, asignaciones S1/S2) se leen de las tablas
  archivadas en `outputs/social_equity_followup_outputs/paper_v6_experiments/tables/`,
  que corresponden a la corrida oficial del manuscrito (2026-04-15, ver README
  del repo). No se recalculan salvo petición explícita.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Raíz del repositorio: el primer ancestro que contiene pyproject.toml (independiente de la
# profundidad a la que se copie esta carpeta)
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(RAIZ / "src"))

from social_eq.followup import (  # noqa: E402  (import tras ajustar sys.path)
    InputFiles,
    ModelConfig,
    ProjectPaths,
    build_access_coeffs,
    load_minimal_inputs,
    make_targets,
)

# Tablas archivadas de la corrida oficial del paper.
TABLAS_PAPER = RAIZ / "outputs" / "social_equity_followup_outputs" / "paper_v6_experiments" / "tables"

SEDES = ("residential", "workplace", "public")
COLUMNAS_PUERTOS = tuple(f"ports_{sede}" for sede in SEDES)


@dataclass
class Contexto:
    """Insumos preparados una sola vez y compartidos por todas las corridas."""

    config: ModelConfig
    df_inputs: pd.DataFrame
    tmat: pd.DataFrame
    target: np.ndarray
    coeffs: dict
    vector_vulnerabilidad: np.ndarray
    mascara_alta_vulnerabilidad: np.ndarray  # cuartil superior (evaluación, como en el paper)


def preparar_contexto(dir_salidas: Path, config: ModelConfig | None = None) -> Contexto:
    """Carga los insumos con el mismo camino de código que el pipeline oficial.

    `dir_salidas` recibe los artefactos auxiliares que `load_minimal_inputs`
    escribe (shapefile descomprimido, input_metadata.json); así no se toca
    `outputs/` del paper.
    """
    config = config or ModelConfig()
    dir_salidas.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        root=RAIZ,
        input_dir=RAIZ / "data" / "inputs",
        output_dir=dir_salidas,
        experiment_dir=dir_salidas,
        figures_dir=dir_salidas / "figuras",
        tables_dir=dir_salidas / "tablas",
        reference_dir=None,
    )
    inputs = InputFiles.from_dir(paths.input_dir)
    _utam, df_inputs, _vuln, _att, tmat, _loss, _tm_alt = load_minimal_inputs(paths, inputs, config)
    _pi, target = make_targets(df_inputs, config)
    coeffs = build_access_coeffs(df_inputs, tmat, config.lambda_v, config)
    vector = df_inputs["vulnerability_index"].to_numpy(dtype=float)
    mascara = vector >= np.quantile(vector, 1.0 - config.vulnerability_top_share)
    return Contexto(
        config=config,
        df_inputs=df_inputs,
        tmat=tmat,
        target=target,
        coeffs=coeffs,
        vector_vulnerabilidad=vector,
        mascara_alta_vulnerabilidad=mascara,
    )


def leer_u_star() -> float:
    """u* de la corrida oficial: fila S0 de table_priority_summary.csv.

    En `summarize_scenario` la métrica `objective_abs_dev` de S0 es exactamente
    el óptimo de la etapa lexicográfica (suma de desviaciones absolutas), así
    que reutilizarla evita re-lanzar S0 en NEOS.
    """
    ruta = TABLAS_PAPER / "table_priority_summary.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Sin la tabla archivada del paper no hay u* de referencia; "
            "use --recalcular-s0 (una corrida NEOS adicional) o restaure la tabla."
        )
    df = pd.read_csv(ruta)
    fila = df.loc[df["scenario"].astype(str).eq("S0")]
    if fila.empty:
        raise RuntimeError(f"{ruta} no contiene la fila S0.")
    return float(fila["objective_abs_dev"].iloc[0])


def leer_asignacion_paper(escenario: str) -> pd.DataFrame:
    """Asignación por UTAM de la corrida oficial (p. ej. 'S1', 'S2')."""
    ruta = TABLAS_PAPER / f"scenario_{escenario}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}; se necesita como referencia de comparación. "
            "Restaure las tablas archivadas del paper antes de lanzar."
        )
    return pd.read_csv(ruta)


def leer_metricas_paper() -> pd.DataFrame:
    """Métricas por escenario de la corrida oficial (contexto del barrido)."""
    ruta = TABLAS_PAPER / "table_equity_metrics_final.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}.")
    return pd.read_csv(ruta)


def comparar_asignaciones(df_nueva: pd.DataFrame, df_referencia: pd.DataFrame) -> dict[str, float]:
    """Cuenta zonas cambiadas y puertos movidos frente a una asignación de referencia.

    - zonas_cambiadas_total: UTAM cuyo total de puertos difiere.
    - zonas_cambiadas_sede: UTAM donde difiere la asignación de al menos una sede.
    - puertos_movidos: 0.5 * suma de |delta total| (los totales coinciden por cuotas,
      así que cada puerto movido se cuenta una vez).
    """
    columnas = ["UTAM", "ports_total", *COLUMNAS_PUERTOS]
    izq = df_nueva[columnas].copy()
    der = df_referencia[columnas].copy()
    junto = izq.merge(der, on="UTAM", how="inner", suffixes=("_nueva", "_ref"))
    if len(junto) != len(izq) or len(junto) != len(der):
        raise RuntimeError(
            "Las asignaciones a comparar no cubren el mismo conjunto de UTAM "
            f"({len(izq)} vs {len(der)}, intersección {len(junto)})."
        )
    delta_total = junto["ports_total_nueva"] - junto["ports_total_ref"]
    cambiada_sede = np.zeros(len(junto), dtype=bool)
    for col in COLUMNAS_PUERTOS:
        cambiada_sede |= (junto[f"{col}_nueva"] != junto[f"{col}_ref"]).to_numpy()
    return {
        "zonas_cambiadas_total": int((delta_total != 0).sum()),
        "zonas_cambiadas_sede": int(cambiada_sede.sum()),
        "puertos_movidos": float(np.abs(delta_total).sum() / 2.0),
    }


def calcular_accesibilidad(n_sol: np.ndarray, coeffs: dict, config: ModelConfig) -> np.ndarray:
    """A_i implicada por una asignación entera (mismos coeficientes del modelo)."""
    n_zonas = n_sol.shape[0]
    acceso = np.zeros(n_zonas, dtype=float)
    for i in range(n_zonas):
        acceso[i] = sum(
            coef * n_sol[j, indice_sede]
            for indice_sede, sede in enumerate(config.venues)
            for (j, coef) in coeffs[sede][i]
        )
    return acceso


def email_neos(config: ModelConfig) -> str:
    """Correo que usará NEOS: variable de entorno o el valor por defecto del config."""
    valor = os.environ.get("NEOS_EMAIL", "")
    if valor and "tu_correo@" not in valor:
        return valor
    return config.neos_email


def redondear6(valor: float) -> float:
    """Redondeo a 6 decimales para las columnas de Gini pedidas por R1.13/R1.14."""
    return float(np.round(float(valor), 6))
