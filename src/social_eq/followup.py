from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverFactory, SolverManagerFactory, TerminationCondition
from scipy.sparse.csgraph import shortest_path
from shapely.affinity import rotate as rotate_geometry
from shapely.geometry import box as shapely_box

warnings.filterwarnings("ignore")
plt.style.use("default")
plt.rcParams["hatch.linewidth"] = 0.5
# Tipografia comun de las figuras (guia JTG, especificacion_figuras.md 1.5): Arial con respaldos
# libres de metrica equivalente; rotulos de 7-8 pt a tamano impreso.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Arimo", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["pdf.fonttype"] = 42

PAPER_PANEL = "#fbf8f4"
PAPER_GRID = "#d9d1c8"
PAPER_TEXT = "#2d2a26"
MAP_EDGE = "#b9b0a3"
MAP_MISSING = "#d9d9d9"
MAP_OUTLINE = "#8a8178"
MAP_SCALE = "#4f4a44"
MAP_DELTA_ZERO = "#e8e5e0"
MAP_DELTA_EDGE = "#c6bdb1"
# Sequential ramp (R1.11): single warm hue family, LIGHT low end -> dark high end,
# monotone in lightness (ColorBrewer YlOrBr-style), so low values stay readable.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "social_eq_seq",
    ["#fff6d0", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#5c2301"],
)
# Sequential ramp for provision intensity (ports per 10,000 residents): one hue, light -> dark,
# distinct from the port-count ramp so the two scales in Figure 6 are not confused.
INTENSITY_CMAP = LinearSegmentedColormap.from_list(
    "social_eq_intensity",
    ["#eaf3f8", "#c6dfec", "#93c1d8", "#5b9ec0", "#2f7aa3", "#1a5a82", "#0e3d5e"],
)
# Nombres de localidades y municipios con su ortografía (el shapefile los trae en mayúsculas sin tilde).
PLACE_NAMES = {
    "USAQUEN": "Usaquén", "ENGATIVA": "Engativá", "FONTIBON": "Fontibón", "CIUDAD BOLIVAR": "Ciudad Bolívar",
    "SAN CRISTOBAL": "San Cristóbal", "CHIA": "Chía", "ZIPAQUIRA": "Zipaquirá", "ANTONIO NARIÑO": "Antonio Nariño",
    "LOS MARTIRES": "Los Mártires", "CANDELARIA": "La Candelaria", "TUNJUELITO": "Tunjuelito", "PUENTE ARANDA": "Puente Aranda",
    "RAFAEL URIBE URIBE": "Rafael Uribe Uribe", "BARRIOS UNIDOS": "Barrios Unidos", "TEUSAQUILLO": "Teusaquillo",
    "SANTA FE": "Santa Fe", "USME": "Usme", "SUBA": "Suba", "BOSA": "Bosa", "KENNEDY": "Kennedy", "CHAPINERO": "Chapinero",
    "SUMAPAZ": "Sumapaz", "SOACHA": "Soacha", "MOSQUERA": "Mosquera", "LA CALERA": "La Calera", "COTA": "Cota",
}


def place_name(raw: str) -> str:
    """Nombre legible de una localidad o municipio a partir del código en mayúsculas."""
    key = str(raw).upper().strip()
    return PLACE_NAMES.get(key, key.title())


# Diverging ramp: two hues around a NEUTRAL midpoint (losses blue, gains warm red).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "social_eq_div",
    ["#1d5a8f", "#7fb3d3", "#f0efec", "#efa583", "#a63a2e"],
)
# Discrete fills used when the delta is effectively categorical (gain/loss maps).
GAIN_FILL = "#D6604D"
LOSS_FILL = "#4393C3"
# Three-class urban/rural/region layer (author decision 9, revision R1 2026-08).
# Classified from the UTAM code alone: UPR* = rural Bogota, numeric >= 500 =
# neighbouring municipalities, numeric < 500 = urban Bogota.
ZONE_FILLS = {
    "urban": "#f0e8d9",
    "rural": "#dfe7d2",
    "region": "#e7e4df",
}
ZONE_LEGEND_LABELS = {
    "urban": "Urban Bogota (111 UTAM)",
    "rural": "Rural Bogota (5 UPR units)",
    "region": "Neighbouring municipalities (26 units)",
}
REGION_HATCH = "///"
REGION_HATCH_COLOR = "#8d8478"
# Localities / municipalities labelled on the maps (LOCNombre / MUNNombre in the
# UTAM2023 shapefile). Kept to the major administrative units to avoid clutter.
MAJOR_LOCALITIES = (
    "SUBA",
    "USAQUEN",
    "ENGATIVA",
    "FONTIBON",
    "KENNEDY",
    "BOSA",
    "CIUDAD BOLIVAR",
    "USME",
    "SAN CRISTOBAL",
    "CHAPINERO",
    "SUMAPAZ",
)
MAJOR_MUNICIPALITIES = (
    "SOACHA",
    "CHIA",
    "MOSQUERA",
    "LA CALERA",
    "ZIPAQUIRA",
)
SCENARIO_PANEL_TITLES = {
    "S0": "Priority-matching baseline",
    "S1": "Territorial equalization",
    "S2": "Socially weighted equalization",
    "S3": "Soft-floor protection",
}
# Paleta de escenarios (Okabe-Ito), RESERVADA a S0-S3: ninguna otra serie la reutiliza.
SCENARIO_COLORS = {
    "S0": "#0072B2",
    "S1": "#009E73",
    "S2": "#D55E00",
    "S3": "#CC79A7",
}
SCENARIO_MARKERS = {"S0": "o", "S1": "s", "S2": "^", "S3": "D"}
SCENARIO_LINESTYLES = {"S0": "-", "S1": (0, (5, 2)), "S2": (0, (1.2, 1.2)), "S3": (0, (6, 2, 1.2, 2))}
# Series que no son escenarios: grises distinguibles tambien por forma o tipo de linea.
NEUTRAL_DARK = "#1B1B1B"
NEUTRAL_MID = "#7F7F7F"
NEUTRAL_LIGHT = "#BDBDBD"
# Paletas clasificadas de los mapas (estilo_cartografico_comun.md, 3).
PORTS_COLORS = ["#FFFFD4", "#FEE391", "#FEC44F", "#FE9929", "#D95F0E", "#993404"]
PORTS_CMAP = ListedColormap(PORTS_COLORS, name="ports_ylorbr6")
INDEX_COLORS = ["#FFFFD4", "#FED98E", "#FE9929", "#D95F0E", "#993404"]
INDEX_CMAP = ListedColormap(INDEX_COLORS, name="index_ylorbr5")
INTENSITY_COLORS = ["#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]
INTENSITY_CLASS_CMAP = ListedColormap(INTENSITY_COLORS, name="intensity_blues7")
# Cambio entero de puertos entre escenarios, clases -2..+2 (RdBu, neutro en cero).
DELTA_CLASS_COLORS = {-2: "#053061", -1: "#4393C3", 0: "#F0F0F0", 1: "#D6604D", 2: "#67001F"}
NO_PORTS_FILL = "#FFFFFF"
NEATLINE_COLOR = "#7F7F7F"
# Marco urbano comun, EPSG:3116, norte arriba (30 x 47.5 km; 111 UTAM urbanas con >= 3 km de margen).
URBAN_FRAME_3116 = (980500.0, 981500.0, 1010500.0, 1029000.0)
PROTECTED_COLORS = {0: "#F0F0F0", 1: "#B2182B"}
TIME_WINDOWS = [
    (-0.5, 5.5, "Residential", "#F2F2F2"),
    (7.5, 17.5, "Work", "#E6E6E6"),
    (17.5, 21.5, "Public", "#D9D9D9"),
    # La ventana residencial es 22-5 h: las horas 22 y 23 llevan el mismo tono, sin rótulo.
    (21.5, 23.5, "", "#F2F2F2"),
]
PAPER_MAIN_FIGURES = (
    "fig_official_ev_growth.png",
    "fig_social_context_panels.png",
    "fig_attention_profile.png",
    "fig_allocation_composite.png",
    "fig_net_change_S2_minus_S1.png",
    "fig_lorenz_curves.png",
    "fig_policy_tradeoff.png",
)
PAPER_SUPPLEMENTARY_FIGURES = (
    "fig_growth_progression.png",
    "fig_gnn_learning_history.png",
    "fig_robustness.png",
)
PAPER_KEEP_FIGURES = PAPER_MAIN_FIGURES + PAPER_SUPPLEMENTARY_FIGURES
PAPER_MAIN_TABLES = (
    "table_scenario_design.csv",
    "table_equity_metrics_final.csv",
    "table_official_ev_growth.csv",
    "table_policy_tradeoff.csv",
)
PAPER_SUPPLEMENTARY_TABLES = (
    "table_priority_summary.csv",
    "table_equity_metrics.csv",
    "table_robustness.csv",
    "table_allocation_changes_S2_vs_S1.csv",
    "table_gnn_performance_available.csv",
    "scenario_S0.csv",
    "scenario_S1.csv",
    "scenario_S2.csv",
    "scenario_S3.csv",
)
PAPER_KEEP_TABLES = PAPER_MAIN_TABLES + PAPER_SUPPLEMENTARY_TABLES


@dataclass
class ModelConfig:
    venues: tuple[str, ...] = ("residential", "workplace", "public")
    quotas: dict[str, int] = field(
        default_factory=lambda: {
            "residential": 7352,
            "workplace": 2739,
            "public": 779,
        }
    )
    lambda_v: dict[str, float] = field(
        default_factory=lambda: {
            "residential": 15.0,
            "workplace": 25.0,
            "public": 10.0,
        }
    )
    epsilon_relax: float = 0.025
    vulnerability_top_share: float = 0.25
    coverage_threshold_min: float = 15.0
    kernel_eps: float = 1e-8
    use_osrm_if_survey_incomplete: bool = False
    solver_backend: str = "neos"
    solver_name: str = "cplex"
    neos_email: str = "jorgei.romerog@utadeo.edu.co"
    solver_time_limit: int = 3600
    solver_mip_gap: float = 0.01
    solver_threads: int | None = None
    epsilon_s1: float = 0.03
    epsilon_s2: float = 0.08
    epsilon_s3: float = 0.08
    s2_kappa: float = 4.0
    s3_protected_share: float = 0.15
    s3_floor_quantile: float = 0.10
    s3_floor_gamma: float = 0.95
    s3_shortfall_penalty: float = 100.0
    robustness_lambda_factors_fast: tuple[float, ...] = (0.75, 1.25)
    robustness_floor_gammas_fast: tuple[float, ...] = (0.65, 0.80, 0.95)
    robustness_time_limit_sec: int = 180
    robustness_mip_gap: float = 0.02
    max_ports_matrix: np.ndarray | None = None

    @property
    def robustness_epsilon_fast(self) -> list[float]:
        return sorted(
            {
                max(0.01, self.epsilon_s2 * 0.70),
                self.epsilon_s2,
                min(0.10, self.epsilon_s2 * 1.40),
            }
        )

    @property
    def robustness_kappa_fast(self) -> list[float]:
        return sorted(
            {
                max(1.0, self.s2_kappa * 0.50),
                self.s2_kappa,
                self.s2_kappa * 1.50,
            }
        )


@dataclass
class InputFiles:
    utam_zip: Path
    priority_csv: Path
    attention_csv: Path
    vulnerability_csv: Path
    travel_matrix_csv: Path
    loss_history_csv: Path | None = None
    alt_travel_matrix_csv: Path | None = None

    @classmethod
    def from_dir(cls, input_dir: Path) -> "InputFiles":
        return cls(
            utam_zip=input_dir / "utam_shapefile.zip",
            priority_csv=input_dir / "priority_utam_scores.csv",
            attention_csv=input_dir / "temporal_attention_by_utam.csv",
            vulnerability_csv=input_dir / "utam_social_vulnerability.csv",
            travel_matrix_csv=input_dir / "utam_travel_time_matrix.csv",
            loss_history_csv=(input_dir / "learning_loss_history.csv"),
            alt_travel_matrix_csv=(input_dir / "utam_travel_time_matrix_osrm.csv"),
        )

    def validate(self) -> None:
        required = {
            "utam_zip": self.utam_zip,
            "priority_csv": self.priority_csv,
            "attention_csv": self.attention_csv,
            "vulnerability_csv": self.vulnerability_csv,
            "travel_matrix_csv": self.travel_matrix_csv,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            missing_text = ", ".join(f"{name}={required[name]}" for name in missing)
            raise FileNotFoundError(f"Missing required inputs: {missing_text}")
        if self.loss_history_csv is not None and not self.loss_history_csv.exists():
            self.loss_history_csv = None
        if self.alt_travel_matrix_csv is not None and not self.alt_travel_matrix_csv.exists():
            self.alt_travel_matrix_csv = None


@dataclass
class ProjectPaths:
    root: Path
    input_dir: Path
    output_dir: Path
    experiment_dir: Path
    figures_dir: Path
    tables_dir: Path
    reference_dir: Path | None = None

    @classmethod
    def from_root(
        cls,
        root: Path,
        input_subdir: str = "data/inputs",
        output_subdir: str = "outputs/social_equity_followup_outputs",
        reference_subdir: str | None = "data/reference/paper_v6_experiments",
    ) -> "ProjectPaths":
        input_dir = root / input_subdir
        output_dir = root / output_subdir
        experiment_dir = output_dir / "paper_v6_experiments"
        figures_dir = experiment_dir / "figures"
        tables_dir = experiment_dir / "tables"
        reference_dir = root / reference_subdir if reference_subdir else None
        return cls(
            root=root,
            input_dir=input_dir,
            output_dir=output_dir,
            experiment_dir=experiment_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            reference_dir=reference_dir,
        )

    def ensure_dirs(self) -> None:
        for directory in [
            self.input_dir,
            self.output_dir,
            self.experiment_dir,
            self.figures_dir,
            self.tables_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioRun:
    scenario_metrics: pd.DataFrame
    scenario_outputs: dict[str, pd.DataFrame]
    df_inputs: pd.DataFrame
    utam: gpd.GeoDataFrame
    loss_hist: pd.DataFrame | None
    tmat: pd.DataFrame
    robust_df: pd.DataFrame
    protected_mask_s3: np.ndarray
    reference_comparison: dict[str, Any] | None = None


def normalize01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mn, mx = values.min(skipna=True), values.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mn) / (mx - mn)


def normalize_zone_code(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    match = pd.Series([text]).str.extract(r"^(UTAM|UPR)0*(\d+)$").iloc[0]
    if match.notna().all():
        return f"{match[0]}{int(match[1])}"
    return text


def unzip_first_shapefile(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    shp_files = list(out_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefile found inside {zip_path}")
    shp_files = sorted(shp_files, key=lambda item: ("utam2023" not in item.name.lower(), len(str(item))))
    return shp_files[0]


def find_utam_col(df: pd.DataFrame) -> str:
    exact = ["UTAM", "utam", "id_utam", "cod_utam", "utam_id", "codigo_utam", "codigo"]
    for column in df.columns:
        if str(column).strip() in exact:
            return str(column)
    for column in df.columns:
        if "utam" in str(column).lower():
            return str(column)
    raise KeyError(f"Could not detect UTAM column in {list(df.columns)}")


def build_venue_weights(attention: pd.DataFrame) -> pd.DataFrame:
    att = attention.copy()
    ucol = find_utam_col(att)
    att = att.rename(columns={ucol: "UTAM"})
    att["UTAM"] = att["UTAM"].map(normalize_zone_code)
    res_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    work_hours = list(range(8, 18))
    pub_hours = [18, 19, 20, 21]
    att["residential"] = att[[f"hour_{hour}" for hour in res_hours if f"hour_{hour}" in att.columns]].sum(axis=1)
    att["workplace"] = att[[f"hour_{hour}" for hour in work_hours if f"hour_{hour}" in att.columns]].sum(axis=1)
    att["public"] = att[[f"hour_{hour}" for hour in pub_hours if f"hour_{hour}" in att.columns]].sum(axis=1)
    return att[["UTAM", "residential", "workplace", "public"]].copy()


def detect_time_cols(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    columns = list(df.columns)
    origin = destination = timec = None
    for column in columns:
        lower = str(column).lower()
        if origin is None and any(token in lower for token in ["origin_utam", "origen_utam", "otaz", "origen", "zona_origen"]):
            origin = str(column)
        if destination is None and any(token in lower for token in ["destination_utam", "destino_utam", "dtaz", "destino", "zona_destino"]):
            destination = str(column)
        if timec is None and any(token in lower for token in ["travel_time_min", "tiempo", "duration", "duracion", "mean_time_min", "time_min"]):
            timec = str(column)
    if origin is None or destination is None or timec is None:
        for column in columns:
            lower = str(column).lower()
            if origin is None and ("orig" in lower or "origen" in lower):
                origin = str(column)
            if destination is None and "dest" in lower:
                destination = str(column)
            if timec is None and any(token in lower for token in ["time", "tiempo", "dur", "min"]):
                timec = str(column)
    return origin, destination, timec


def make_square_time_matrix(tm_df: pd.DataFrame, utam_list: list[str]) -> tuple[pd.DataFrame, float]:
    origin_col, destination_col, time_col = detect_time_cols(tm_df)
    if any(col is None for col in [origin_col, destination_col, time_col]):
        raise RuntimeError(f"Could not detect origin/destination/time columns in matrix: {tm_df.columns.tolist()}")

    data = tm_df[[origin_col, destination_col, time_col]].copy()
    data.columns = ["origin_utam", "destination_utam", "travel_time_min"]
    data["origin_utam"] = data["origin_utam"].map(normalize_zone_code)
    data["destination_utam"] = data["destination_utam"].map(normalize_zone_code)
    data["travel_time_min"] = pd.to_numeric(data["travel_time_min"], errors="coerce")
    data = data.dropna().copy()
    data = data.groupby(["origin_utam", "destination_utam"], as_index=False)["travel_time_min"].median()

    pivot = data.pivot(index="origin_utam", columns="destination_utam", values="travel_time_min")
    pivot = pivot.reindex(index=utam_list, columns=utam_list)
    for utam in utam_list:
        pivot.loc[utam, utam] = 0.0
    pivot = pivot.combine_first(pivot.T)
    coverage = float(np.isfinite(pivot.to_numpy(dtype=float)).mean())
    return pivot.astype(float), coverage


def complete_time_matrix_with_graph_closure(pivot: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    values = pivot.to_numpy(dtype=float)
    symmetric = np.where(
        np.isfinite(values) & np.isfinite(values.T),
        np.minimum(values, values.T),
        np.where(np.isfinite(values), values, values.T),
    )
    np.fill_diagonal(symmetric, 0.0)
    graph = np.where(np.isfinite(symmetric), symmetric, np.inf)
    sp = shortest_path(graph, directed=False)
    completed = values.copy()
    missing_mask = ~np.isfinite(completed)
    completed[missing_mask] = sp[missing_mask]
    np.fill_diagonal(completed, 0.0)
    completed_df = pd.DataFrame(completed, index=pivot.index, columns=pivot.columns)
    coverage = float(np.isfinite(completed_df.to_numpy(dtype=float)).mean())
    return completed_df, coverage


def gini(x: pd.Series | np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    values = np.maximum(values, 0)
    if values.size == 0 or values.sum() <= 0:
        return float("nan")
    values = np.sort(values)
    n = values.size
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def weighted_gini(x: pd.Series | np.ndarray, w: pd.Series | np.ndarray) -> float:
    x_values = np.asarray(x, dtype=float)
    w_values = np.asarray(w, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(w_values) & (w_values > 0)
    x_values, w_values = x_values[mask], w_values[mask]
    if x_values.size == 0 or np.sum(w_values) <= 0 or np.sum(x_values * w_values) <= 0:
        return float("nan")
    total_weight = w_values.sum()
    mu = np.sum(w_values * x_values) / total_weight
    numerator = 0.0
    for index in range(len(x_values)):
        numerator += np.sum(w_values[index] * w_values * np.abs(x_values[index] - x_values))
    return float(numerator / (2 * mu * total_weight * total_weight))


def lorenz_curve(x: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=float)
    values = np.maximum(values, 0)
    values = np.sort(values)
    if values.size == 0 or values.sum() <= 0:
        return np.array([0, 1]), np.array([0, 1])
    cumulative = np.cumsum(values)
    return np.linspace(0, 1, values.size + 1), np.r_[0, cumulative / cumulative[-1]]


def weighted_lorenz(x: pd.Series | np.ndarray, w: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    w_values = np.asarray(w, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(w_values) & (w_values > 0)
    x_values, w_values = x_values[mask], w_values[mask]
    order = np.argsort(x_values)
    x_values, w_values = x_values[order], w_values[order]
    cumulative_weight = np.cumsum(w_values)
    cumulative_xw = np.cumsum(x_values * w_values)
    if cumulative_weight.size == 0 or cumulative_xw[-1] <= 0:
        return np.array([0, 1]), np.array([0, 1])
    return np.r_[0, cumulative_weight / cumulative_weight[-1]], np.r_[0, cumulative_xw / cumulative_xw[-1]]


def coverage_within_threshold(
    n_matrix: np.ndarray,
    tmat: pd.DataFrame,
    high_mask: np.ndarray,
    threshold: float,
) -> float:
    reachable = []
    total_ports = n_matrix.sum(axis=1)
    for idx in range(len(high_mask)):
        if not high_mask[idx]:
            continue
        ok = ((tmat.iloc[idx].to_numpy(dtype=float) <= threshold) & (total_ports > 0)).any()
        reachable.append(float(ok))
    return float(np.mean(reachable)) if reachable else float("nan")


def build_inputs(
    priority: pd.DataFrame,
    venue: pd.DataFrame,
    vulnerability: pd.DataFrame,
    utam_list: list[str],
    config: ModelConfig,
) -> pd.DataFrame:
    base = pd.DataFrame({"UTAM": utam_list})
    base = base.merge(priority[["UTAM", "priority_norm"]], on="UTAM", how="left")
    base = base.merge(venue, on="UTAM", how="left")
    base = base.merge(vulnerability[["UTAM", "vulnerability_index"]], on="UTAM", how="left")
    base["priority_norm"] = pd.to_numeric(base["priority_norm"], errors="coerce").fillna(0.0)
    for venue_name in config.venues:
        base[venue_name] = pd.to_numeric(base[venue_name], errors="coerce")
    venue_sum = base[list(config.venues)].sum(axis=1)
    for venue_name in config.venues:
        base[venue_name] = np.where(venue_sum > 0, base[venue_name] / venue_sum, 1 / len(config.venues))
    base["vulnerability_index"] = normalize01(
        base["vulnerability_index"].fillna(base["vulnerability_index"].median())
    )
    return base


def make_targets(df_inputs: pd.DataFrame, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    score = df_inputs["priority_norm"].to_numpy(dtype=float)
    score = np.maximum(score, 1e-9)
    venue_weights = {venue: df_inputs[venue].to_numpy(dtype=float) for venue in config.venues}
    pi = np.zeros((len(df_inputs), len(config.venues)), dtype=float)
    target = np.zeros_like(pi)
    for venue_index, venue_name in enumerate(config.venues):
        shares = score * venue_weights[venue_name]
        denom = shares.sum()
        shares = np.ones_like(shares) / len(shares) if denom <= 0 else shares / denom
        pi[:, venue_index] = shares
        target[:, venue_index] = shares * config.quotas[venue_name]
    return pi, target


def build_access_coeffs(
    df_inputs: pd.DataFrame,
    tmat: pd.DataFrame,
    lambda_v: dict[str, float],
    config: ModelConfig,
) -> dict[str, list[list[tuple[int, float]]]]:
    n = len(df_inputs)
    coeffs = {venue: [[] for _ in range(n)] for venue in config.venues}
    for i in range(n):
        for j in range(n):
            tij = float(tmat.iloc[i, j])
            if not np.isfinite(tij):
                continue
            for venue_name in config.venues:
                coef = float(df_inputs.iloc[i][venue_name]) * math.exp(-tij / float(lambda_v[venue_name]))
                if coef > config.kernel_eps:
                    coeffs[venue_name][i].append((j, coef))
    return coeffs


def solve_model(
    model: ConcreteModel,
    config: ModelConfig,
    tee: bool = False,
    time_limit: int | None = None,
    mip_gap: float | None = None,
) -> Any:
    options: dict[str, Any] = {}
    if config.solver_backend == "neos":
        if not os.environ.get("NEOS_EMAIL") or "tu_correo@" in os.environ.get("NEOS_EMAIL", ""):
            os.environ["NEOS_EMAIL"] = config.neos_email
        if time_limit is not None:
            options["timelimit"] = int(time_limit)
        elif config.solver_time_limit is not None:
            options["timelimit"] = int(config.solver_time_limit)
        if mip_gap is not None:
            options["mipgap"] = float(mip_gap)
        elif config.solver_mip_gap is not None:
            options["mipgap"] = float(config.solver_mip_gap)
        if config.solver_threads is not None:
            options["threads"] = int(config.solver_threads)

        solver_manager = SolverManagerFactory("neos")
        if solver_manager is None:
            raise RuntimeError("NEOS solver manager is not available in this Pyomo installation.")
        print(f"Submitting model to NEOS using solver={config.solver_name}", flush=True)
        results = solver_manager.solve(
            model,
            solver=config.solver_name,
            tee=tee,
            load_solutions=True,
            options=options,
        )
    else:
        solver = SolverFactory(config.solver_name)
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        elif config.solver_time_limit is not None:
            options["time_limit"] = float(config.solver_time_limit)
        if mip_gap is not None:
            options["mip_rel_gap"] = float(mip_gap)
        elif config.solver_mip_gap is not None:
            options["mip_rel_gap"] = float(config.solver_mip_gap)
        if config.solver_threads is not None:
            options["threads"] = int(config.solver_threads)
        results = solver.solve(model, tee=tee, load_solutions=True, options=options)

    model._solver_status = getattr(results.solver, "status", None)
    model._termination_condition = getattr(results.solver, "termination_condition", None)
    model._solver_results = results
    return results


def solve_safe(model: ConcreteModel, accept_feasible: bool = False) -> bool:
    tc = getattr(model, "_termination_condition", None)
    if tc == TerminationCondition.optimal:
        return True
    if accept_feasible and tc == TerminationCondition.feasible:
        return True
    if accept_feasible and tc in {
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.maxEvaluations,
        TerminationCondition.other,
    }:
        try:
            _ = value(next(model.component_data_objects(Var, active=True)))
            return True
        except Exception:
            return False
    return False


def extract_solution(model: ConcreteModel, n: int, n_venues: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_sol = np.zeros((n, n_venues), dtype=int)
    u_sol = np.zeros((n, n_venues), dtype=float)
    a_sol = np.zeros(n, dtype=float) if hasattr(model, "A") else None
    for i in range(n):
        for venue_index in range(n_venues):
            n_sol[i, venue_index] = int(round(value(model.n[i, venue_index])))
            u_sol[i, venue_index] = float(value(model.u[i, venue_index]))
        if a_sol is not None:
            a_sol[i] = float(value(model.A[i]))
    return n_sol, u_sol, a_sol


def extract_solution_refined(
    model: ConcreteModel,
    n: int,
    n_venues: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    n_sol, u_sol, a_sol = extract_solution(model, n, n_venues)
    d_sol = np.array([float(value(model.d[k])) for k in model.H], dtype=float) if hasattr(model, "d") else None
    return n_sol, u_sol, a_sol, d_sol


def solve_s0(target: np.ndarray, config: ModelConfig) -> tuple[ConcreteModel, float]:
    n, n_venues = target.shape
    model = ConcreteModel()
    model.I = RangeSet(0, n - 1)
    model.V = RangeSet(0, n_venues - 1)
    model.n = Var(model.I, model.V, domain=NonNegativeIntegers)
    model.u = Var(model.I, model.V, domain=NonNegativeReals)
    model.quota = Constraint(
        model.V,
        rule=lambda mdl, venue_index: sum(mdl.n[i, venue_index] for i in mdl.I)
        == int(config.quotas[config.venues[venue_index]]),
    )
    model.abs1 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= mdl.n[i, venue_index] - float(target[i, venue_index]),
    )
    model.abs2 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= float(target[i, venue_index]) - mdl.n[i, venue_index],
    )
    model.obj = Objective(expr=sum(model.u[i, venue_index] for i in model.I for venue_index in model.V), sense=minimize)
    solve_model(model, config=config, tee=False)
    return model, float(value(model.obj))


def solve_equity(
    target: np.ndarray,
    coeffs: dict[str, list[list[tuple[int, float]]]],
    u_star: float,
    config: ModelConfig,
    nu: np.ndarray | None = None,
    floor_set: np.ndarray | None = None,
    floor_value: float | None = None,
    epsilon: float | None = None,
) -> ConcreteModel:
    epsilon_value = config.epsilon_relax if epsilon is None else float(epsilon)
    n, n_venues = target.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_weights = np.ones(len(pairs), dtype=float)
    if nu is not None:
        pair_weights = np.array([1.0 + (nu[i] + nu[j]) / 2.0 for i, j in pairs], dtype=float)

    model = ConcreteModel()
    model.I = RangeSet(0, n - 1)
    model.V = RangeSet(0, n_venues - 1)
    model.P = RangeSet(0, len(pairs) - 1)
    model.n = Var(model.I, model.V, domain=NonNegativeIntegers)
    model.u = Var(model.I, model.V, domain=NonNegativeReals)
    model.A = Var(model.I, domain=NonNegativeReals)
    model.z = Var(model.P, domain=NonNegativeReals)
    model.quota = Constraint(
        model.V,
        rule=lambda mdl, venue_index: sum(mdl.n[i, venue_index] for i in mdl.I)
        == int(config.quotas[config.venues[venue_index]]),
    )
    model.abs1 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= mdl.n[i, venue_index] - float(target[i, venue_index]),
    )
    model.abs2 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= float(target[i, venue_index]) - mdl.n[i, venue_index],
    )
    model.lex = Constraint(
        expr=sum(model.u[i, venue_index] for i in model.I for venue_index in model.V) <= float(u_star) * (1.0 + epsilon_value)
    )

    def accessibility_rule(mdl: ConcreteModel, i: int) -> Any:
        expr = 0.0
        for venue_index, venue_name in enumerate(config.venues):
            expr += sum(coef * mdl.n[j, venue_index] for j, coef in coeffs[venue_name][i])
        return mdl.A[i] == expr

    model.Adef = Constraint(model.I, rule=accessibility_rule)
    model.g1 = Constraint(model.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pairs[p][0]] - mdl.A[pairs[p][1]])
    model.g2 = Constraint(model.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pairs[p][1]] - mdl.A[pairs[p][0]])

    if floor_set is not None and floor_value is not None:
        floor_indices = [idx for idx, flag in enumerate(floor_set) if flag]
        if floor_indices:
            model.floorI = RangeSet(0, len(floor_indices) - 1)
            model.floor = Constraint(model.floorI, rule=lambda mdl, k: mdl.A[floor_indices[k]] >= float(floor_value))

    model.obj = Objective(expr=sum(float(pair_weights[p]) * model.z[p] for p in model.P), sense=minimize)
    solve_model(model, config=config, tee=False)
    return model


def solve_equity_refined(
    target: np.ndarray,
    coeffs: dict[str, list[list[tuple[int, float]]]],
    u_star: float,
    config: ModelConfig,
    nu: np.ndarray | None = None,
    epsilon: float = 0.05,
    kappa: float = 4.0,
    floor_mask: np.ndarray | None = None,
    floor_reference: np.ndarray | None = None,
    floor_gamma: float = 0.80,
    floor_quantile: float = 0.10,
    shortfall_penalty: float = 100.0,
    use_hard_floor: bool = False,
    caps: np.ndarray | None = None,
    time_limit: int | None = None,
    mip_gap: float | None = None,
) -> tuple[ConcreteModel, dict[str, Any] | None]:
    n, n_venues = target.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_weights = np.ones(len(pairs), dtype=float)
    if nu is not None:
        pair_weights = np.array(
            [1.0 + float(kappa) * (float(nu[i]) + float(nu[j])) / 2.0 for i, j in pairs],
            dtype=float,
        )

    model = ConcreteModel()
    model.I = RangeSet(0, n - 1)
    model.V = RangeSet(0, n_venues - 1)
    model.P = RangeSet(0, len(pairs) - 1)
    model.n = Var(model.I, model.V, domain=NonNegativeIntegers)
    model.u = Var(model.I, model.V, domain=NonNegativeReals)
    model.A = Var(model.I, domain=NonNegativeReals)
    model.z = Var(model.P, domain=NonNegativeReals)

    model.quota = Constraint(
        model.V,
        rule=lambda mdl, venue_index: sum(mdl.n[i, venue_index] for i in mdl.I)
        == int(config.quotas[config.venues[venue_index]]),
    )
    if caps is not None:
        model.cap = Constraint(model.I, model.V, rule=lambda mdl, i, venue_index: mdl.n[i, venue_index] <= float(caps[i, venue_index]))
    model.abs1 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= mdl.n[i, venue_index] - float(target[i, venue_index]),
    )
    model.abs2 = Constraint(
        model.I,
        model.V,
        rule=lambda mdl, i, venue_index: mdl.u[i, venue_index] >= float(target[i, venue_index]) - mdl.n[i, venue_index],
    )
    model.lex = Constraint(
        expr=sum(model.u[i, venue_index] for i in model.I for venue_index in model.V) <= float(u_star) * (1.0 + float(epsilon))
    )

    def accessibility_rule(mdl: ConcreteModel, i: int) -> Any:
        expr = 0.0
        for venue_index, venue_name in enumerate(config.venues):
            expr += sum(coef * mdl.n[j, venue_index] for j, coef in coeffs[venue_name][i])
        return mdl.A[i] == expr

    model.Adef = Constraint(model.I, rule=accessibility_rule)
    model.g1 = Constraint(model.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pairs[p][0]] - mdl.A[pairs[p][1]])
    model.g2 = Constraint(model.P, rule=lambda mdl, p: mdl.z[p] >= mdl.A[pairs[p][1]] - mdl.A[pairs[p][0]])

    floor_targets = None
    protected_indices: list[int] = []
    if floor_mask is not None and floor_reference is not None:
        protected_indices = [idx for idx, flag in enumerate(floor_mask) if flag]
        if protected_indices:
            base_floor = float(np.quantile(floor_reference[protected_indices], float(floor_quantile)))
            floor_value = float(floor_gamma) * base_floor
            floor_targets = {
                "base_floor": base_floor,
                "floor_value": floor_value,
                "protected_count": len(protected_indices),
                "protected_indices": protected_indices,
            }
            model.H = RangeSet(0, len(protected_indices) - 1)
            if use_hard_floor:
                model.floor = Constraint(model.H, rule=lambda mdl, k: mdl.A[protected_indices[k]] >= floor_value)
            else:
                model.d = Var(model.H, domain=NonNegativeReals)
                model.floor = Constraint(
                    model.H,
                    rule=lambda mdl, k: mdl.A[protected_indices[k]] + mdl.d[k] >= floor_value,
                )

    gap_term = sum(float(pair_weights[p]) * model.z[p] for p in model.P)
    if hasattr(model, "d"):
        shortfall_term = float(shortfall_penalty) * sum(model.d[k] for k in model.H)
        model.obj = Objective(expr=gap_term + shortfall_term, sense=minimize)
    else:
        model.obj = Objective(expr=gap_term, sense=minimize)

    solve_model(model, config=config, tee=False, time_limit=time_limit, mip_gap=mip_gap)
    return model, floor_targets


def summarize_scenario(
    name: str,
    df_inputs: pd.DataFrame,
    n_sol: np.ndarray,
    u_sol: np.ndarray,
    a_sol: np.ndarray,
    tmat: pd.DataFrame,
    high_mask: np.ndarray,
    config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = pd.DataFrame({"UTAM": df_inputs["UTAM"]})
    for venue_index, venue_name in enumerate(config.venues):
        out[f"ports_{venue_name}"] = n_sol[:, venue_index]
    out["ports_total"] = n_sol.sum(axis=1)
    out["priority_norm"] = df_inputs["priority_norm"].to_numpy(dtype=float)
    out["vulnerability_index"] = df_inputs["vulnerability_index"].to_numpy(dtype=float)
    out["accessibility"] = a_sol
    out["high_vulnerability"] = high_mask.astype(int)

    rho_ports = pd.Series(out["priority_norm"]).corr(pd.Series(out["ports_total"]), method="spearman")
    rho_access = pd.Series(out["priority_norm"]).corr(pd.Series(out["accessibility"]), method="spearman")
    weighted_gini_value = weighted_gini(out["accessibility"], 1.0 + out["vulnerability_index"])
    unweighted_gini_value = gini(out["accessibility"])
    top_mask = out["vulnerability_index"] >= out["vulnerability_index"].quantile(1 - config.vulnerability_top_share)
    share_vuln_q = (
        float(out.loc[top_mask, "accessibility"].sum() / out["accessibility"].sum())
        if out["accessibility"].sum() > 0
        else float("nan")
    )
    coverage = coverage_within_threshold(
        n_sol,
        tmat,
        high_mask,
        threshold=config.coverage_threshold_min,
    )
    mean_high = float(out.loc[top_mask, "accessibility"].mean()) if top_mask.any() else float("nan")
    metrics = {
        "scenario": name,
        "rho_priority_ports": float(rho_ports) if pd.notna(rho_ports) else float("nan"),
        "rho_priority_access": float(rho_access) if pd.notna(rho_access) else float("nan"),
        "objective_abs_dev": float(u_sol.sum()),
        "accessibility_gini": float(unweighted_gini_value) if pd.notna(unweighted_gini_value) else float("nan"),
        "accessibility_weighted_gini": float(weighted_gini_value) if pd.notna(weighted_gini_value) else float("nan"),
        "share_access_most_vulnerable_quartile": float(share_vuln_q) if pd.notna(share_vuln_q) else float("nan"),
        "coverage_within_threshold_vulnerable": float(coverage) if pd.notna(coverage) else float("nan"),
        "mean_access_high_vulnerability": float(mean_high) if pd.notna(mean_high) else float("nan"),
    }
    return out, metrics


def summarize_protected_set(
    a_sol: np.ndarray,
    protected_mask: np.ndarray,
    floor_value: float | None = None,
) -> dict[str, float]:
    values = np.asarray(a_sol, dtype=float)
    mask = np.asarray(protected_mask, dtype=bool)
    protected_values = values[mask]
    if protected_values.size == 0:
        return {
            "protected_mean_access": float("nan"),
            "protected_min_access": float("nan"),
            "protected_p10_access": float("nan"),
            "protected_p25_access": float("nan"),
            "protected_count": 0,
            "protected_share_meeting_floor": float("nan"),
        }
    out = {
        "protected_mean_access": float(np.mean(protected_values)),
        "protected_min_access": float(np.min(protected_values)),
        "protected_p10_access": float(np.quantile(protected_values, 0.10)),
        "protected_p25_access": float(np.quantile(protected_values, 0.25)),
        "protected_count": int(protected_values.size),
        "protected_share_meeting_floor": float("nan"),
    }
    if floor_value is not None:
        out["protected_share_meeting_floor"] = float(np.mean(protected_values >= float(floor_value)))
    return out


def load_minimal_inputs(paths: ProjectPaths, inputs: InputFiles, config: ModelConfig) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    inputs.validate()
    priority = pd.read_csv(inputs.priority_csv)
    priority = priority.rename(columns={find_utam_col(priority): "UTAM"})
    priority["UTAM"] = priority["UTAM"].map(normalize_zone_code)

    attention = pd.read_csv(inputs.attention_csv)
    attention = attention.rename(columns={find_utam_col(attention): "UTAM"})
    attention["UTAM"] = attention["UTAM"].map(normalize_zone_code)
    venue = build_venue_weights(attention)

    vulnerability = pd.read_csv(inputs.vulnerability_csv)
    vulnerability = vulnerability.rename(columns={find_utam_col(vulnerability): "UTAM"})
    vulnerability["UTAM"] = vulnerability["UTAM"].map(normalize_zone_code)

    tm_main = pd.read_csv(inputs.travel_matrix_csv)
    tm_alt = pd.read_csv(inputs.alt_travel_matrix_csv) if inputs.alt_travel_matrix_csv is not None else None
    loss_hist = pd.read_csv(inputs.loss_history_csv) if inputs.loss_history_csv is not None else None

    utam_shp = unzip_first_shapefile(inputs.utam_zip, paths.output_dir / "utam_shp")
    utam = gpd.read_file(utam_shp)
    utam_col = find_utam_col(utam)
    utam["UTAM"] = utam[utam_col].map(normalize_zone_code)
    if utam.crs is None:
        utam = utam.set_crs(4326)

    utam_list = utam["UTAM"].astype(str).str.strip().tolist()
    df_inputs = build_inputs(priority, venue, vulnerability, utam_list, config)
    tmat, chosen_name, coverage = ensure_complete_time_matrix(utam_list, tm_main, tm_alt)
    metadata = {
        "tmat_source": chosen_name,
        "tmat_coverage": coverage,
    }
    (paths.output_dir / "input_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return utam, df_inputs, vulnerability, attention, tmat, loss_hist, tm_alt


def ensure_complete_time_matrix(
    utam_list: list[str],
    tm_main: pd.DataFrame | None,
    tm_alt: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, float]:
    chosen_name = ""
    pivot, coverage = None, 0.0
    if tm_main is not None:
        pivot, coverage = make_square_time_matrix(tm_main, utam_list)
        chosen_name = "survey"
    if coverage < 0.95 and tm_alt is not None:
        pivot_alt, coverage_alt = make_square_time_matrix(tm_alt, utam_list)
        if coverage_alt >= coverage:
            pivot, coverage, chosen_name = pivot_alt, coverage_alt, "alternative"
    if pivot is not None and coverage < 0.95:
        completed, completed_coverage = complete_time_matrix_with_graph_closure(pivot)
        if completed_coverage >= coverage:
            pivot, coverage, chosen_name = completed, completed_coverage, f"{chosen_name}_graph_completed"
    if pivot is None or coverage < 0.95:
        raise RuntimeError(f"Travel-time matrix is incomplete (coverage={coverage:.3f}).")
    return pivot.astype(float), chosen_name, coverage


@contextmanager
def temporary_solver_profile(
    config: ModelConfig,
    time_limit: int | None = None,
    mip_gap: float | None = None,
):
    original_time_limit = config.solver_time_limit
    original_mip_gap = config.solver_mip_gap
    if time_limit is not None:
        config.solver_time_limit = time_limit
    if mip_gap is not None:
        config.solver_mip_gap = mip_gap
    try:
        yield
    finally:
        config.solver_time_limit = original_time_limit
        config.solver_mip_gap = original_mip_gap


def run_refined_scenarios(paths: ProjectPaths, inputs: InputFiles, config: ModelConfig) -> ScenarioRun:
    utam, df_inputs, vulnerability, attention, tmat, loss_hist, tm_alt = load_minimal_inputs(paths, inputs, config)
    utam_list = df_inputs["UTAM"].tolist()
    _, target = make_targets(df_inputs, config)
    coeffs = build_access_coeffs(df_inputs, tmat, config.lambda_v, config)
    vulnerability_vector = df_inputs["vulnerability_index"].to_numpy(dtype=float)
    high_mask_eval = vulnerability_vector >= np.quantile(vulnerability_vector, 1.0 - config.vulnerability_top_share)
    protected_mask_s3 = vulnerability_vector >= np.quantile(vulnerability_vector, 1.0 - config.s3_protected_share)

    results_list: list[dict[str, Any]] = []
    scenario_outputs: dict[str, pd.DataFrame] = {}

    print(f"Time matrix source: {(paths.output_dir / 'input_metadata.json')}", flush=True)
    print("Solving S0 via NEOS/CPLEX...", flush=True)
    model_s0, u_star = solve_s0(target, config=config)
    if not solve_safe(model_s0):
        raise RuntimeError(f"S0 did not solve successfully: {getattr(model_s0, '_termination_condition', None)}")
    n0, u0, _ = extract_solution(model_s0, len(df_inputs), len(config.venues))
    a0 = np.zeros(len(df_inputs))
    for i in range(len(df_inputs)):
        a0[i] = sum(
            coef * n0[j, venue_index]
            for venue_index, venue_name in enumerate(config.venues)
            for (j, coef) in coeffs[venue_name][i]
        )
    s0_df, s0_metrics = summarize_scenario("S0", df_inputs, n0, u0, a0, tmat, high_mask_eval, config=config)
    s0_metrics.update({"epsilon_used": 0.0, "kappa_used": float("nan"), "floor_type": "none"})
    results_list.append(s0_metrics)
    scenario_outputs["S0"] = s0_df

    print("Solving S1 via NEOS/CPLEX...", flush=True)
    model_s1 = solve_equity(target, coeffs, u_star, config=config, nu=None, epsilon=max(config.epsilon_relax, config.epsilon_s1))
    if not solve_safe(model_s1):
        raise RuntimeError(f"S1 did not solve successfully: {getattr(model_s1, '_termination_condition', None)}")
    n1, u1, a1 = extract_solution(model_s1, len(df_inputs), len(config.venues))
    assert a1 is not None
    s1_df, s1_metrics = summarize_scenario("S1", df_inputs, n1, u1, a1, tmat, high_mask_eval, config=config)
    s1_metrics.update({"epsilon_used": max(config.epsilon_relax, config.epsilon_s1), "kappa_used": 0.0, "floor_type": "none"})
    results_list.append(s1_metrics)
    scenario_outputs["S1"] = s1_df

    print("Solving S2 via NEOS/CPLEX...", flush=True)
    model_s2, _ = solve_equity_refined(
        target=target,
        coeffs=coeffs,
        u_star=u_star,
        config=config,
        nu=vulnerability_vector,
        epsilon=max(config.epsilon_relax, config.epsilon_s2),
        kappa=config.s2_kappa,
        floor_mask=None,
        floor_reference=None,
        use_hard_floor=False,
        caps=config.max_ports_matrix,
    )
    if not solve_safe(model_s2):
        raise RuntimeError(f"S2 did not solve successfully: {getattr(model_s2, '_termination_condition', None)}")
    n2, u2, a2, _ = extract_solution_refined(model_s2, len(df_inputs), len(config.venues))
    assert a2 is not None
    s2_df, s2_metrics = summarize_scenario("S2", df_inputs, n2, u2, a2, tmat, high_mask_eval, config=config)
    s2_metrics.update({"epsilon_used": max(config.epsilon_relax, config.epsilon_s2), "kappa_used": config.s2_kappa, "floor_type": "none"})
    results_list.append(s2_metrics)
    scenario_outputs["S2"] = s2_df

    print("Solving S3 via NEOS/CPLEX...", flush=True)
    model_s3, floor_info_s3 = solve_equity_refined(
        target=target,
        coeffs=coeffs,
        u_star=u_star,
        config=config,
        nu=vulnerability_vector,
        epsilon=max(config.epsilon_relax, config.epsilon_s3),
        kappa=config.s2_kappa,
        floor_mask=protected_mask_s3,
        floor_reference=a2,
        floor_gamma=config.s3_floor_gamma,
        floor_quantile=config.s3_floor_quantile,
        shortfall_penalty=config.s3_shortfall_penalty,
        use_hard_floor=False,
        caps=config.max_ports_matrix,
    )
    if not solve_safe(model_s3, accept_feasible=True):
        raise RuntimeError(f"S3 did not solve successfully: {getattr(model_s3, '_termination_condition', None)}")
    n3, u3, a3, d3 = extract_solution_refined(model_s3, len(df_inputs), len(config.venues))
    assert a3 is not None
    s3_df, s3_metrics = summarize_scenario("S3", df_inputs, n3, u3, a3, tmat, high_mask_eval, config=config)
    protected_stats = summarize_protected_set(
        a_sol=a3,
        protected_mask=protected_mask_s3,
        floor_value=floor_info_s3["floor_value"] if floor_info_s3 is not None else None,
    )
    s3_metrics.update(
        {
            "epsilon_used": max(config.epsilon_relax, config.epsilon_s3),
            "kappa_used": config.s2_kappa,
            "floor_type": "soft",
            "s3_base_floor": floor_info_s3["base_floor"] if floor_info_s3 is not None else float("nan"),
            "s3_floor_value": floor_info_s3["floor_value"] if floor_info_s3 is not None else float("nan"),
            "s3_protected_count": floor_info_s3["protected_count"] if floor_info_s3 is not None else 0,
            "s3_total_shortfall": float(np.sum(d3)) if d3 is not None else 0.0,
            "s3_mean_shortfall": float(np.mean(d3)) if d3 is not None and len(d3) > 0 else 0.0,
            "s3_max_shortfall": float(np.max(d3)) if d3 is not None and len(d3) > 0 else 0.0,
        }
    )
    s3_metrics.update({f"s3_{key}": value_ for key, value_ in protected_stats.items()})
    results_list.append(s3_metrics)
    scenario_outputs["S3"] = s3_df

    scenario_metrics = pd.DataFrame(results_list)
    scenario_metrics["scenario"] = pd.Categorical(
        scenario_metrics["scenario"],
        categories=["S0", "S1", "S2", "S3"],
        ordered=True,
    )
    scenario_metrics = scenario_metrics.sort_values("scenario").reset_index(drop=True)

    print("Running robustness battery via NEOS/CPLEX...", flush=True)
    robust_df = run_robustness(
        paths=paths,
        config=config,
        scenario_metrics=scenario_metrics,
        scenario_outputs=scenario_outputs,
        df_inputs=df_inputs,
        vulnerability=vulnerability,
        tmat=tmat,
        tm_alt=tm_alt,
        coeffs=coeffs,
        target=target,
        u_star=u_star,
        vulnerability_vector=vulnerability_vector,
        a2=a2,
        protected_mask_robust=protected_mask_s3,
        robust_eval_mask=high_mask_eval,
    )

    export_section_tables(paths, scenario_metrics, scenario_outputs, df_inputs, attention, loss_hist, robust_df)
    export_figures(paths, utam, df_inputs, loss_hist, scenario_outputs, protected_mask_s3, robust_df)
    save_scenario_outputs(paths, scenario_outputs)
    save_run_manifest(paths, inputs, config, scenario_metrics, robust_df)
    package_outputs(paths)

    reference_comparison = compare_with_reference(paths)
    return ScenarioRun(
        scenario_metrics=scenario_metrics,
        scenario_outputs=scenario_outputs,
        df_inputs=df_inputs,
        utam=utam,
        loss_hist=loss_hist,
        tmat=tmat,
        robust_df=robust_df,
        protected_mask_s3=protected_mask_s3,
        reference_comparison=reference_comparison,
    )


def run_robustness(
    paths: ProjectPaths,
    config: ModelConfig,
    scenario_metrics: pd.DataFrame,
    scenario_outputs: dict[str, pd.DataFrame],
    df_inputs: pd.DataFrame,
    vulnerability: pd.DataFrame,
    tmat: pd.DataFrame,
    tm_alt: pd.DataFrame | None,
    coeffs: dict[str, list[list[tuple[int, float]]]],
    target: np.ndarray,
    u_star: float,
    vulnerability_vector: np.ndarray,
    a2: np.ndarray,
    protected_mask_robust: np.ndarray,
    robust_eval_mask: np.ndarray,
) -> pd.DataFrame:
    baseline_row = scenario_metrics.loc[scenario_metrics["scenario"].astype(str).eq("S2")].iloc[0]
    baseline_weighted_gini = float(baseline_row["accessibility_weighted_gini"])
    baseline_coverage = float(baseline_row["coverage_within_threshold_vulnerable"])
    baseline_rank = scenario_outputs["S2"]["ports_total"].rank(method="average")
    robust_rows: list[dict[str, Any]] = []

    def rank_stability(x: pd.Series, y: pd.Series) -> float:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    def append_robust_row(row: dict[str, Any]) -> None:
        robust_rows.append(row)
        pd.DataFrame(robust_rows).to_csv(paths.tables_dir / "table_robustness_fast.csv", index=False)

    def record_nan_case(specification: str, interpretation: str, reason: str) -> None:
        append_robust_row(
            {
                "alternative_specification": specification,
                "delta_weighted_gini": float("nan"),
                "delta_vulnerable_coverage": float("nan"),
                "rank_stability": float("nan"),
                "interpretation": f"{interpretation} ({reason})",
            }
        )

    def summarize_against_baseline(label: str, df_case: pd.DataFrame, metrics_case: dict[str, Any], interpretation: str) -> None:
        append_robust_row(
            {
                "alternative_specification": label,
                "delta_weighted_gini": float(metrics_case["accessibility_weighted_gini"]) - baseline_weighted_gini,
                "delta_vulnerable_coverage": float(metrics_case["coverage_within_threshold_vulnerable"]) - baseline_coverage,
                "rank_stability": rank_stability(df_case["ports_total"].rank(method="average"), baseline_rank),
                "interpretation": interpretation,
            }
        )

    def run_weighted_case_standard(
        label: str,
        coeff_case: dict[str, list[list[tuple[int, float]]]],
        nu_case: np.ndarray,
        epsilon_case: float,
        kappa_case: float,
    ) -> tuple[pd.DataFrame | None, dict[str, Any] | None, str]:
        with temporary_solver_profile(
            config=config,
            time_limit=config.robustness_time_limit_sec,
            mip_gap=config.robustness_mip_gap,
        ):
            model_case, _ = solve_equity_refined(
                target=target,
                coeffs=coeff_case,
                u_star=u_star,
                config=config,
                nu=nu_case,
                epsilon=epsilon_case,
                kappa=kappa_case,
                floor_mask=None,
                floor_reference=None,
                shortfall_penalty=config.s3_shortfall_penalty,
                use_hard_floor=False,
                caps=config.max_ports_matrix,
                time_limit=config.robustness_time_limit_sec,
                mip_gap=config.robustness_mip_gap,
            )
        ok = solve_safe(model_case, accept_feasible=True)
        termination = str(getattr(model_case, "_termination_condition", "unknown")).lower()
        if not ok:
            return None, None, termination
        n_case, u_case, a_case, _ = extract_solution_refined(model_case, len(df_inputs), len(config.venues))
        assert a_case is not None
        df_case, metrics_case = summarize_scenario(
            label,
            df_inputs,
            n_case,
            u_case,
            a_case,
            tmat,
            robust_eval_mask,
            config=config,
        )
        return df_case, metrics_case, termination

    def run_weighted_case_soft_floor(
        label: str,
        coeff_case: dict[str, list[list[tuple[int, float]]]],
        nu_case: np.ndarray,
        epsilon_case: float,
        kappa_case: float,
        floor_gamma: float,
    ) -> tuple[pd.DataFrame | None, dict[str, Any] | None, str, dict[str, Any] | None, np.ndarray | None]:
        with temporary_solver_profile(
            config=config,
            time_limit=config.robustness_time_limit_sec,
            mip_gap=config.robustness_mip_gap,
        ):
            model_case, floor_info = solve_equity_refined(
                target=target,
                coeffs=coeff_case,
                u_star=u_star,
                config=config,
                nu=nu_case,
                epsilon=epsilon_case,
                kappa=kappa_case,
                floor_mask=protected_mask_robust,
                floor_reference=a2,
                floor_gamma=floor_gamma,
                floor_quantile=config.s3_floor_quantile,
                shortfall_penalty=config.s3_shortfall_penalty,
                use_hard_floor=False,
                caps=config.max_ports_matrix,
                time_limit=config.robustness_time_limit_sec,
                mip_gap=config.robustness_mip_gap,
            )
        ok = solve_safe(model_case, accept_feasible=True)
        termination = str(getattr(model_case, "_termination_condition", "unknown")).lower()
        if not ok:
            return None, None, termination, floor_info, None
        n_case, u_case, a_case, d_case = extract_solution_refined(model_case, len(df_inputs), len(config.venues))
        assert a_case is not None
        df_case, metrics_case = summarize_scenario(
            label,
            df_inputs,
            n_case,
            u_case,
            a_case,
            tmat,
            robust_eval_mask,
            config=config,
        )
        if d_case is not None:
            metrics_case["total_shortfall"] = float(np.sum(d_case))
            metrics_case["mean_shortfall"] = float(np.mean(d_case)) if len(d_case) > 0 else 0.0
            metrics_case["max_shortfall"] = float(np.max(d_case)) if len(d_case) > 0 else 0.0
        protected_stats = summarize_protected_set(
            a_sol=a_case,
            protected_mask=protected_mask_robust,
            floor_value=floor_info["floor_value"] if floor_info is not None else None,
        )
        for key, value_ in protected_stats.items():
            metrics_case[f"protected_{key}"] = value_
        return df_case, metrics_case, termination, floor_info, d_case

    for factor in config.robustness_lambda_factors_fast:
        label = f"{'Low' if factor < 1 else 'High'} decay (all lambda x {factor:.2f})"
        print(f"Robustness: {label}", flush=True)
        lambda_alt = {venue: value_ * factor for venue, value_ in config.lambda_v.items()}
        coeff_alt = build_access_coeffs(df_inputs, tmat, lambda_alt, config=config)
        df_alt, metrics_alt, termination = run_weighted_case_standard(
            label=f"S2_lambda_{factor:.2f}",
            coeff_case=coeff_alt,
            nu_case=vulnerability_vector,
            epsilon_case=max(config.epsilon_relax, config.epsilon_s2),
            kappa_case=config.s2_kappa,
        )
        if df_alt is None or metrics_alt is None:
            record_nan_case(label, "Accessibility-decay sensitivity", f"termination={termination}")
        else:
            summarize_against_baseline(label, df_alt, metrics_alt, "Accessibility-decay sensitivity")

    raw_vcols = [
        column
        for column in vulnerability.columns
        if column not in ["UTAM", "vulnerability_index"] and pd.api.types.is_numeric_dtype(vulnerability[column])
    ]
    if len(raw_vcols) >= 2:
        print("Robustness: PCA weights for vulnerability index", flush=True)
        x_frame = vulnerability.set_index("UTAM").reindex(df_inputs["UTAM"])[raw_vcols].apply(pd.to_numeric, errors="coerce")
        x_frame = x_frame.fillna(x_frame.median())
        x_standardized = (x_frame - x_frame.mean()) / (x_frame.std(ddof=0) + 1e-9)
        x_matrix = x_standardized.to_numpy(dtype=float)
        u_matrix, singular_values, _ = np.linalg.svd(x_matrix, full_matrices=False)
        pc1 = u_matrix[:, 0] * singular_values[0]
        if np.corrcoef(pc1, vulnerability_vector)[0, 1] < 0:
            pc1 = -pc1
        v_pca = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
        df_pca, metrics_pca, termination = run_weighted_case_standard(
            label="S2_pca_vulnerability",
            coeff_case=coeffs,
            nu_case=v_pca,
            epsilon_case=max(config.epsilon_relax, config.epsilon_s2),
            kappa_case=config.s2_kappa,
        )
        if df_pca is None or metrics_pca is None:
            record_nan_case(
                "PCA weights for vulnerability index",
                "Sensitivity to vulnerability weighting construction",
                f"termination={termination}",
            )
        else:
            summarize_against_baseline(
                "PCA weights for vulnerability index",
                df_pca,
                metrics_pca,
                "Sensitivity to vulnerability weighting construction",
            )
    else:
        record_nan_case(
            "PCA weights for vulnerability index",
            "Sensitivity to vulnerability weighting construction",
            "insufficient_numeric_components",
        )

    for epsilon_case in config.robustness_epsilon_fast:
        label = f"Alternative epsilon tolerance ({epsilon_case:.3f})"
        print(f"Robustness: {label}", flush=True)
        df_eps, metrics_eps, termination = run_weighted_case_standard(
            label=f"S2_eps_{epsilon_case:.3f}",
            coeff_case=coeffs,
            nu_case=vulnerability_vector,
            epsilon_case=epsilon_case,
            kappa_case=config.s2_kappa,
        )
        if df_eps is None or metrics_eps is None:
            record_nan_case(label, "Fidelity-equity trade-off sensitivity", f"termination={termination}")
        else:
            summarize_against_baseline(label, df_eps, metrics_eps, "Fidelity-equity trade-off sensitivity")

    for kappa_case in config.robustness_kappa_fast:
        label = f"Alternative vulnerability emphasis kappa ({kappa_case:.2f})"
        print(f"Robustness: {label}", flush=True)
        df_kap, metrics_kap, termination = run_weighted_case_standard(
            label=f"S2_kappa_{kappa_case:.2f}",
            coeff_case=coeffs,
            nu_case=vulnerability_vector,
            epsilon_case=max(config.epsilon_relax, config.epsilon_s2),
            kappa_case=kappa_case,
        )
        if df_kap is None or metrics_kap is None:
            record_nan_case(label, "Sensitivity to social-weight intensity", f"termination={termination}")
        else:
            summarize_against_baseline(label, df_kap, metrics_kap, "Sensitivity to social-weight intensity")

    for gamma_case in config.robustness_floor_gammas_fast:
        label = f"Soft floor gamma ({gamma_case:.2f})"
        print(f"Robustness: {label}", flush=True)
        df_floor, metrics_floor, termination, floor_info, _ = run_weighted_case_soft_floor(
            label=f"S3_gamma_{gamma_case:.2f}",
            coeff_case=coeffs,
            nu_case=vulnerability_vector,
            epsilon_case=max(config.epsilon_relax, config.epsilon_s3),
            kappa_case=config.s2_kappa,
            floor_gamma=gamma_case,
        )
        if df_floor is None or metrics_floor is None:
            record_nan_case(label, "Soft minimum-service guarantee sensitivity", f"termination={termination}")
        else:
            interpretation = "Soft minimum-service guarantee sensitivity"
            if floor_info is not None:
                interpretation += f" (floor={floor_info['floor_value']:.3f})"
            summarize_against_baseline(label, df_floor, metrics_floor, interpretation)

    if tm_alt is not None:
        print("Robustness: Alternative travel-time matrix", flush=True)
        try:
            tmat_alt_sq, coverage_alt = make_square_time_matrix(tm_alt, df_inputs["UTAM"].tolist())
            if coverage_alt >= 0.95:
                coeff_alt_t = build_access_coeffs(df_inputs, tmat_alt_sq, config.lambda_v, config=config)
                df_alt_t, metrics_alt_t, termination = run_weighted_case_standard(
                    label="S2_alt_time_matrix",
                    coeff_case=coeff_alt_t,
                    nu_case=vulnerability_vector,
                    epsilon_case=max(config.epsilon_relax, config.epsilon_s2),
                    kappa_case=config.s2_kappa,
                )
                if df_alt_t is None or metrics_alt_t is None:
                    record_nan_case("Alternative travel-time matrix", "Impedance-source sensitivity", f"termination={termination}")
                else:
                    summarize_against_baseline("Alternative travel-time matrix", df_alt_t, metrics_alt_t, "Impedance-source sensitivity")
            else:
                record_nan_case("Alternative travel-time matrix", "Impedance-source sensitivity", f"coverage_too_low={coverage_alt:.3f}")
        except Exception as exc:
            record_nan_case("Alternative travel-time matrix", "Impedance-source sensitivity", f"error={exc}")
    else:
        record_nan_case("Alternative travel-time matrix", "Impedance-source sensitivity", "tm_alt_not_available")

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(paths.tables_dir / "table_robustness.csv", index=False)
    robust_df.to_csv(paths.tables_dir / "table_robustness_fast.csv", index=False)
    return robust_df


def export_section_tables(
    paths: ProjectPaths,
    scenario_metrics: pd.DataFrame,
    scenario_outputs: dict[str, pd.DataFrame],
    df_inputs: pd.DataFrame,
    attention: pd.DataFrame,
    loss_hist: pd.DataFrame | None,
    robust_df: pd.DataFrame,
) -> None:
    gnn_rows: list[dict[str, Any]] = []
    if loss_hist is not None and not loss_hist.empty:
        cols_lower = {str(column).lower(): column for column in loss_hist.columns}
        metric_cols = [
            column
            for column in loss_hist.columns
            if any(token in str(column).lower() for token in ["loss", "rmse", "mae"])
        ]
        if "epoch" in cols_lower:
            loss_hist = loss_hist.sort_values(cols_lower["epoch"])
        for column in metric_cols:
            values = pd.to_numeric(loss_hist[column], errors="coerce").dropna()
            gnn_rows.append(
                {
                    "metric": column,
                    "final_value": values.iloc[-1] if len(values) else float("nan"),
                    "best_value": values.min() if len(values) else float("nan"),
                }
            )

    hour_cols = [column for column in attention.columns if str(column).startswith("hour_")]
    if hour_cols:
        h_matrix = attention[hour_cols].to_numpy(dtype=float)
        h_matrix = np.nan_to_num(h_matrix, nan=0.0)
        hhi = np.mean((h_matrix / np.maximum(h_matrix.sum(axis=1, keepdims=True), 1e-12)) ** 2)
        gnn_rows.append({"metric": "attention_concentration_hhi", "final_value": hhi, "best_value": hhi})
    pd.DataFrame(gnn_rows).to_csv(paths.tables_dir / "table_gnn_performance_available.csv", index=False)

    priority_summary = scenario_metrics[["scenario", "rho_priority_ports", "objective_abs_dev"]].copy()
    u0 = float(priority_summary.loc[priority_summary["scenario"].eq("S0"), "objective_abs_dev"].iloc[0])
    priority_summary["relative_fidelity_loss_vs_S0_pct"] = np.where(
        priority_summary["scenario"].eq("S0"),
        np.nan,
        100.0 * (priority_summary["objective_abs_dev"] - u0) / max(u0, 1e-9),
    )
    priority_summary.to_csv(paths.tables_dir / "table_priority_summary.csv", index=False)
    scenario_design = scenario_metrics[
        [
            "scenario",
            "epsilon_used",
            "kappa_used",
            "floor_type",
            "objective_abs_dev",
            "coverage_within_threshold_vulnerable",
        ]
    ].copy()
    role_map = {
        "S0": "Priority-matching baseline",
        "S1": "Equity relaxation without vulnerability weights",
        "S2": "Socially weighted equity scenario",
        "S3": "Soft-floor protection for vulnerable geography",
    }
    scenario_design["policy_role"] = scenario_design["scenario"].map(role_map)
    scenario_design["relative_fidelity_loss_vs_S0_pct"] = scenario_design["scenario"].map(
        priority_summary.set_index("scenario")["relative_fidelity_loss_vs_S0_pct"]
    ).fillna(0.0)
    scenario_design["vulnerable_coverage_pct"] = 100.0 * scenario_design["coverage_within_threshold_vulnerable"]
    if {"S2", "S3"}.issubset(set(scenario_metrics["scenario"].astype(str))):
        s2_row = scenario_metrics.loc[scenario_metrics["scenario"].eq("S2")].iloc[0]
        s3_row = scenario_metrics.loc[scenario_metrics["scenario"].eq("S3")].iloc[0]
        s3_same_as_s2 = bool(
            math.isclose(float(s2_row["objective_abs_dev"]), float(s3_row["objective_abs_dev"]), rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                float(s2_row["coverage_within_threshold_vulnerable"]),
                float(s3_row["coverage_within_threshold_vulnerable"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(s2_row["accessibility_weighted_gini"]),
                float(s3_row["accessibility_weighted_gini"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    else:
        s3_same_as_s2 = False
    scenario_design["binding_status"] = np.where(
        scenario_design["scenario"].eq("S3") & s3_same_as_s2,
        "Non-binding floor; allocation equals S2",
        np.where(
            scenario_design["scenario"].eq("S3"),
            "Floor changes the allocation",
            "",
        ),
    )
    scenario_design = scenario_design[
        [
            "scenario",
            "policy_role",
            "epsilon_used",
            "kappa_used",
            "floor_type",
            "binding_status",
            "relative_fidelity_loss_vs_S0_pct",
            "vulnerable_coverage_pct",
            "objective_abs_dev",
        ]
    ]
    scenario_design.to_csv(paths.tables_dir / "table_scenario_design.csv", index=False)
    tradeoff_table = scenario_metrics[
        [
            "scenario",
            "objective_abs_dev",
            "accessibility_weighted_gini",
            "coverage_within_threshold_vulnerable",
            "mean_access_high_vulnerability",
        ]
    ].copy()
    baseline_weighted_gini = float(
        tradeoff_table.loc[tradeoff_table["scenario"].eq("S0"), "accessibility_weighted_gini"].iloc[0]
    )
    baseline_high_access = float(
        tradeoff_table.loc[tradeoff_table["scenario"].eq("S0"), "mean_access_high_vulnerability"].iloc[0]
    )
    tradeoff_table = tradeoff_table.merge(
        priority_summary[["scenario", "relative_fidelity_loss_vs_S0_pct"]],
        on="scenario",
        how="left",
    )
    tradeoff_table["relative_fidelity_loss_vs_S0_pct"] = tradeoff_table["relative_fidelity_loss_vs_S0_pct"].fillna(0.0)
    tradeoff_table["vulnerable_coverage_pct"] = 100.0 * tradeoff_table["coverage_within_threshold_vulnerable"]
    baseline_coverage_pct = float(tradeoff_table.loc[tradeoff_table["scenario"].eq("S0"), "vulnerable_coverage_pct"].iloc[0])
    tradeoff_table["vulnerable_coverage_gain_pp_vs_S0"] = (
        tradeoff_table["vulnerable_coverage_pct"] - baseline_coverage_pct
    )
    tradeoff_table["weighted_gini_improvement_vs_S0_bp"] = 10000.0 * (
        baseline_weighted_gini - tradeoff_table["accessibility_weighted_gini"]
    )
    tradeoff_table["high_vulnerability_access_delta_vs_S0"] = (
        tradeoff_table["mean_access_high_vulnerability"] - baseline_high_access
    )
    tradeoff_table.to_csv(paths.tables_dir / "table_policy_tradeoff.csv", index=False)

    s1_df = scenario_outputs["S1"]
    s2_df = scenario_outputs["S2"]
    allocation_changes = s1_df[["UTAM", "ports_total"]].rename(columns={"ports_total": "S1_ports"}).merge(
        s2_df[["UTAM", "ports_total", "vulnerability_index"]].rename(columns={"ports_total": "S2_ports"}),
        on="UTAM",
        how="left",
    )
    allocation_changes["net_change"] = allocation_changes["S2_ports"] - allocation_changes["S1_ports"]
    allocation_changes["vulnerability_quartile"] = pd.qcut(
        allocation_changes["vulnerability_index"],
        4,
        labels=[1, 2, 3, 4],
        duplicates="drop",
    )
    allocation_changes["interpretation"] = np.where(
        allocation_changes["net_change"] > 0,
        "Gain under socially weighted equity",
        np.where(allocation_changes["net_change"] < 0, "Loss under socially weighted equity", "No net change"),
    )
    allocation_changes = allocation_changes.sort_values("net_change", ascending=False)
    alloc_changes_table = pd.concat(
        [
            allocation_changes.head(5),
            allocation_changes.tail(5).sort_values("net_change"),
        ],
        ignore_index=True,
    )
    alloc_changes_table.to_csv(paths.tables_dir / "table_allocation_changes_S2_vs_S1.csv", index=False)

    equity_metrics = scenario_metrics[
        [
            "scenario",
            "accessibility_gini",
            "accessibility_weighted_gini",
            "share_access_most_vulnerable_quartile",
            "coverage_within_threshold_vulnerable",
            "mean_access_high_vulnerability",
        ]
    ].copy()
    equity_metrics.to_csv(paths.tables_dir / "table_equity_metrics.csv", index=False)
    scenario_metrics.to_csv(paths.tables_dir / "table_equity_metrics_final.csv", index=False)
    robust_df.to_csv(paths.tables_dir / "table_robustness_fast.csv", index=False)
    official_growth = load_official_ev_growth(paths)
    if official_growth is not None and len(official_growth):
        official_growth.to_csv(paths.tables_dir / "table_official_ev_growth.csv", index=False)


def configure_line_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", color=PAPER_TEXT, pad=10)
    ax.set_xlabel(xlabel, color=PAPER_TEXT)
    ax.set_ylabel(ylabel, color=PAPER_TEXT)
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color=PAPER_GRID)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6d655e")
    ax.spines["bottom"].set_color("#6d655e")
    ax.tick_params(colors=PAPER_TEXT)


def add_time_window_bands(ax: Any) -> None:
    for start, end, label, color in TIME_WINDOWS:
        ax.axvspan(start, end, color=color, alpha=0.9, zorder=0)
        if not label:
            continue
        ax.text(
            (start + end) / 2,
            0.975,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.0,
            color="#4d4d4d",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=4,
        )


def shorten_robustness_label(text: str) -> str:
    replacements = {
        "Alternative travel-time matrix": "Alternative time matrix",
        "Low decay (all lambda x 0.75)": "Low decay",
        "High decay (all lambda x 1.25)": "High decay",
        "Alternative epsilon tolerance (0.056)": "ε = 0.056",
        "Alternative epsilon tolerance (0.080)": "ε = 0.080",
        "Alternative epsilon tolerance (0.100)": "ε = 0.100",
        "Alternative vulnerability emphasis kappa (2.00)": "κ = 2.0",
        "Alternative vulnerability emphasis kappa (4.00)": "κ = 4.0",
        "Alternative vulnerability emphasis kappa (6.00)": "κ = 6.0",
        "PCA weights for vulnerability index": "PCA vulnerability index",
        "Soft floor gamma (0.65)": "γ = 0.65",
        "Soft floor gamma (0.80)": "γ = 0.80",
        "Soft floor gamma (0.95)": "γ = 0.95",
    }
    return replacements.get(text, text)


def horizontalize_bogota(
    gdf: gpd.GeoDataFrame, target_epsg: int = 3116
) -> tuple[gpd.GeoDataFrame, float, tuple[float, float]]:
    """Rotate the projected geometry so the long axis of Bogota lies horizontal.

    Returns the rotated frame, the TOTAL rotation applied (degrees,
    counter-clockwise) and the rotation origin, so companion layers (points,
    the north arrow) can be transformed consistently.
    """
    projected = gdf.to_crs(target_epsg) if gdf.crs is not None and gdf.crs.to_epsg() != target_epsg else gdf.copy()
    if projected.crs is None:
        projected = projected.set_crs(target_epsg)
    centroids = np.column_stack([projected.geometry.centroid.x, projected.geometry.centroid.y])
    origin = tuple(projected.unary_union.centroid.coords[0])
    if len(centroids) < 2:
        return projected, 0.0, origin
    centered = centroids - centroids.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    angle = float(np.degrees(np.arctan2(vh[0, 1], vh[0, 0])))
    rotated = projected.copy()
    rotated["geometry"] = rotated.geometry.apply(lambda geom: rotate_geometry(geom, -angle, origin=origin))
    total_rotation = -angle
    minx, miny, maxx, maxy = rotated.total_bounds
    if (maxy - miny) > (maxx - minx):
        rotated["geometry"] = rotated.geometry.apply(lambda geom: rotate_geometry(geom, 90, origin=origin))
        total_rotation += 90.0
    return rotated, total_rotation, origin


def classify_zone_class(utam_code: Any) -> str:
    """Three-class layer requested by the author (decisiones_autor.md, item 9).

    Uses ONLY the UTAM code pattern of the UTAM2023 shapefile:
    - ``UPR*``            -> "rural"  (rural Bogota, 5 units, DANE data exists)
    - numeric code >= 500 -> "region" (neighbouring municipalities, 26 units;
                                       22 of them lack usable DANE population)
    - numeric code < 500  -> "urban"  (urban Bogota, 111 UTAM)
    """
    text = str(utam_code).strip().upper()
    if text.startswith("UPR"):
        return "rural"
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and int(digits) >= 500:
        return "region"
    return "urban"


def prepare_context_layers(utam: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float, tuple[float, float]]:
    """Marco de mapa en EPSG:3116 con NORTE ARRIBA y las columnas que usan las figuras.

    Revision de imprenta del 2026-10-11 (guia JTG, estilo_cartografico_comun.md): los mapas ya no
    se giran. `horizontalize_bogota` sigue en el modulo pero no se llama; se devuelve una rotacion
    de 0 grados y el centroide como origen para que los llamadores que rotan capas
    complementarias (puntos OSM en E3) sigan funcionando sin cambios.
    """
    keep = [column for column in ("UTAM", "LOCNombre", "MUNNombre", "geometry") if column in utam.columns]
    utam_map = utam[keep].copy()
    if utam_map.crs is None:
        utam_map = utam_map.set_crs(4326)
    utam_map = utam_map.to_crs(3116)
    origin = tuple(utam_map.unary_union.centroid.coords[0])
    utam_map["zone_class"] = utam_map["UTAM"].map(classify_zone_class)
    return utam_map, 0.0, origin


def label_halo(linewidth: float = 2.2) -> list[Any]:
    return [patheffects.withStroke(linewidth=linewidth, foreground="white", alpha=0.92)]


def add_north_arrow(
    ax: Any,
    rotation_deg: float,
    *,
    extent: tuple[float, float, float, float],
    x_frac: float = 0.07,
    y_frac: float = 0.95,
    size_frac: float = 0.08,
) -> list[tuple[float, float, float, float]]:
    """Flecha de norte: triangulo relleno + "N" (7 pt), arriba a la derecha.

    Con norte arriba (`rotation_deg = 0`) la flecha es vertical. `y_frac` es la posicion de la
    letra N; la flecha queda debajo. Devuelve la caja ocupada para que los rotulos no la pisen.
    """
    minx, miny, maxx, maxy = extent
    width = maxx - minx
    height = maxy - miny
    if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
        return []
    length = size_frac * height
    theta = math.radians(90.0 + rotation_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    x_n = minx + x_frac * width
    y_n = miny + y_frac * height
    x_tip = x_n - dx * 0.30 * length
    y_tip = y_n - dy * 0.30 * length
    x_base = x_tip - dx * length
    y_base = y_tip - dy * length
    ax.annotate(
        "",
        xy=(x_tip, y_tip),
        xytext=(x_base, y_base),
        arrowprops={"arrowstyle": "-|>", "color": MAP_SCALE, "linewidth": 1.1, "mutation_scale": 9, "shrinkA": 0.0, "shrinkB": 0.0},
        annotation_clip=False,
        zorder=11,
    )
    ax.text(x_n, y_n, "N", ha="center", va="center", fontsize=7.5, fontweight="bold", color=MAP_SCALE,
            path_effects=label_halo(1.8), zorder=11)
    cx = 0.5 * (x_base + x_n)
    cy = 0.5 * (y_base + y_n)
    return [(cx, cy, 0.06 * width, 0.5 * abs(y_n - y_base) + 0.03 * height)]


def add_region_ring_overlay(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    *,
    linewidth: float = 0.5,
    fill_colors: pd.Series | None = None,
) -> None:
    """Rayado del anillo de municipios vecinos (decision 9 del autor).

    Con `fill_colors` (color de relleno de cada unidad) el rayado pasa a blanco sobre rellenos
    oscuros, para que se lea igual sobre la clase alta de cualquier rampa; la leyenda
    (`region_ring_legend_handle`) muestra las dos variantes.
    """
    if "zone_class" not in gdf.columns:
        return
    ring = gdf.loc[gdf["zone_class"].eq("region")]
    if ring.empty:
        return
    if fill_colors is None:
        ring.plot(ax=ax, facecolor="none", edgecolor=REGION_HATCH_COLOR, linewidth=linewidth, hatch=REGION_HATCH, alpha=0.75, zorder=4)
        return
    colores = fill_colors.reindex(ring.index).fillna(MAP_MISSING)
    oscuro = colores.map(lambda c: _luminance(c) < 0.45)
    for mask, edge in ((~oscuro, REGION_HATCH_COLOR), (oscuro, "white")):
        subset = ring.loc[mask]
        if not subset.empty:
            subset.plot(ax=ax, facecolor="none", edgecolor=edge, linewidth=linewidth, hatch=REGION_HATCH, alpha=0.9, zorder=4)
    ring.boundary.plot(ax=ax, color=MAP_OUTLINE, linewidth=0.35, zorder=4)


def region_ring_legend_patch(label: str = "Neighbouring municipalities") -> Patch:
    return Patch(facecolor="white", edgecolor=REGION_HATCH_COLOR, hatch=REGION_HATCH, linewidth=0.6, label=label)


def label_box(
    ax: Any,
    bounds: tuple[float, float, float, float],
    text: str,
    fontsize: float,
    x: float,
    y: float,
    *,
    ha: str = "center",
    va: str = "center",
) -> tuple[float, float, float, float]:
    """Caja aproximada de un rotulo en unidades de dato: (cx, cy, semiancho, semialto).

    No se miden las letras con el backend (las posiciones de los ejes aun no son
    definitivas con constrained_layout): basta con el ancho medio de caracter de
    una fuente sans, 0.52 em, y la altura de linea, 1.2 em.
    """
    minx, miny, maxx, maxy = bounds
    fig_w, fig_h = ax.figure.get_size_inches()
    # Tamano real del eje: se fuerza un dibujado (una vez por eje) para que
    # constrained_layout y el aspecto igual fijen su posicion. Antes se usaba
    # una fraccion fija del lienzo, que en las figuras de tres paneles apilados
    # subestimaba las cajas unas tres veces y dejaba pasar colisiones
    # (revision editorial 2026-09-02, Figuras 4 y 6).
    if not getattr(ax, "_social_eq_layout_done", False):
        ax.figure.canvas.draw()
        ax._social_eq_layout_done = True
    pos = ax.get_position()
    ax_w = max(pos.width * fig_w, 1e-6)
    ax_h = max(pos.height * fig_h, 1e-6)
    escala = max((maxx - minx) / ax_w, (maxy - miny) / ax_h)
    dato_por_pulgada_x = escala
    dato_por_pulgada_y = escala
    semiancho = 0.5 * len(text) * 0.52 * (fontsize / 72.0) * dato_por_pulgada_x
    semialto = 0.5 * 1.2 * (fontsize / 72.0) * dato_por_pulgada_y
    cx = x if ha == "center" else (x + semiancho if ha == "left" else x - semiancho)
    cy = y if va == "center" else (y - semialto if va == "top" else y + semialto)
    return cx, cy, semiancho, semialto


def add_locality_labels(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    *,
    locality_names: tuple[str, ...] = MAJOR_LOCALITIES,
    municipality_names: tuple[str, ...] = MAJOR_MUNICIPALITIES,
    fontsize: float = 6.9,
    color: str = "#4a443c",
    extent: tuple[float, float, float, float] | None = None,
    reserved: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Label major localities (LOCNombre) and neighbouring municipalities.

    Only names present in the shapefile are used; text sits at the
    representative point of the dissolved administrative unit.

    Las figuras se dibujan al ancho al que se imprimen (R2.m2), asi que en el
    nucleo urbano los rotulos ya no caben todos: se colocan por area descendente
    y se omite el que caeria encima de uno ya puesto. `reserved` permite pasar
    puntos ya ocupados (por ejemplo el del centro historico). Devuelve los
    puntos efectivamente ocupados.
    """

    def within_extent(point: Any) -> bool:
        if extent is None:
            return True
        minx, miny, maxx, maxy = extent
        return bool(minx <= point.x <= maxx and miny <= point.y <= maxy)

    limites = tuple(extent) if extent is not None else tuple(gdf.total_bounds)
    ocupados: list[tuple[float, float, float, float]] = list(reserved or [])

    def cabe(caja: tuple[float, float, float, float]) -> bool:
        cx, cy, hw, hh = caja
        if extent is not None:
            ex0, ey0, ex1, ey1 = extent
            if cx - hw < ex0 or cx + hw > ex1 or cy - hh < ey0 or cy + hh > ey1:
                return False  # el rotulo saldria del recorte
        for ox, oy, ohw, ohh in ocupados:
            if abs(cx - ox) < (hw + ohw) and abs(cy - oy) < (hh + ohh):
                return False
        return True

    halo = label_halo()
    if "LOCNombre" in gdf.columns:
        localities = gdf.dropna(subset=["LOCNombre"])
        candidatos = []
        for name in locality_names:
            subset = localities.loc[localities["LOCNombre"].astype(str).str.upper().str.strip().eq(name)]
            if subset.empty:
                continue
            unidad = subset.dissolve()
            candidatos.append((float(unidad.geometry.area.iloc[0]), name, unidad.geometry.representative_point().iloc[0]))
        # Las unidades grandes se rotulan primero: son las que el lector busca.
        for _area, name, point in sorted(candidatos, key=lambda c: -c[0]):
            caja = label_box(ax, limites, place_name(name), fontsize, point.x, point.y)
            if not within_extent(point) or not cabe(caja):
                continue
            ocupados.append(caja)
            ax.text(
                point.x,
                point.y,
                place_name(name),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                path_effects=halo,
                zorder=9,
            )
    if "MUNNombre" in gdf.columns and "zone_class" in gdf.columns:
        region = gdf.loc[gdf["zone_class"].eq("region")]
        cand_mun = []
        for name in municipality_names:
            subset = region.loc[region["MUNNombre"].astype(str).str.upper().str.strip().eq(name)]
            if subset.empty:
                continue
            unidad = subset.dissolve()
            cand_mun.append((float(unidad.geometry.area.iloc[0]), name, unidad.geometry.representative_point().iloc[0]))
        for _area, name, point in sorted(cand_mun, key=lambda c: -c[0]):
            caja = label_box(ax, limites, place_name(name), fontsize - 0.5, point.x, point.y)
            if not within_extent(point) or not cabe(caja):
                continue
            ocupados.append(caja)
            ax.text(
                point.x,
                point.y,
                place_name(name),
                ha="center",
                va="center",
                fontsize=fontsize - 0.5,
                fontstyle="italic",
                color="#6c655c",
                path_effects=halo,
                zorder=9,
            )
    return ocupados


def add_city_centre_marker(
    ax: Any, gdf: gpd.GeoDataFrame, *, fontsize: float = 6.9
) -> list[tuple[float, float, float, float]]:
    """Star marker on the historic city centre (Candelaria locality).

    Devuelve el punto del rotulo para que `add_locality_labels` no escriba encima.
    """
    if "LOCNombre" not in gdf.columns:
        return []
    subset = gdf.loc[gdf["LOCNombre"].astype(str).str.upper().str.strip().eq("CANDELARIA")]
    if subset.empty:
        return []
    point = subset.dissolve().geometry.representative_point().iloc[0]
    minx, _, maxx, _ = gdf.total_bounds
    offset = 0.008 * (maxx - minx)
    ax.scatter([point.x], [point.y], marker="*", s=95, color=PAPER_TEXT, edgecolor="white", linewidth=0.7, zorder=10)
    ax.text(
        point.x + offset,
        point.y - offset,
        "City centre",
        ha="left",
        va="top",
        fontsize=fontsize,
        color=PAPER_TEXT,
        path_effects=label_halo(),
        zorder=10,
    )
    return [
        label_box(
            ax,
            tuple(gdf.total_bounds),
            "City centre",
            fontsize,
            point.x + offset,
            point.y - offset,
            ha="left",
            va="top",
        )
    ]


def make_ports_norm(scenario_outputs: dict[str, pd.DataFrame]) -> BoundaryNorm | None:
    """Clases fijas de puertos por zona (R1.11), seis clases YlOrBr; la ultima es abierta (300+)."""
    all_ports: list[float] = []
    for name in ("S0", "S1", "S2", "S3"):
        frame = scenario_outputs.get(name)
        if frame is not None and "ports_total" in frame.columns:
            all_ports.extend(pd.to_numeric(frame["ports_total"], errors="coerce").dropna().tolist())
    if not all_ports:
        return None
    top = float(np.max(all_ports))
    bounds = [1.0, 10.0, 25.0, 75.0, 150.0, 300.0, max(top + 1.0, 301.0)]
    return BoundaryNorm(bounds, ncolors=len(PORTS_COLORS))


def label_open_top_class(cbar: Any, norm: BoundaryNorm, *, unit: str = "") -> None:
    """Rotula todos los cortes de una escala clasificada; la ultima clase se marca abierta ("300+").

    Revision 2026-10-11 (guia JTG, Fig. 6): antes el corte de 300 quedaba sin numero entre 150 y
    "300+". Ahora cada corte lleva su numero y el extremo derecho dice "+".
    """
    bounds = list(norm.boundaries)
    cbar.set_ticks(bounds)
    labels = [f"{b:g}{unit}" for b in bounds[:-1]] + ["+"]
    cbar.set_ticklabels(labels)


def plot_map_panel(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: Any,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    linewidth: float = 0.35,
    edgecolor: str = MAP_EDGE,
    zero_fill: str | None = None,
    zero_tol: float = 1e-9,
    show_scale_bar: bool = False,
    focus_extent: tuple[float, float, float, float] | None = None,
    show_focus_box: bool = False,
    show_focus_inset: bool = False,
) -> None:
    plot_map_values(
        ax,
        gdf,
        column,
        cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        linewidth=linewidth,
        edgecolor=edgecolor,
        zero_fill=zero_fill,
        zero_tol=zero_tol,
    )
    draw_map_outline(ax, gdf)
    if show_focus_inset and focus_extent is not None:
        draw_focus_box(ax, focus_extent)
        add_focus_inset(
            ax,
            gdf,
            column,
            cmap,
            focus_extent,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            linewidth=max(0.30, linewidth),
            edgecolor=edgecolor,
            zero_fill=zero_fill,
            zero_tol=zero_tol,
        )
    elif show_focus_box and focus_extent is not None:
        draw_focus_box(ax, focus_extent)
    if show_scale_bar:
        add_scale_bar(ax, gdf)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold", color=PAPER_TEXT, pad=8)
    ax.set_axis_off()
    ax.set_facecolor(PAPER_PANEL)
    ax.set_aspect("equal")
    ax.margins(0.01)


def add_horizontal_colorbar(
    fig: Any,
    ax: Any,
    cmap: Any,
    norm: Any,
    label: str,
    *,
    fraction: float = 0.05,
    pad: float = 0.04,
    cax: Any | None = None,
    shrink: float | None = None,
    spacing: str | None = None,
) -> Any:
    colorbar_kwargs: dict[str, Any] = {
        "orientation": "horizontal",
    }
    if spacing is not None:
        colorbar_kwargs["spacing"] = spacing
    if cax is not None:
        colorbar_kwargs["cax"] = cax
    else:
        colorbar_kwargs.update(
            {
                "ax": ax,
                "fraction": fraction,
                "pad": pad,
                "aspect": 35,
            }
        )
        if shrink is not None:
            colorbar_kwargs["shrink"] = shrink
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), **colorbar_kwargs)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=PAPER_TEXT)
    cbar.set_label(label, fontsize=9, color=PAPER_TEXT)
    return cbar


def plot_binary_map_panel(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    *,
    show_scale_bar: bool = False,
    focus_extent: tuple[float, float, float, float] | None = None,
    show_focus_box: bool = False,
) -> None:
    colors = gdf[column].fillna(0).astype(int).map(PROTECTED_COLORS).fillna(MAP_MISSING)
    gdf.assign(_fill=colors).plot(color=colors, ax=ax, linewidth=0.35, edgecolor=MAP_EDGE)
    draw_map_outline(ax, gdf)
    if show_focus_box and focus_extent is not None:
        draw_focus_box(ax, focus_extent)
    if show_scale_bar:
        add_scale_bar(ax, gdf)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold", color=PAPER_TEXT, pad=8)
    ax.set_axis_off()
    ax.set_facecolor(PAPER_PANEL)
    ax.set_aspect("equal")
    ax.margins(0.01)


def plot_map_values(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: Any,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    linewidth: float = 0.35,
    edgecolor: str = MAP_EDGE,
    zero_fill: str | None = None,
    zero_tol: float = 1e-9,
) -> None:
    kwargs: dict[str, Any] = {
        "column": column,
        "ax": ax,
        "cmap": cmap,
        "linewidth": linewidth,
        "edgecolor": edgecolor,
        "missing_kwds": {"color": MAP_MISSING},
    }
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    if zero_fill is not None:
        values = pd.to_numeric(gdf[column], errors="coerce")
        zero_mask = values.notna() & np.isclose(values.to_numpy(dtype=float), 0.0, atol=zero_tol)
        if bool(np.any(zero_mask)):
            gdf.loc[zero_mask].plot(ax=ax, color=zero_fill, linewidth=linewidth, edgecolor=edgecolor)
        nonzero_mask = values.notna() & ~zero_mask
        if bool(np.any(nonzero_mask)):
            gdf.loc[nonzero_mask].plot(**kwargs)
        missing_mask = values.isna()
        if bool(np.any(missing_mask)):
            gdf.loc[missing_mask].plot(ax=ax, color=MAP_MISSING, linewidth=linewidth, edgecolor=edgecolor)
    else:
        gdf.plot(**kwargs)


def draw_map_outline(ax: Any, gdf: gpd.GeoDataFrame) -> None:
    if gdf.empty:
        return
    outline = gdf.dissolve()
    outline.boundary.plot(
        ax=ax,
        color=MAP_OUTLINE,
        linewidth=0.9,
        zorder=5,
    )


def choose_scale_length(width_m: float) -> float:
    candidates = np.array([2000, 5000, 10000, 15000, 20000, 25000, 50000], dtype=float)
    target = max(width_m * 0.16, 2000.0)
    idx = int(np.argmin(np.abs(candidates - target)))
    return float(candidates[idx])


def derive_focus_extent(
    gdf: gpd.GeoDataFrame,
    *,
    area_quantile: float = 0.60,
    pad_fraction: float = 0.055,
) -> tuple[float, float, float, float] | None:
    if gdf.empty or gdf.crs is None or not bool(getattr(gdf.crs, "is_projected", False)):
        return None
    areas = pd.Series(gdf.geometry.area)
    threshold = float(areas.quantile(area_quantile))
    subset = gdf.loc[areas.le(threshold)].copy()
    if subset.empty:
        return None
    centroids = subset.geometry.centroid
    x_low, x_high = centroids.x.quantile([0.05, 0.95]).tolist()
    y_low, y_high = centroids.y.quantile([0.08, 0.94]).tolist()
    core_mask = centroids.x.between(x_low, x_high) & centroids.y.between(y_low, y_high)
    core = subset.loc[core_mask].copy()
    if core.empty:
        core = subset
    minx, miny, maxx, maxy = core.total_bounds
    width = maxx - minx
    height = maxy - miny
    if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
        return None
    xpad = pad_fraction * width
    ypad = pad_fraction * height
    return (minx - xpad, miny - ypad, maxx + xpad, maxy + ypad)


def draw_focus_box(ax: Any, focus_extent: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = focus_extent
    rect = Rectangle(
        (minx, miny),
        maxx - minx,
        maxy - miny,
        fill=False,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        edgecolor=MAP_OUTLINE,
        zorder=6,
    )
    ax.add_patch(rect)


def add_scale_bar(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    *,
    extent: tuple[float, float, float, float] | None = None,
    x_frac: float = 0.06,
    y_frac: float = 0.04,
    fontsize: float = 6.8,
) -> list[tuple[float, float, float, float]]:
    """Barra de escala alternada (blanco / gris oscuro) con 0, mitad y longitud total en km.

    `x_frac` es el extremo IZQUIERDO de la barra en fraccion del extent (revision 2026-10-11:
    antes era el centro). Devuelve la caja ocupada para reservarla frente a los rotulos.
    """
    if gdf.empty or gdf.crs is None or not bool(getattr(gdf.crs, "is_projected", False)):
        return []
    minx, miny, maxx, maxy = extent if extent is not None else gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
        return []
    length_m = choose_scale_length(width)
    x0 = minx + x_frac * width
    y0 = miny + y_frac * height
    bar_h = 0.011 * height
    half = 0.5 * length_m
    for i, color in enumerate((MAP_SCALE, "white")):
        ax.add_patch(Rectangle((x0 + i * half, y0), half, bar_h, facecolor=color, edgecolor=MAP_SCALE, linewidth=0.6, zorder=10))
    km = length_m / 1000.0
    for pos, texto in ((0.0, "0"), (length_m, f"{km:g} km")):
        ax.text(x0 + pos, y0 + bar_h + 0.008 * height, texto, ha="center", va="bottom", fontsize=fontsize, color=MAP_SCALE, path_effects=label_halo(1.6), zorder=10)
    return [(x0 + half, y0 + 0.02 * height, half + 0.06 * width, 0.035 * height)]


def add_focus_inset(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: Any,
    focus_extent: tuple[float, float, float, float],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    linewidth: float = 0.35,
    edgecolor: str = MAP_EDGE,
    zero_fill: str | None = None,
    zero_tol: float = 1e-9,
) -> None:
    inset_ax = ax.inset_axes([0.62, 0.14, 0.28, 0.36], transform=ax.transAxes)
    plot_map_values(
        inset_ax,
        gdf,
        column,
        cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        linewidth=linewidth,
        edgecolor=edgecolor,
        zero_fill=zero_fill,
        zero_tol=zero_tol,
    )
    draw_map_outline(inset_ax, gdf)
    minx, miny, maxx, maxy = focus_extent
    inset_ax.set_xlim(minx, maxx)
    inset_ax.set_ylim(miny, maxy)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_facecolor("white")
    inset_ax.margins(0.0)
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(MAP_OUTLINE)
        spine.set_linewidth(0.9)
    inset_ax.text(
        0.03,
        0.97,
        "Urban focus",
        transform=inset_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color=MAP_SCALE,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def load_optional_growth_long(paths: ProjectPaths) -> pd.DataFrame | None:
    candidate_names = [
        "annual_growth_by_utam.csv",
        "utam_annual_growth.csv",
        "growth_by_year_utam.csv",
    ]
    for name in candidate_names:
        candidate = paths.input_dir / name
        if not candidate.exists():
            continue
        growth = pd.read_csv(candidate)
        growth = growth.rename(columns={find_utam_col(growth): "UTAM"})
        growth["UTAM"] = growth["UTAM"].map(normalize_zone_code)
        lower_map = {str(column).lower(): column for column in growth.columns}
        if {"utam", "year"}.issubset(lower_map):
            value_col = next(
                (
                    column
                    for column in growth.columns
                    if str(column).lower() not in {"utam", "year"} and pd.api.types.is_numeric_dtype(growth[column])
                ),
                None,
            )
            if value_col is not None:
                return growth[["UTAM", lower_map["year"], value_col]].rename(
                    columns={lower_map["year"]: "year", value_col: "value"}
                )
        year_cols = [column for column in growth.columns if str(column).lower().startswith("year_")]
        if year_cols:
            long_df = growth.melt(
                id_vars=["UTAM"],
                value_vars=year_cols,
                var_name="year",
                value_name="value",
            )
            long_df["year"] = long_df["year"].str.extract(r"(\d+)").astype(int)
            return long_df
    return None


def load_official_ev_growth(paths: ProjectPaths) -> pd.DataFrame | None:
    candidate = paths.input_dir / "bogota_official_ev_growth.csv"
    if not candidate.exists():
        return None
    growth = pd.read_csv(candidate)
    required = {"year", "checkpoint_date", "electric_vehicles"}
    if not required.issubset({str(column) for column in growth.columns}):
        return None
    growth = growth.copy()
    growth["year"] = pd.to_numeric(growth["year"], errors="coerce")
    growth["checkpoint_date"] = pd.to_datetime(growth["checkpoint_date"], errors="coerce")
    growth["electric_vehicles"] = pd.to_numeric(growth["electric_vehicles"], errors="coerce")
    growth = growth.dropna(subset=["year", "electric_vehicles"]).sort_values(["year", "checkpoint_date"]).reset_index(drop=True)
    if growth.empty:
        return None
    growth["year"] = growth["year"].astype(int)
    growth["annual_addition"] = growth["electric_vehicles"].diff()
    growth["annual_growth_pct"] = 100.0 * growth["annual_addition"] / growth["electric_vehicles"].shift(1)
    growth["checkpoint_label"] = np.where(
        growth["checkpoint_date"].notna(),
        growth["checkpoint_date"].dt.strftime("%Y\n%b"),
        growth["year"].astype(str),
    )
    return growth


def urban_core_extent(gdf: gpd.GeoDataFrame, pad_m: float = 3000.0) -> tuple[float, float, float, float]:
    """Marco urbano comun, norte arriba (estilo_cartografico_comun.md).

    Si la capa esta en EPSG:3116 se devuelve la constante `URBAN_FRAME_3116` (30 x 47.5 km), la
    misma en las Figs. 4, 6, 7 y C1, para que los mapas comparables tengan exactamente el mismo
    marco. En otro CRS se recurre a la caja de las UTAM urbanas con `pad_m` de margen.
    """
    try:
        if gdf.crs is not None and gdf.crs.to_epsg() == 3116:
            return URBAN_FRAME_3116
    except Exception:  # noqa: BLE001
        pass
    if "zone_class" not in gdf.columns or not gdf["zone_class"].eq("urban").any():
        return tuple(gdf.total_bounds)
    minx, miny, maxx, maxy = gdf.loc[gdf["zone_class"].eq("urban")].total_bounds
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def draw_locator(ax: Any, gdf: gpd.GeoDataFrame, crop: tuple[float, float, float, float], *, title: str | None = None) -> None:
    """Mapa de situacion: region completa (norte arriba) en tres clases con el recuadro del marco."""
    for zone_name, fill in ZONE_FILLS.items():
        subset = gdf.loc[gdf["zone_class"].eq(zone_name)] if "zone_class" in gdf.columns else gdf.iloc[0:0]
        if not subset.empty:
            subset.plot(ax=ax, color=fill, edgecolor="none")
    add_region_ring_overlay(ax, gdf)
    draw_map_outline(ax, gdf)
    ax.add_patch(
        Rectangle((crop[0], crop[1]), crop[2] - crop[0], crop[3] - crop[1], fill=False, edgecolor=NEUTRAL_DARK, linewidth=1.0, zorder=8)
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_aspect("equal")
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.03 * (maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    add_neatline(ax)
    if title:
        panel_label(ax, title)


def _titulo_panel(ax: Any, title: str) -> None:
    panel_label(ax, title)


def fig_social_context_panels(
    figures_dir: Path,
    utam_map: gpd.GeoDataFrame,
    df_inputs: pd.DataFrame,
    protected_mask_s3: np.ndarray,
    rotation_deg: float,
    imputed_units: set[str] | None = None,
) -> None:
    """Figura 4 - insumos socioespaciales, norte arriba, en una fila a doble columna.

    Revision de imprenta del 2026-10-11: (a)-(c) en el marco urbano comun y (d) la region completa
    como localizador, todos con norte arriba; rotulos "(a)" iguales; indices 0-1 en cinco clases de
    0.2 (R1.11). `imputed_units` son las unidades sin dato censal cuyo indice de vulnerabilidad el
    modelo imputa con la mediana (`build_inputs`); en (b) se muestran como "sin dato censal" en vez
    de con el color de la mediana.
    """
    gdf_priority = utam_map.merge(df_inputs[["UTAM", "priority_norm", "vulnerability_index"]], on="UTAM", how="left")
    if imputed_units:
        gdf_priority["vulnerability_shown"] = gdf_priority["vulnerability_index"].where(~gdf_priority["UTAM"].isin(imputed_units))
    else:
        gdf_priority["vulnerability_shown"] = gdf_priority["vulnerability_index"]
    protected_df = pd.DataFrame({"UTAM": df_inputs["UTAM"], "protected_s3": np.asarray(protected_mask_s3).astype(int)})
    gdf_protected = utam_map.merge(protected_df, on="UTAM", how="left")
    crop = urban_core_extent(utam_map)
    index_norm = BoundaryNorm([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001], ncolors=len(INDEX_COLORS))

    frame_ratio = (crop[3] - crop[1]) / (crop[2] - crop[0])
    rb = utam_map.total_bounds
    locator_ratio = (rb[3] - rb[1]) / (rb[2] - rb[0])
    fig = plt.figure(figsize=(7.05, 3.80))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, frame_ratio / locator_ratio], height_ratios=[1.0, 0.25],
                            left=0.005, right=0.995, top=0.95, bottom=0.0, wspace=0.06, hspace=0.04)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_loc = fig.add_subplot(grid[0, 3])

    paneles = (
        (axes[0], gdf_priority, "priority_norm", "(a)"),
        (axes[1], gdf_priority, "vulnerability_shown", "(b)"),
    )
    for ax, gdf, column, label in paneles:
        plot_map_values(ax, gdf, column, INDEX_CMAP, norm=index_norm, linewidth=0.25)
        add_region_ring_overlay(ax, gdf, fill_colors=fill_colors_for(gdf, column, INDEX_CMAP, index_norm))
        draw_map_outline(ax, gdf)
        set_map_frame(ax, crop)
        panel_label(ax, label)
    colors = gdf_protected["protected_s3"].fillna(0).astype(int).map(PROTECTED_COLORS).fillna(MAP_MISSING)
    gdf_protected.plot(color=colors, ax=axes[2], linewidth=0.25, edgecolor=MAP_EDGE)
    add_region_ring_overlay(axes[2], gdf_protected)
    draw_map_outline(axes[2], gdf_protected)
    set_map_frame(axes[2], crop)
    panel_label(axes[2], "(c)")

    for ax in axes:
        reservado = add_north_arrow(ax, rotation_deg, extent=crop)
        reservado += add_scale_bar(ax, gdf_priority, extent=crop)
        if ax is axes[0]:
            reservado += add_city_centre_marker(ax, gdf_priority, fontsize=7.0)
            add_locality_labels(ax, gdf_priority, extent=crop, reserved=reservado, fontsize=7.0)

    draw_locator(ax_loc, utam_map, crop, title="(d)")
    for ax in axes + [ax_loc]:
        ax.set_anchor("N")

    fila = grid[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.10)
    ax_cbar = fig.add_subplot(fila[0, 0])
    ax_leg = fig.add_subplot(fila[0, 1])
    pos = ax_cbar.get_position()
    ax_cbar.set_position([pos.x0 + 0.03, pos.y0 + 0.55 * pos.height, pos.width - 0.05, 0.22 * pos.height])
    cbar = add_horizontal_colorbar(fig, ax_cbar, INDEX_CMAP, index_norm, "Normalized index, 0–1 (a, b)", cax=ax_cbar, spacing="proportional")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_label("Normalized index, 0–1 (a, b)", fontsize=7.5)
    ax_leg.axis("off")
    handles = [
        Patch(facecolor=PROTECTED_COLORS[1], edgecolor="none", label="Protected zone, S3 (c)"),
        Patch(facecolor=PROTECTED_COLORS[0], edgecolor=MAP_EDGE, linewidth=0.5, label="Other zone (c)"),
        Patch(facecolor=MAP_MISSING, edgecolor=MAP_EDGE, linewidth=0.5, label="No census data (b)"),
        region_ring_legend_handle(),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=NEUTRAL_DARK, markeredgecolor="white", markersize=8, label="Historic city centre"),
        Patch(facecolor="none", edgecolor=NEUTRAL_DARK, linewidth=1.0, label="Frame of (a)–(c), in (d)"),
    ]
    labels = [h.get_label() if not isinstance(h, tuple) else "Neighbouring municipalities" for h in handles]
    ax_leg.legend(handles=handles, labels=labels, loc="center left", ncol=2, frameon=False, fontsize=7.0,
                  handletextpad=0.5, columnspacing=1.0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    save_figure(fig, figures_dir, "fig_social_context_panels")
    plt.close(fig)


def make_intensity_norm() -> BoundaryNorm:
    """Cortes fijos para puertos por 10.000 residentes: 10 y 15 enmarcan la brecha 10.4 / 15.6."""
    return BoundaryNorm([0.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 1e9], ncolors=len(INTENSITY_COLORS))


def fig_allocation_composite(
    figures_dir: Path,
    utam_map: gpd.GeoDataFrame,
    scenario_outputs: dict[str, pd.DataFrame],
    rotation_deg: float,
    population: pd.DataFrame | None = None,
) -> None:
    """Figura 6 - asignacion por escenario, norte arriba, cuatro paneles en fila.

    Revision de imprenta del 2026-10-11: (a)-(c) puertos por zona en S0, S1 y S2 con seis clases
    comunes YlOrBr; (d) puertos por 10.000 residentes en S2 (Blues). Mismo marco urbano en los
    cuatro paneles; norte y escala en cada uno; "sin puertos" en blanco y "sin poblacion censal" en
    gris, cada uno con su entrada. El rayado del anillo cambia a blanco sobre las clases oscuras y la
    leyenda muestra ambas variantes (antes: rayado sobre azul oscuro en el mapa y sobre blanco en la
    leyenda).
    """
    intensity_panel = population is not None and "S2" in scenario_outputs
    core = ["S0", "S1", "S2"] if intensity_panel else ["S0", "S1", "S2", "S3"]
    ordered_scenarios = [name for name in core if name in scenario_outputs]
    if not ordered_scenarios:
        return
    ports_norm = make_ports_norm(scenario_outputs)
    if ports_norm is None:
        return
    crop = urban_core_extent(utam_map)
    paneles = list(ordered_scenarios) + (["intensity"] if intensity_panel else [])
    intensity_norm = make_intensity_norm()

    fig = plt.figure(figsize=(7.05, 3.55))
    grid = fig.add_gridspec(2, len(paneles), height_ratios=[1.0, 0.24], left=0.005, right=0.995, top=0.95, bottom=0.0,
                            wspace=0.05, hspace=0.04)
    axes = []
    for idx, scenario_name in enumerate(paneles):
        axis = fig.add_subplot(grid[0, idx])
        axes.append(axis)
        letra = f"({chr(97 + idx)})"
        if scenario_name == "intensity":
            pop = population[["UTAM", "dane_population"]].copy()
            pop["UTAM"] = pop["UTAM"].map(normalize_zone_code)
            s2 = scenario_outputs["S2"][["UTAM", "ports_total"]].copy()
            s2["UTAM"] = s2["UTAM"].map(normalize_zone_code)
            gdf = utam_map.copy()
            gdf["UTAM"] = gdf["UTAM"].map(normalize_zone_code)
            gdf = gdf.merge(s2, on="UTAM", how="left").merge(pop, on="UTAM", how="left")
            with np.errstate(divide="ignore", invalid="ignore"):
                gdf["ports_per_10k"] = np.where(
                    pd.to_numeric(gdf["dane_population"], errors="coerce") > 0,
                    1e4 * pd.to_numeric(gdf["ports_total"], errors="coerce") / pd.to_numeric(gdf["dane_population"], errors="coerce"),
                    np.nan,
                )
            column, cmap, norm, header = "ports_per_10k", INTENSITY_CLASS_CMAP, intensity_norm, "S2, per capita"
        else:
            gdf = utam_map.merge(scenario_outputs[scenario_name][["UTAM", "ports_total"]], on="UTAM", how="left")
            column, cmap, norm, header = "ports_total", PORTS_CMAP, ports_norm, scenario_name
        plot_map_values(axis, gdf, column, cmap, norm=norm, linewidth=0.25, zero_fill=NO_PORTS_FILL)
        add_region_ring_overlay(axis, gdf, fill_colors=fill_colors_for(gdf, column, cmap, norm, zero_fill=NO_PORTS_FILL))
        draw_map_outline(axis, gdf)
        set_map_frame(axis, crop)
        panel_label(axis, f"{letra} {header}")
        reservado = add_north_arrow(axis, rotation_deg, extent=crop)
        reservado += add_scale_bar(axis, gdf, extent=crop)
        if idx == 0:
            reservado += add_city_centre_marker(axis, gdf, fontsize=7.0)
            add_locality_labels(axis, gdf, extent=crop, reserved=reservado, fontsize=7.0)

    if intensity_panel:
        fila = grid[1, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.12)
        cbar_ax = fig.add_subplot(fila[0, 0])
        cbar2_ax = fig.add_subplot(fila[0, 1])
        legend_ax = fig.add_subplot(fila[0, 2])
    else:
        fila = grid[1, :].subgridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.10)
        cbar_ax = fig.add_subplot(fila[0, 0])
        cbar2_ax = None
        legend_ax = fig.add_subplot(fila[0, 1])
    for cax in [c for c in (cbar_ax, cbar2_ax) if c is not None]:
        pos = cax.get_position()
        cax.set_position([pos.x0 + 0.02, pos.y0 + 0.55 * pos.height, pos.width - 0.03, 0.22 * pos.height])
    cbar = add_horizontal_colorbar(fig, axes, PORTS_CMAP, ports_norm, "Charging ports per zone, (a)–(c)", cax=cbar_ax, spacing="uniform")
    label_open_top_class(cbar, ports_norm)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_label("Charging ports per zone, (a)–(c)", fontsize=7.5)
    handles: list[Any] = [Patch(facecolor=NO_PORTS_FILL, edgecolor=MAP_EDGE, linewidth=0.6, label="No ports allocated")]
    if cbar2_ax is not None:
        cbar2 = add_horizontal_colorbar(fig, axes, INTENSITY_CLASS_CMAP, intensity_norm, "Ports per 10,000 residents, S2 (d)", cax=cbar2_ax, spacing="uniform")
        label_open_top_class(cbar2, intensity_norm)
        cbar2.ax.tick_params(labelsize=7.0)
        cbar2.set_label("Ports per 10,000 residents, S2 (d)", fontsize=7.5)
        handles.append(Patch(facecolor=MAP_MISSING, edgecolor=MAP_EDGE, linewidth=0.6, label="No census population (d)"))
    handles.append(region_ring_legend_handle())
    labels = [h.get_label() if not isinstance(h, tuple) else "Neighbouring municipalities" for h in handles]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, labels=labels, loc="center left", ncol=1, frameon=False, fontsize=7.0,
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    save_figure(fig, figures_dir, "fig_allocation_composite")
    plt.close(fig)


def _luminance(color: Any) -> float:
    """Luminancia relativa aproximada (0 negro, 1 blanco) de un color de matplotlib."""
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def add_neatline(ax: Any) -> None:
    """Marco fino alrededor del panel de mapa (neat line, guia JTG 1.3)."""
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=NEATLINE_COLOR, linewidth=0.5, zorder=20, clip_on=False))


def panel_label(ax: Any, text: str, *, x: float = 0.0, y: float = 1.012, fontsize: float = 9.0) -> None:
    """Rotulo de panel comun a todas las figuras: "(a)" en negrita de 9 pt, arriba a la izquierda,
    seguido como mucho de un encabezado corto en regular de 8 pt ("(a) S0")."""
    letra, _, resto = text.partition(" ")
    t = ax.text(x, y, letra, transform=ax.transAxes, ha="left", va="bottom", fontsize=fontsize, fontweight="bold", color=PAPER_TEXT, clip_on=False)
    if resto:
        ax.annotate(resto, xy=(1.0, 0.0), xycoords=t, xytext=(3.5, 0), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.0, color=PAPER_TEXT, annotation_clip=False)


def fill_colors_for(gdf: gpd.GeoDataFrame, column: str, cmap: Any, norm: Any, *, zero_fill: str | None = None) -> pd.Series:
    """Color de relleno de cada unidad con la misma regla que `plot_map_values`."""
    values = pd.to_numeric(gdf[column], errors="coerce")
    out = []
    for v in values:
        if not np.isfinite(v):
            out.append(MAP_MISSING)
        elif zero_fill is not None and abs(v) < 1e-9:
            out.append(zero_fill)
        else:
            out.append(cmap(norm(v)))
    return pd.Series(out, index=gdf.index)


def set_map_frame(ax: Any, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    add_neatline(ax)


def region_ring_legend_handle() -> tuple[Patch, Patch]:
    """Muestra doble del rayado del anillo: sobre relleno claro (gris) y sobre oscuro (blanco)."""
    return (
        Patch(facecolor="#FEE391", edgecolor=REGION_HATCH_COLOR, hatch=REGION_HATCH, linewidth=0.0),
        Patch(facecolor="#993404", edgecolor="white", hatch=REGION_HATCH, linewidth=0.0),
    )


def save_figure(fig: Any, figures_dir: Path, stem: str) -> None:
    """PNG a >= 300 ppp y >= 2.244 px de ancho, y PDF vectorial con las mismas medidas."""
    width_in = fig.get_size_inches()[0]
    dpi = int(max(340, math.ceil(2300 / max(width_in, 1e-6))))
    fig.savefig(figures_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")


def _net_change_classes(values: pd.Series) -> pd.Series:
    """Clase entera -2..+2 (se recorta a +-2; los cambios observados no pasan de 2 puertos)."""
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).round().astype(int).clip(-2, 2)
    return v


def plot_net_change_map(
    ax: Any,
    gdf: gpd.GeoDataFrame,
    column: str,
    crop: tuple[float, float, float, float],
    rotation_deg: float,
    *,
    scale_and_north: bool = True,
) -> list[tuple[float, float, float, float]]:
    """Mapa de cambio neto entero entre escenarios con clases discretas {-2,...,+2}."""
    clases = _net_change_classes(gdf[column])
    colores = clases.map(DELTA_CLASS_COLORS)
    zero = clases.eq(0)
    gdf.loc[zero].plot(ax=ax, color=colores[zero], edgecolor=MAP_EDGE, linewidth=0.22)
    if (~zero).any():
        gdf.loc[~zero].plot(ax=ax, color=colores[~zero], edgecolor="white", linewidth=0.45, zorder=3)
    add_region_ring_overlay(ax, gdf, fill_colors=colores)
    draw_map_outline(ax, gdf)
    set_map_frame(ax, crop)
    reservado: list[tuple[float, float, float, float]] = []
    if scale_and_north:
        reservado += add_north_arrow(ax, rotation_deg, extent=crop)
        reservado += add_scale_bar(ax, gdf, extent=crop)
    return reservado


def net_change_legend_handles(classes_present: list[int] | None = None) -> list[Patch]:
    etiquetas = {-2: "−2 ports", -1: "−1 port", 0: "No change", 1: "+1 port", 2: "+2 ports"}
    clases = classes_present if classes_present is not None else [-2, -1, 0, 1, 2]
    return [Patch(facecolor=DELTA_CLASS_COLORS[c], edgecolor=MAP_EDGE, linewidth=0.5, label=etiquetas[c]) for c in clases]


def zones_outside_frame(gdf: gpd.GeoDataFrame, column: str, crop: tuple[float, float, float, float]) -> list[tuple[str, int]]:
    """Unidades con cambio cuya superficie queda entera fuera del marco."""
    caja = shapely_box(*crop)
    cambios = gdf.loc[_net_change_classes(gdf[column]).ne(0)]
    fuera = cambios.loc[~cambios.geometry.intersects(caja)]
    return [(str(u), int(round(float(v)))) for u, v in zip(fuera["UTAM"], fuera[column])]


def fig_net_change_S2_minus_S1(
    figures_dir: Path,
    utam_map: gpd.GeoDataFrame,
    scenario_outputs: dict[str, pd.DataFrame],
    rotation_deg: float,
) -> None:
    """Figura 7 - cambio neto S2 menos S1, norte arriba, a 1.5 columnas.

    Revision de imprenta del 2026-10-11 (guia JTG, ficha Fig. 7): sin titulo interno; las lineas
    guia largas, que se cruzaban, se sustituyen por un numero dentro de cada zona (con halo) y una
    clave a la derecha con el identificador UTAM y el cambio (R1.12). Los numeros siguen el orden
    norte-sur. Las zonas cuya superficie cae entera fuera del marco urbano se listan en la clave.
    """
    if not {"S1", "S2"}.issubset(scenario_outputs):
        return
    delta = scenario_outputs["S2"][["UTAM", "ports_total"]].merge(
        scenario_outputs["S1"][["UTAM", "ports_total"]], on="UTAM", suffixes=("_S2", "_S1"),
    )
    delta["net_change"] = delta["ports_total_S2"] - delta["ports_total_S1"]
    gdf = utam_map.merge(delta[["UTAM", "net_change"]], on="UTAM", how="left")
    gdf["net_change"] = pd.to_numeric(gdf["net_change"], errors="coerce").fillna(0)
    changed = gdf.loc[gdf["net_change"].ne(0)].copy()
    if changed.empty:
        return
    crop = urban_core_extent(utam_map)
    caja = shapely_box(*crop)

    fig = plt.figure(figsize=(5.20, 4.70))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62], left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_key = fig.add_subplot(grid[0, 1])
    reservado = plot_net_change_map(ax, gdf, "net_change", crop, rotation_deg)

    dentro = changed.loc[changed.geometry.intersects(caja)].copy()
    dentro["rep"] = dentro.geometry.intersection(caja).representative_point()
    dentro["_y"] = dentro["rep"].apply(lambda p: p.y)
    dentro = dentro.sort_values("_y", ascending=False).reset_index(drop=True)
    for i, row in dentro.iterrows():
        ax.text(row["rep"].x, row["rep"].y, str(i + 1), ha="center", va="center", fontsize=7.0, fontweight="bold",
                color=NEUTRAL_DARK, path_effects=label_halo(2.0), zorder=12)
    reservado += add_city_centre_marker(ax, gdf, fontsize=7.0)
    reservado += [(p.x, p.y, 0.035 * (crop[2] - crop[0]), 0.02 * (crop[3] - crop[1])) for p in dentro["rep"]]
    add_locality_labels(
        ax, gdf,
        locality_names=("SUBA", "USAQUEN", "ENGATIVA", "FONTIBON", "KENNEDY", "BOSA", "CIUDAD BOLIVAR", "USME", "SAN CRISTOBAL"),
        municipality_names=("SOACHA", "COTA", "CHIA", "MOSQUERA", "LA CALERA"),
        fontsize=7.0, color="#6d655e", extent=crop, reserved=reservado,
    )

    def fmt(n: int) -> str:
        return f"+{n}" if n > 0 else f"−{abs(n)}"

    ax_key.axis("off")
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    y = 0.985
    ax_key.text(0.0, y, "Zones that change", ha="left", va="top", fontsize=7.5, fontweight="bold", color=PAPER_TEXT)
    y -= 0.055
    paso = 0.040
    for i, row in dentro.iterrows():
        n = int(row["net_change"])
        ax_key.add_patch(Rectangle((0.0, y - 0.024), 0.07, 0.026, facecolor=DELTA_CLASS_COLORS[int(np.clip(n, -2, 2))], edgecolor="white", linewidth=0.4))
        ax_key.text(0.035, y - 0.011, str(i + 1), ha="center", va="center", fontsize=6.8, fontweight="bold", color=NEUTRAL_DARK, path_effects=label_halo(1.6))
        ax_key.text(0.11, y - 0.011, f"{row['UTAM']}  ({fmt(n)})", ha="left", va="center", fontsize=7.0, color=PAPER_TEXT)
        y -= paso
    fuera = [(u, n) for u, n in zip(changed["UTAM"], changed["net_change"]) if u not in set(dentro["UTAM"])]
    if fuera:
        y -= 0.01
        ax_key.text(0.0, y, "Outside the frame:\n" + ", ".join(f"{u} ({fmt(int(n))})" for u, n in fuera), ha="left", va="top", fontsize=7.0, color=PAPER_TEXT)
        y -= 0.09
    y -= 0.01
    handles = net_change_legend_handles([1, -1, 0]) + [region_ring_legend_patch(),
               Line2D([0], [0], marker="*", color="none", markerfacecolor=NEUTRAL_DARK, markeredgecolor="white", markersize=8, label="Historic city centre")]
    ax_key.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.04, y), frameon=False, fontsize=7.0, handlelength=1.6, borderaxespad=0.0)
    save_figure(fig, figures_dir, "fig_net_change_S2_minus_S1")
    plt.close(fig)


def ports_per_capita_by_quartile(
    df_scen: pd.DataFrame, vulnerability_csv: Path | None, top_share: float = 0.25
) -> tuple[float, float] | None:
    """Puertos por 10.000 habitantes en el cuartil vulnerable y en el resto de zonas.

    El cuartil se define sobre las 142 zonas (cuantil 1 - top_share del indice de
    vulnerabilidad) y las razones se calculan sobre las zonas con poblacion DANE > 0.
    Es el indicador de intensidad que sustituye a la cobertura cuando esta se satura.
    """
    if vulnerability_csv is None or not Path(vulnerability_csv).exists():
        return None
    pobl = pd.read_csv(vulnerability_csv)
    if "dane_population" not in pobl.columns:
        return None
    pobl["UTAM"] = pobl["UTAM"].map(normalize_zone_code)
    d = df_scen[["UTAM", "ports_total", "vulnerability_index"]].copy()
    d["UTAM"] = d["UTAM"].map(normalize_zone_code)
    d = d.merge(pobl[["UTAM", "dane_population"]], on="UTAM", how="left")
    vq = d["vulnerability_index"] >= d["vulnerability_index"].quantile(1.0 - top_share)
    con_pob = d["dane_population"].fillna(0.0) > 0
    d, vq = d.loc[con_pob], vq.loc[con_pob]

    def razon(mask: pd.Series) -> float:
        return float(1e4 * d.loc[mask, "ports_total"].sum() / d.loc[mask, "dane_population"].sum())

    return razon(vq), razon(~vq)


def _familia_robustez(text: str) -> str:
    lower = text.lower()
    if "decay" in lower:
        return "Decay"
    if "weighting construction" in lower or "pca" in lower:
        return "Vulnerability index"
    if "trade-off sensitivity" in lower or "epsilon" in lower:
        return "Epsilon"
    if "social-weight intensity" in lower or "kappa" in lower:
        return "Kappa"
    if "minimum-service" in lower or "soft floor" in lower:
        return "Soft floor"
    return "Other"


def fig_policy_tradeoff(
    figures_dir: Path,
    tradeoff_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    s3_summary: pd.Series | None,
    percapita: dict[str, tuple[float, float] | None],
    sweep_df: pd.DataFrame | None = None,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_percapita: tuple[float, float] | None = None,
    frontier_percapita: list[tuple[str, tuple[float, float]]] | None = None,
) -> None:
    """Figura 9 - sintesis del compromiso de politica (revision de imprenta 2026-10-11).

    (a) Gini ponderado frente a perdida de fidelidad (escala simetrica-logaritmica) con S0-S2 en
    la paleta de escenarios, la frontera de tolerancia en gris y el benchmark de equidad directa
    como rombo negro rotulado; (b) puertos por 10.000 residentes por grupo de vulnerabilidad, en
    grises, con valores de dos decimales en vertical para que no choquen; (c) robustez del Gini
    ponderado con familias distinguidas por FORMA (no por los colores de los escenarios) y leyenda
    fuera de la columna x = 0; (d) salvaguarda del conjunto protegido de S3. Griegas reales.
    """
    fam_marcadores = {"Decay": "o", "Epsilon": "s", "Kappa": "^", "Vulnerability index": "D", "Soft floor": "v", "Other": "P"}
    fam_etiquetas = {"Decay": "Decay (λ)", "Epsilon": "Tolerance (ε)", "Kappa": "Social weight (κ)",
                     "Vulnerability index": "Vulnerability index", "Soft floor": "Soft floor (γ)", "Other": "Other"}
    fig = plt.figure(figsize=(7.05, 5.30), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    def estilo(ax: Any, letra: str, xlabel: str, ylabel: str) -> None:
        configure_line_axis(ax, "", xlabel, ylabel)
        ax.set_title("")
        panel_label(ax, letra, y=1.02)
        ax.xaxis.label.set_fontsize(7.5)
        ax.yaxis.label.set_fontsize(7.5)
        ax.tick_params(labelsize=7.0)

    def eps_text(value: float) -> str:
        return f"ε = {value:g}"

    # ---------------- (a) precio de la equidad ----------------
    td = tradeoff_df.copy()
    td["scenario"] = td["scenario"].astype(str)
    td = td.set_index("scenario")
    familia = []
    for name in ("S0", "S1", "S2"):
        if name in td.index:
            familia.append((name, float(td.loc[name, "relative_fidelity_loss_vs_S0_pct"]), float(td.loc[name, "accessibility_weighted_gini"])))
    if sweep_df is not None and len(sweep_df):
        for _, r in sweep_df.sort_values("epsilon").iterrows():
            familia.append((f"eps={float(r['epsilon'])}", float(r["perdida_fidelidad_vs_S0_pct"]), float(r["gini_ponderado_6d"])))
    estilo(ax_a, "(a)", "Fidelity loss relative to S0 (%, symmetric log scale)", "Vulnerability-weighted Gini")
    ax_a.grid(axis="x", linestyle="-", linewidth=0.6, color=PAPER_GRID)
    ax_a.set_xscale("symlog", linthresh=1.0)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    xs = [f[1] for f in familia]; ys = [f[2] for f in familia]
    ax_a.plot(xs, ys, color=NEUTRAL_LIGHT, linewidth=1.3, zorder=1)

    def punto(ax: Any, name: str, x: float, y: float, grande: bool = True) -> None:
        if name in SCENARIO_COLORS:
            ax.scatter(x, y, s=34 if grande else 20, marker=SCENARIO_MARKERS[name], color=SCENARIO_COLORS[name], edgecolor="white", linewidth=0.6, zorder=4)
        else:
            ax.scatter(x, y, s=18 if grande else 12, marker="o", facecolor="white", edgecolor=NEUTRAL_MID, linewidth=0.9, zorder=3)

    for name, x, y in familia:
        punto(ax_a, name, x, y)
    etiquetas_a = {"S0": (2, 6), "S1": (-4, 7), "S2": (4, 7)}
    for name, x, y in familia:
        if name in etiquetas_a:
            ax_a.annotate(name, xy=(x, y), xytext=etiquetas_a[name], textcoords="offset points", fontsize=7.0, color=PAPER_TEXT)
        elif name.startswith("eps=") and float(name.split("=")[1]) in (1.0, 10.0, 100.0):
            ax_a.annotate(eps_text(float(name.split("=")[1])), xy=(x, y), xytext=(5, 3), textcoords="offset points", fontsize=7.0, color=NEUTRAL_DARK)
    if benchmark_df is not None and len(benchmark_df):
        for k, (_, r) in enumerate(benchmark_df.iterrows()):
            bx, by = float(r["perdida_fidelidad_vs_S0_pct"]), float(r["gini_ponderado_6d"])
            ax_a.scatter(bx, by, s=30, marker="D", color=NEUTRAL_DARK, edgecolor="white", linewidth=0.6, zorder=4)
            if k == 0:
                    ax_a.annotate("Direct-equity\nbenchmark", xy=(bx, by), xytext=(-7, -1), textcoords="offset points", ha="right", va="center",
                              fontsize=7.0, color=NEUTRAL_DARK)
    if familia:
        y_all = ys + ([float(v) for v in benchmark_df["gini_ponderado_6d"]] if benchmark_df is not None and len(benchmark_df) else [])
        pad = 0.06 * (max(y_all) - min(y_all) or 0.01)
        ax_a.set_ylim(min(y_all) - pad, max(y_all) + 2.2 * pad)
        publicada = [f for f in familia if not f[0].startswith("eps=") or float(f[0].split("=")[1]) <= 0.30]
        xz = [f[1] for f in publicada]; yz = [f[2] for f in publicada]
        ins = ax_a.inset_axes([0.17, 0.12, 0.42, 0.34])
        ins.plot(xz, yz, color=NEUTRAL_LIGHT, linewidth=1.1, zorder=1)
        for name, x, y in publicada:
            punto(ins, name, x, y, grande=False)
        ins.set_xlim(-1.5, max(xz) + 2.0)
        yp = 0.15 * (max(yz) - min(yz) or 1e-4)
        ins.set_ylim(min(yz) - yp, max(yz) + yp)
        ins.tick_params(labelsize=6.6, length=2)
        ins.set_facecolor("white")
        for spine in ins.spines.values():
            spine.set_edgecolor(NEUTRAL_MID)
            spine.set_linewidth(0.6)
        ins.set_title("ε ≤ 0.30", fontsize=7.0, color=NEUTRAL_DARK, pad=2, loc="left")
        ins.yaxis.set_major_locator(plt.MaxNLocator(3))
        ins.xaxis.set_major_locator(plt.MaxNLocator(4))

    # ---------------- (b) puertos per capita ----------------
    estilo(ax_b, "(b)", "", "Ports per 10,000 residents")
    ax_b.set_axisbelow(True)
    cats: list[tuple[str, tuple[float, float]]] = [(n, percapita[n]) for n in ("S0", "S1", "S2") if percapita.get(n) is not None]
    for etiqueta, valores in (frontier_percapita or []):
        texto = str(etiqueta)
        if texto.startswith("eps"):
            texto = eps_text(float(texto.split()[-1].split("\n")[-1]))
        cats.append((texto, valores))
    if benchmark_percapita is not None:
        cats.append(("Direct\nequity", benchmark_percapita))
    if cats:
        xpos = np.arange(len(cats)); w = 0.38
        vuln = [c[1][0] for c in cats]; resto = [c[1][1] for c in cats]
        ax_b.bar(xpos - w / 2, vuln, width=w, color=NEUTRAL_DARK, label="Most vulnerable quartile (36 zones)", zorder=2)
        ax_b.bar(xpos + w / 2, resto, width=w, color=NEUTRAL_LIGHT, edgecolor=NEUTRAL_MID, linewidth=0.5, label="Remaining zones", zorder=2)
        for i, (v, r) in enumerate(zip(vuln, resto)):
            ax_b.annotate(f"{v:.2f}", xy=(i - w / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", rotation=90, fontsize=6.8, color=PAPER_TEXT)
            ax_b.annotate(f"{r:.2f}", xy=(i + w / 2, r), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", rotation=90, fontsize=6.8, color=PAPER_TEXT)
        ax_b.set_xticks(xpos)
        ax_b.set_xticklabels([c[0] for c in cats], fontsize=7.0)
        ax_b.set_ylim(0, max(vuln + resto) * 1.45)
        ax_b.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax_b.legend(loc="upper left", frameon=False, fontsize=7.0, ncol=1)

    # ---------------- (c) robustez del Gini ----------------
    estilo(ax_c, "(c)", "Change in weighted Gini vs S2 (basis points)", "")
    ax_c.grid(axis="y", visible=False)
    ax_c.grid(axis="x", linestyle="-", linewidth=0.6, color=PAPER_GRID)
    rb_df = robust_df.copy()
    rb_df = rb_df.loc[pd.to_numeric(rb_df["delta_weighted_gini"], errors="coerce").notna()].copy()
    if len(rb_df):
        rb_df["bp"] = 1e4 * pd.to_numeric(rb_df["delta_weighted_gini"], errors="coerce")
        rb_df["label"] = rb_df["alternative_specification"].astype(str).map(shorten_robustness_label)
        rb_df["fam"] = rb_df["interpretation"].astype(str).map(_familia_robustez)
        rb_df = rb_df.sort_values("bp").reset_index(drop=True)
        ypos = np.arange(len(rb_df))
        ax_c.axvline(0.0, linestyle=(0, (4, 2)), color=NEUTRAL_MID, linewidth=0.9, zorder=1)
        ax_c.hlines(ypos, 0.0, rb_df["bp"], color=NEUTRAL_LIGHT, linewidth=1.4, zorder=2)
        for fam, sub in rb_df.groupby("fam"):
            ax_c.scatter(sub["bp"], sub.index, s=22, marker=fam_marcadores.get(fam, "o"), color=NEUTRAL_DARK, edgecolor="white", linewidth=0.5, zorder=3)
        ax_c.set_yticks(ypos)
        ax_c.set_yticklabels(rb_df["label"], fontsize=7.0)
        ax_c.tick_params(axis="y", pad=2)
        xmin, xmax = float(rb_df["bp"].min()), float(rb_df["bp"].max())
        ax_c.set_xlim(xmin - 0.08 * (xmax - xmin), xmax + 0.08 * (xmax - xmin))
        handles = [Line2D([0], [0], marker=fam_marcadores[f], color="none", markerfacecolor=NEUTRAL_DARK, markeredgecolor="white", markersize=5.5, label=fam_etiquetas[f])
                   for f in ("Decay", "Epsilon", "Kappa", "Vulnerability index", "Soft floor") if f in set(rb_df["fam"])]
        ax_c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=6.8,
                    handletextpad=0.2, columnspacing=0.8)

    # ---------------- (d) salvaguarda de S3 ----------------
    if s3_summary is not None and np.isfinite(pd.to_numeric(s3_summary.get("s3_floor_value"), errors="coerce")):
        floor_value = float(pd.to_numeric(s3_summary["s3_floor_value"], errors="coerce"))
        p_min = float(pd.to_numeric(s3_summary["s3_protected_min_access"], errors="coerce"))
        p_10 = float(pd.to_numeric(s3_summary["s3_protected_p10_access"], errors="coerce"))
        p_25 = float(pd.to_numeric(s3_summary["s3_protected_p25_access"], errors="coerce"))
        p_mean = float(pd.to_numeric(s3_summary["s3_protected_mean_access"], errors="coerce"))
        estilo(ax_d, "(d)", "Protected-set accessibility (decay-weighted ports)", "")
        ax_d.grid(axis="x", linestyle="-", linewidth=0.6, color=PAPER_GRID)
        ax_d.grid(axis="y", visible=False)
        ax_d.tick_params(axis="y", left=False, labelleft=False)
        ax_d.spines["left"].set_visible(False)
        ax_d.hlines(0.0, p_min, p_mean, color=NEUTRAL_LIGHT, linewidth=2.4, zorder=1)
        ax_d.axvline(floor_value, linestyle=(0, (4, 2)), color=NEUTRAL_MID, linewidth=1.0, zorder=0)
        for label, value_, marker in (("Min", p_min, "o"), ("P10", p_10, "s"), ("P25", p_25, "^"), ("Mean", p_mean, "D")):
            ax_d.scatter(value_, 0.0, s=34, marker=marker, color=NEUTRAL_DARK, edgecolor="white", linewidth=0.7, zorder=3)
            ax_d.annotate(f"{label}\n{value_:.1f}", xy=(value_, 0.0), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                          fontsize=7.0, color=PAPER_TEXT)
        ax_d.annotate(f"Soft floor\n{floor_value:.1f}", xy=(floor_value, 0.0), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                      fontsize=7.0, color=NEUTRAL_DARK, bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        x_span = max(p_mean - p_min, 1.0)
        x_pad = max(12.0, 0.12 * x_span)
        ax_d.set_xlim(min(p_min, floor_value) - x_pad, p_mean + x_pad)
        ax_d.set_ylim(-0.5, 0.5)
    else:
        ax_d.axis("off")
        panel_label(ax_d, "(d)")
        ax_d.text(0.0, 0.9, "S3 protected-set safeguard: not available for this run", transform=ax_d.transAxes, fontsize=7, color="#6d655e")

    save_figure(fig, figures_dir, "fig_policy_tradeoff")
    plt.close(fig)


def export_figures(
    paths: ProjectPaths,
    utam: gpd.GeoDataFrame,
    df_inputs: pd.DataFrame,
    loss_hist: pd.DataFrame | None,
    scenario_outputs: dict[str, pd.DataFrame],
    protected_mask_s3: np.ndarray,
    robust_df: pd.DataFrame,
) -> None:
    utam_map, rotation_deg, _rotation_origin = prepare_context_layers(utam)
    focus_extent = derive_focus_extent(utam_map)
    gdf_priority = utam_map.merge(df_inputs[["UTAM", "priority_norm", "vulnerability_index"]], on="UTAM", how="left")
    protected_df = pd.DataFrame({"UTAM": df_inputs["UTAM"], "protected_s3": protected_mask_s3.astype(int)})
    gdf_protected = utam_map.merge(protected_df, on="UTAM", how="left")

    priority_norm = Normalize(
        vmin=float(np.nanmin(gdf_priority["priority_norm"].to_numpy(dtype=float))),
        vmax=float(np.nanmax(gdf_priority["priority_norm"].to_numpy(dtype=float))),
    )
    vulnerability_norm = Normalize(
        vmin=float(np.nanmin(gdf_priority["vulnerability_index"].to_numpy(dtype=float))),
        vmax=float(np.nanmax(gdf_priority["vulnerability_index"].to_numpy(dtype=float))),
    )

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), constrained_layout=True)
    plot_map_panel(
        ax,
        gdf_priority,
        "priority_norm",
        "Learned priority surface across Bogota",
        SEQUENTIAL_CMAP,
        norm=priority_norm,
        show_scale_bar=True,
        focus_extent=focus_extent,
        show_focus_inset=True,
    )
    add_horizontal_colorbar(fig, ax, SEQUENTIAL_CMAP, priority_norm, "Normalized priority")
    fig.savefig(paths.figures_dir / "fig_priority_map.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), constrained_layout=True)
    plot_binary_map_panel(
        ax,
        gdf_protected,
        "protected_s3",
        "Protected S3 geography for the soft-floor policy",
        show_scale_bar=True,
        focus_extent=focus_extent,
        show_focus_box=True,
    )
    ax.legend(
        handles=[
            Patch(facecolor=PROTECTED_COLORS[1], edgecolor="none", label="Protected UTAM"),
            Patch(facecolor=PROTECTED_COLORS[0], edgecolor="none", label="Unprotected UTAM"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(paths.figures_dir / "fig_S3_protected_set.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Unidades sin dato censal: el modelo les imputa la mediana del indice (build_inputs); la
    # Figura 4b las muestra como "sin dato censal" para no presentar la imputacion como medida.
    vuln_csv = paths.input_dir / "utam_social_vulnerability.csv"
    imputed_units: set[str] | None = None
    if vuln_csv.exists():
        vuln_src = pd.read_csv(vuln_csv)
        vuln_src["UTAM"] = vuln_src["UTAM"].map(normalize_zone_code)
        conocidas = set(vuln_src.loc[vuln_src["vulnerability_index"].notna(), "UTAM"])
        imputed_units = set(df_inputs["UTAM"].map(normalize_zone_code)) - conocidas
    fig_social_context_panels(paths.figures_dir, utam_map, df_inputs, protected_mask_s3, rotation_deg, imputed_units=imputed_units)

    if loss_hist is not None and len(loss_hist):
        fig, ax = plt.subplots(1, 1, figsize=(11, 4.4), constrained_layout=True)
        cols_lower = {str(column).lower(): column for column in loss_hist.columns}
        x = (
            pd.to_numeric(loss_hist[cols_lower["epoch"]], errors="coerce").to_numpy(dtype=float)
            if "epoch" in cols_lower
            else np.arange(len(loss_hist), dtype=float)
        )
        for column in loss_hist.columns:
            if str(column).lower() == "epoch" or not pd.api.types.is_numeric_dtype(loss_hist[column]):
                continue
            values = pd.to_numeric(loss_hist[column], errors="coerce").to_numpy(dtype=float)
            color = SCENARIO_COLORS["S0"] if "train" in str(column).lower() else SCENARIO_COLORS["S2"]
            ax.plot(x, values, color=color, linewidth=2.2, alpha=0.25)
            ax.plot(
                x,
                pd.Series(values).rolling(window=5, min_periods=1).mean(),
                color=color,
                linewidth=2.6,
                label=str(column).replace("_", " "),
            )
        configure_line_axis(ax, "GNN learning history", "Iteration", "Loss / error")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.savefig(paths.figures_dir / "fig_gnn_learning_history.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Figura 5 (revision de imprenta 2026-10-11): sin titulos internos, rotulos "(a)", series que no
    # son escenarios en grises distinguibles por forma y tipo de linea, franjas horarias en grises.
    fig, axes = plt.subplots(1, 2, figsize=(5.50, 2.55), constrained_layout=True)
    attention = pd.read_csv(paths.input_dir / "temporal_attention_by_utam.csv")
    attention = attention.rename(columns={find_utam_col(attention): "UTAM"})
    attention["UTAM"] = attention["UTAM"].map(normalize_zone_code)
    hour_cols = [f"hour_{hour}" for hour in range(24) if f"hour_{hour}" in attention.columns]
    priority_raw = pd.read_csv(paths.input_dir / "priority_utam_scores.csv")
    priority_raw = priority_raw.rename(columns={find_utam_col(priority_raw): "UTAM"})
    priority_raw["UTAM"] = priority_raw["UTAM"].map(normalize_zone_code)
    trip_cols = [f"trip_hour_{hour:02d}" for hour in range(24) if f"trip_hour_{hour:02d}" in priority_raw.columns]
    hours = np.arange(24)

    def estilo_horario(axis: Any, letra: str) -> None:
        configure_line_axis(axis, "", "Hour of day", "Relative intensity (max = 1)")
        axis.set_title("")
        panel_label(axis, letra, y=1.02)
        axis.set_xticks(np.arange(0, 25, 3))
        axis.set_xticks(hours, minor=True)
        axis.set_xlim(-0.5, 23.5)
        axis.set_ylim(-0.03, 1.17)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.tick_params(labelsize=7.0)
        axis.xaxis.label.set_fontsize(7.5)
        axis.yaxis.label.set_fontsize(7.5)

    if hour_cols:
        mean_attention = attention[hour_cols].mean(axis=0).to_numpy(dtype=float)
        mean_attention = mean_attention / max(float(np.nanmax(mean_attention)), 1e-9)
        smoothed_attention = pd.Series(mean_attention).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
        trip_profile = None
        if len(trip_cols) == 24:
            trip_profile = priority_raw[trip_cols].mean(axis=0).to_numpy(dtype=float)
            trip_profile = trip_profile / max(float(np.nanmax(trip_profile)), 1e-9)
        add_time_window_bands(axes[0])
        axes[0].plot(hours, smoothed_attention, color=NEUTRAL_DARK, linewidth=1.6, label="ST-GNN attention")
        axes[0].scatter(hours, mean_attention, color=NEUTRAL_DARK, s=9, zorder=3)
        if trip_profile is not None:
            axes[0].plot(hours, trip_profile, color=NEUTRAL_MID, linewidth=1.4, linestyle=(0, (4, 2)), label="Observed trip profile")
        estilo_horario(axes[0], "(a)")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=7.0, handlelength=2.2)

        attention_vuln = attention.merge(df_inputs[["UTAM", "vulnerability_index"]], on="UTAM", how="left")
        top_cut = float(attention_vuln["vulnerability_index"].quantile(0.75))
        low_cut = float(attention_vuln["vulnerability_index"].quantile(0.25))
        high_profile = attention_vuln.loc[attention_vuln["vulnerability_index"] >= top_cut, hour_cols].mean(axis=0).to_numpy(dtype=float)
        low_profile = attention_vuln.loc[attention_vuln["vulnerability_index"] <= low_cut, hour_cols].mean(axis=0).to_numpy(dtype=float)
        scale = max(float(np.nanmax(high_profile)), float(np.nanmax(low_profile)), 1e-9)
        high_profile = high_profile / scale
        low_profile = low_profile / scale
        add_time_window_bands(axes[1])
        axes[1].plot(hours, pd.Series(high_profile).rolling(window=3, center=True, min_periods=1).mean(), color=NEUTRAL_DARK, linewidth=1.6, label="Most vulnerable quartile")
        axes[1].plot(hours, pd.Series(low_profile).rolling(window=3, center=True, min_periods=1).mean(), color=NEUTRAL_MID, linewidth=1.4, linestyle=(0, (4, 2)), label="Least vulnerable quartile")
        axes[1].scatter(hours, high_profile, color=NEUTRAL_DARK, s=9, zorder=3)
        axes[1].scatter(hours, low_profile, facecolor="white", edgecolor=NEUTRAL_MID, linewidth=0.8, s=11, zorder=3)
        estilo_horario(axes[1], "(b)")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=7.0, handlelength=2.2)
    save_figure(fig, paths.figures_dir, "fig_attention_profile")
    plt.close(fig)

    ordered_scenarios = [name for name in ["S0", "S1", "S2", "S3"] if name in scenario_outputs]
    ports_norm = make_ports_norm(scenario_outputs)

    if ports_norm is not None:
        for scenario_name in ordered_scenarios:
            gdf = utam_map.merge(scenario_outputs[scenario_name][["UTAM", "ports_total", "accessibility"]], on="UTAM", how="left")
            fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), constrained_layout=True)
            plot_map_panel(
                ax,
                gdf,
                "ports_total",
                f"{scenario_name} total port allocation",
                SEQUENTIAL_CMAP,
                norm=ports_norm,
                zero_fill=MAP_DELTA_ZERO,
                show_scale_bar=True,
                focus_extent=focus_extent,
                show_focus_box=True,
            )
            add_region_ring_overlay(ax, gdf)
            add_horizontal_colorbar(fig, ax, SEQUENTIAL_CMAP, ports_norm, "Charging ports", spacing="uniform")
            fig.savefig(paths.figures_dir / f"fig_allocation_{scenario_name}.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    poblacion_csv = paths.input_dir / "utam_social_vulnerability.csv"
    poblacion = pd.read_csv(poblacion_csv) if poblacion_csv.exists() else None
    fig_allocation_composite(paths.figures_dir, utam_map, scenario_outputs, rotation_deg, population=poblacion)
    fig_net_change_S2_minus_S1(paths.figures_dir, utam_map, scenario_outputs, rotation_deg)

    delta_specs: list[tuple[str, str, str, str]] = []
    if {"S0", "S1"}.issubset(scenario_outputs):
        delta_specs.append(("S1", "S0", "delta_s1_s0", "S1 minus S0"))
    if {"S1", "S2"}.issubset(scenario_outputs):
        delta_specs.append(("S2", "S1", "delta_s2_s1", "S2 minus S1"))
    if {"S2", "S3"}.issubset(scenario_outputs):
        delta_specs.append(("S3", "S2", "delta_s3_s2", "S3 minus S2"))
    if {"S0", "S3"}.issubset(scenario_outputs):
        delta_specs.append(("S3", "S0", "delta_s3_s0", "S3 minus S0"))

    growth_long = load_optional_growth_long(paths)
    if growth_long is not None and len(growth_long):
        growth_long["year"] = pd.to_numeric(growth_long["year"], errors="coerce")
        growth_long["value"] = pd.to_numeric(growth_long["value"], errors="coerce")
        annual_specs = []
        for year in sorted(growth_long["year"].dropna().astype(int).unique()):
            year_df = growth_long.loc[growth_long["year"].eq(year), ["UTAM", "value"]]
            annual_specs.append((year_df, str(year)))
        if len(annual_specs) > 1:
            growth_maps: list[tuple[gpd.GeoDataFrame, str]] = []
            previous_df, previous_label = annual_specs[0]
            for current_df, current_label in annual_specs[1:]:
                diff_df = current_df.merge(previous_df, on="UTAM", how="left", suffixes=("_curr", "_prev"))
                diff_df["delta_value"] = diff_df["value_curr"] - diff_df["value_prev"]
                growth_maps.append((utam_map.merge(diff_df[["UTAM", "delta_value"]], on="UTAM", how="left"), f"{current_label} minus {previous_label}"))
                previous_df, previous_label = current_df, current_label
        else:
            growth_maps = []
    else:
        growth_maps = []
        for current_name, previous_name, delta_column, label in delta_specs:
            delta = scenario_outputs[current_name][["UTAM", "ports_total"]].merge(
                scenario_outputs[previous_name][["UTAM", "ports_total"]],
                on="UTAM",
                suffixes=(f"_{current_name}", f"_{previous_name}"),
            )
            delta[delta_column] = delta[f"ports_total_{current_name}"] - delta[f"ports_total_{previous_name}"]
            growth_maps.append((utam_map.merge(delta[["UTAM", delta_column]], on="UTAM", how="left"), label))
    filtered_growth_maps: list[tuple[gpd.GeoDataFrame, str, str]] = []
    for gdf_delta, label in growth_maps:
        delta_cols = [column for column in gdf_delta.columns if str(column).startswith("delta_")]
        if not delta_cols:
            continue
        delta_col = delta_cols[-1]
        delta_values = pd.to_numeric(gdf_delta[delta_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(delta_values) == 0 or float(np.nanmax(np.abs(delta_values))) <= 1e-9:
            continue
        filtered_growth_maps.append((gdf_delta, label, delta_col))
    growth_maps = [(gdf_delta, label) for gdf_delta, label, _ in filtered_growth_maps]

    if growth_maps:
        all_growth_values = np.concatenate(
            [
                pd.to_numeric(
                    gdf.select_dtypes(include=[np.number]).iloc[:, -1],
                    errors="coerce",
                ).fillna(0.0).to_numpy(dtype=float)
                for gdf, _ in growth_maps
            ]
        )
        delta_max = float(np.nanmax(np.abs(all_growth_values))) if len(all_growth_values) else 1.0
        delta_max = delta_max if np.isfinite(delta_max) and delta_max > 0 else 1.0
        delta_norm = TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max)

        # "S2 minus S1" is produced by fig_net_change_S2_minus_S1 (labelled design).
        label_to_path = {
            "S3 minus S2": paths.figures_dir / "fig_net_change_S3_minus_S2.png",
        }
        for gdf_delta, label, column in filtered_growth_maps:
            if label not in label_to_path:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), constrained_layout=True)
            plot_map_panel(
                ax,
                gdf_delta,
                column,
                f"Net reallocation in ports: {label}",
                DIVERGING_CMAP,
                norm=delta_norm,
                linewidth=0.45,
                edgecolor=MAP_DELTA_EDGE,
                zero_fill=MAP_DELTA_ZERO,
                show_scale_bar=True,
                focus_extent=focus_extent,
                show_focus_inset=True,
            )
            add_horizontal_colorbar(fig, ax, DIVERGING_CMAP, delta_norm, "Net change in ports")
            fig.savefig(label_to_path[label], dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # Suplemento, Figura C1 (revision de imprenta 2026-10-11): norte arriba, el MISMO marco
        # urbano que la Figura 7, norte y escala en cada panel, y clases enteras {-2,...,+2} en
        # lugar de una rampa continua (los cambios son enteros). Las unidades con cambio que caen
        # enteras fuera del marco se enumeran bajo cada panel.
        n_maps = len(filtered_growth_maps)
        crop = urban_core_extent(utam_map)
        fig = plt.figure(figsize=(7.00, 4.05))
        grid = fig.add_gridspec(3, n_maps, height_ratios=[1.0, 0.20, 0.10], left=0.005, right=0.995, top=0.95, bottom=0.0,
                                wspace=0.05, hspace=0.03)
        clases_presentes: set[int] = set()
        for idx, (gdf_delta, label, column) in enumerate(filtered_growth_maps):
            axis = fig.add_subplot(grid[0, idx])
            plot_net_change_map(axis, gdf_delta, column, crop, rotation_deg)
            if idx == 0:
                add_city_centre_marker(axis, gdf_delta, fontsize=7.0)
            panel_label(axis, f"({chr(97 + idx)}) {label.replace(' minus ', ' − ')}")
            clases_presentes.update(int(c) for c in _net_change_classes(gdf_delta[column]).unique())
            fuera = zones_outside_frame(gdf_delta, column, crop)
            nota_ax = fig.add_subplot(grid[1, idx])
            nota_ax.axis("off")
            if fuera:
                texto = "Outside frame:\n" + "\n".join(f"{u} ({'+' if n > 0 else '−'}{abs(n)})" for u, n in fuera)
            else:
                texto = "Outside frame: none"
            nota_ax.text(0.0, 1.0, texto, ha="left", va="top", fontsize=6.6, color=PAPER_TEXT, wrap=True, transform=nota_ax.transAxes)
        legend_ax = fig.add_subplot(grid[2, :])
        legend_ax.axis("off")
        handles = net_change_legend_handles(sorted(clases_presentes)) + [region_ring_legend_patch()]
        legend_ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=7.0,
                         title="Net change in charging ports between scenarios", title_fontsize=7.5)
        figure_name = "fig_growth_over_time" if growth_long is not None else "fig_growth_progression"
        save_figure(fig, paths.figures_dir, figure_name)
        plt.close(fig)

    official_growth = load_official_ev_growth(paths)
    if official_growth is not None and len(official_growth):
        # Figura 3 (revision de imprenta 2026-10-11, guia JTG ficha Fig. 3). Los intervalos entre
        # cortes oficiales miden 8 a 14 meses, asi que (b) muestra adiciones ANUALIZADAS
        # (vehiculos por ano = adicion / anos del intervalo) y la linea discontinua es la adicion
        # anual media del periodo (stock final - stock inicial) / anos; ambas en la misma unidad.
        # Sin porcentajes dentro de las barras ni titulos internos; series en gris (la paleta de
        # escenarios queda reservada).
        dates = pd.to_datetime(official_growth["checkpoint_date"])
        counts = official_growth["electric_vehicles"].to_numpy(dtype=float)
        additions = official_growth["annual_addition"].to_numpy(dtype=float)
        years_float = (dates - dates.iloc[0]).dt.days.to_numpy(dtype=float) / 365.25 + dates.iloc[0].year + (dates.iloc[0].dayofyear / 365.25)
        interval_years = np.diff((dates - dates.iloc[0]).dt.days.to_numpy(dtype=float)) / 365.25
        annualised = additions[1:] / interval_years
        span_years = float((dates.iloc[-1] - dates.iloc[0]).days) / 365.25
        mean_annual = float(counts[-1] - counts[0]) / round(span_years)
        fig, axes = plt.subplots(1, 2, figsize=(5.50, 2.45), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)
        miles = FuncFormatter(lambda v, _pos: f"{int(round(v)):,}")
        etiquetas = [d.strftime("%b\n%Y") for d in dates]
        axes[0].set_axisbelow(True)
        axes[0].plot(years_float, counts, color=NEUTRAL_DARK, linewidth=1.5, zorder=2)
        axes[0].scatter(years_float, counts, s=20, color=NEUTRAL_DARK, edgecolor="white", linewidth=0.8, zorder=3)
        for k, (x_val, y_val) in enumerate(zip(years_float, counts)):
            axes[0].annotate(f"{int(round(y_val)):,}", xy=(x_val, y_val), xytext=(0, 5) if k == 0 else (-4, 3), textcoords="offset points",
                             ha="center" if k == 0 else "right", va="bottom", fontsize=7.0, color=PAPER_TEXT)
        configure_line_axis(axes[0], "", "Official checkpoint", "Registered EVs (vehicles)")
        axes[0].set_xticks(years_float)
        axes[0].set_xticklabels(etiquetas)
        axes[0].set_xlim(years_float[0] - 0.4, years_float[-1] + 0.4)
        axes[0].set_ylim(0.0, float(np.nanmax(counts)) * 1.15)
        axes[0].yaxis.set_major_formatter(miles)

        centros = 0.5 * (years_float[1:] + years_float[:-1])
        anchos = 0.86 * np.diff(years_float)
        axes[1].set_axisbelow(True)
        axes[1].bar(centros, annualised, width=anchos, color=NEUTRAL_LIGHT, edgecolor=NEUTRAL_MID, linewidth=0.5, zorder=2)
        axes[1].axhline(mean_annual, linestyle=(0, (4, 2)), color=NEUTRAL_DARK, linewidth=1.0, zorder=3)
        for x_val, y_val in zip(centros, annualised):
            axes[1].annotate(f"{int(round(y_val)):,}", xy=(x_val, y_val), xytext=(0, -3), textcoords="offset points",
                             ha="center", va="top", fontsize=7.0, color=PAPER_TEXT, zorder=4)
        meses = [int(round(12 * v)) for v in interval_years]
        configure_line_axis(axes[1], "", "Interval between checkpoints (months)", "EVs added per year (vehicles/year)")
        axes[1].set_xticks(centros)
        axes[1].set_xticklabels([f"{m} mo" for m in meses])
        axes[1].set_xlim(years_float[0], years_float[-1])
        axes[1].set_ylim(0.0, max(float(np.nanmax(annualised)), mean_annual) * 1.18)
        axes[1].yaxis.set_major_formatter(miles)
        axes[1].text(years_float[0] + 0.08, mean_annual, f"mean {int(round(mean_annual)):,}", ha="left", va="bottom",
                     fontsize=7.0, color=NEUTRAL_DARK, bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}, zorder=5)
        for axis, letra in zip(axes, ("(a)", "(b)")):
            axis.set_title("")
            panel_label(axis, letra, y=1.02)
            axis.tick_params(labelsize=7.0)
            axis.xaxis.label.set_fontsize(7.5)
            axis.yaxis.label.set_fontsize(7.5)
        save_figure(fig, paths.figures_dir, "fig_official_ev_growth")
        plt.close(fig)

    # Figura 8 (revision de imprenta 2026-10-11). Las cuatro curvas de Lorenz coinciden a escala
    # completa (diferencia vertical maxima < 0.001), asi que (a) muestra las curvas de S0 con ambas
    # ponderaciones y (b)-(c) la diferencia de cada escenario frente a S0, en puntos porcentuales de
    # la proporcion acumulada, calculada sobre una malla comun por interpolacion de las curvas
    # empiricas. Ningun dato nuevo: son las mismas curvas, restadas.
    grid_x = np.linspace(0.0, 1.0, 501)
    curvas: dict[tuple[str, str], np.ndarray] = {}
    for scenario_name in ordered_scenarios:
        df_scenario = scenario_outputs[scenario_name]
        xu, yu = lorenz_curve(df_scenario["accessibility"])
        xw, yw = weighted_lorenz(df_scenario["accessibility"], 1.0 + df_scenario["vulnerability_index"])
        curvas[(scenario_name, "u")] = np.interp(grid_x, np.asarray(xu, dtype=float), np.asarray(yu, dtype=float))
        curvas[(scenario_name, "w")] = np.interp(grid_x, np.asarray(xw, dtype=float), np.asarray(yw, dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]}, constrained_layout=True)
    base = ordered_scenarios[0]
    axes[0].plot([0, 1], [0, 1], linestyle=(0, (1.5, 1.5)), color=NEUTRAL_MID, linewidth=0.9, label="Equality")
    axes[0].plot(grid_x, curvas[(base, "u")], color=NEUTRAL_DARK, linewidth=1.5, label=f"{base}, unweighted")
    axes[0].plot(grid_x, curvas[(base, "w")], color=NEUTRAL_MID, linewidth=1.3, linestyle=(0, (4, 2)), label=f"{base}, weighted")
    axes[0].set_anchor("N")
    configure_line_axis(axes[0], "", "Cumulative share of zones or weight", "Cumulative share of accessibility")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=2.0, borderaxespad=0.2)
    limite = 0.0
    for axis, peso, eje in ((axes[1], "u", "Cumulative share of zones"), (axes[2], "w", "Cumulative share of vulnerability weight")):
        axis.axhline(0.0, color=NEUTRAL_MID, linewidth=0.8, zorder=1)
        for scenario_name in ordered_scenarios[1:]:
            dif = 100.0 * (curvas[(scenario_name, peso)] - curvas[(base, peso)])
            limite = max(limite, float(np.nanmax(np.abs(dif))))
            axis.plot(grid_x, dif, color=SCENARIO_COLORS.get(scenario_name, NEUTRAL_DARK), linewidth=1.3,
                      linestyle=SCENARIO_LINESTYLES.get(scenario_name, "-"), label=f"{scenario_name} − {base}")
        configure_line_axis(axis, "", eje, f"Difference from {base} (percentage points)")
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    for axis in axes[1:]:
        axis.set_ylim(-0.12 * limite, 1.45 * limite)
    axes[1].legend(loc="upper center", frameon=False, fontsize=6.8, ncol=3, handlelength=2.2, columnspacing=0.7, handletextpad=0.4)
    for axis, letra in zip(axes, ("(a)", "(b)", "(c)")):
        axis.set_title("")
        panel_label(axis, letra, y=1.02)
        axis.tick_params(labelsize=7.0)
        axis.xaxis.label.set_fontsize(7.5)
        axis.yaxis.label.set_fontsize(7.5)
    save_figure(fig, paths.figures_dir, "fig_lorenz_curves")
    plt.close(fig)

    tradeoff_path = paths.tables_dir / "table_policy_tradeoff.csv"
    if tradeoff_path.exists():
        tradeoff_df = pd.read_csv(tradeoff_path)
        for column in ["relative_fidelity_loss_vs_S0_pct", "accessibility_weighted_gini", "vulnerable_coverage_pct"]:
            if column in tradeoff_df.columns:
                tradeoff_df[column] = pd.to_numeric(tradeoff_df[column], errors="coerce")
        tradeoff_df["relative_fidelity_loss_vs_S0_pct"] = tradeoff_df["relative_fidelity_loss_vs_S0_pct"].fillna(0.0)
        s3_summary: pd.Series | None = None
        equity_metrics_path = paths.tables_dir / "table_equity_metrics_final.csv"
        if equity_metrics_path.exists():
            equity_metrics_df = pd.read_csv(equity_metrics_path)
            if "scenario" in equity_metrics_df.columns and equity_metrics_df["scenario"].astype(str).eq("S3").any():
                s3_summary = equity_metrics_df.loc[equity_metrics_df["scenario"].astype(str).eq("S3")].iloc[0]
        vulnerability_csv = paths.input_dir / "utam_social_vulnerability.csv"
        percapita = {
            name: ports_per_capita_by_quartile(df_scen, vulnerability_csv, 0.25)
            for name, df_scen in scenario_outputs.items()
        }
        fig_policy_tradeoff(paths.figures_dir, tradeoff_df, robust_df, s3_summary, percapita)

    if len(robust_df):
        robust_plot = robust_df.copy()
        robust_plot["alternative_specification"] = robust_plot["alternative_specification"].astype(str)
        robust_plot["display_label"] = robust_plot["alternative_specification"].map(shorten_robustness_label)
        robust_plot["delta_weighted_gini_bp"] = 10000.0 * pd.to_numeric(robust_plot["delta_weighted_gini"], errors="coerce")
        # Las especificaciones sin corrida (p. ej., la matriz de tiempos alternativa cuando no
        # esta disponible) no se dibujan como filas vacias; quedan en la tabla CSV.
        robust_plot = robust_plot.loc[robust_plot["delta_weighted_gini_bp"].notna()]
        robust_plot = robust_plot.sort_values("delta_weighted_gini", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(robust_plot))
        # Suplemento, Figura C2 (2026-09-02): al ancho de impresion (7.0 in),
        # con el eje y compartido para que las 13 etiquetas se lean una sola
        # vez a 8.4 pt; los titulos se acortan porque el eje x ya dice contra
        # que se compara.
        # Suplemento, Figura C2 (revision de imprenta 2026-10-11): griegas reales, sin titulos
        # internos, series en gris (la paleta de escenarios queda reservada), texto de 7-7.5 pt.
        fig, axes = plt.subplots(1, 2, figsize=(7.00, max(3.0, 0.20 * len(robust_plot) + 1.1)), sharey=True, constrained_layout=True)
        for axis, metric, baseline, xlabel, letra in [
            (axes[0], "delta_weighted_gini_bp", 0.0, "Change in weighted Gini vs S2 (basis points)", "(a)"),
            (axes[1], "rank_stability", 1.0, "Spearman rank correlation with S2", "(b)"),
        ]:
            values = pd.to_numeric(robust_plot[metric], errors="coerce").to_numpy(dtype=float)
            axis.hlines(y_pos, baseline, values, color=NEUTRAL_LIGHT, linewidth=1.6)
            axis.scatter(values, y_pos, color=NEUTRAL_DARK, s=18, zorder=3)
            axis.axvline(baseline, linestyle=(0, (4, 2)), color=NEUTRAL_MID, linewidth=0.9)
            configure_line_axis(axis, "", xlabel, "")
            axis.set_title("")
            panel_label(axis, letra, y=1.01)
            axis.set_yticks(y_pos)
            axis.set_yticklabels(robust_plot["display_label"], fontsize=7.0)
            axis.tick_params(axis="y", pad=3, labelleft=axis is axes[0])
            axis.tick_params(axis="x", labelsize=7.0)
            axis.xaxis.label.set_fontsize(7.5)
            if metric == "rank_stability":
                finite = values[np.isfinite(values)]
                lo = float(np.floor(finite.min() * 1e5) / 1e5) if len(finite) else 0.99997
                axis.set_xticks([t for t in np.arange(lo, 1.0 + 1e-9, 1e-5) if t <= 1.0 + 1e-9])
                axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.5f}"))
                axis.tick_params(axis="x", labelrotation=30)
                for label in axis.get_xticklabels():
                    label.set_ha("right")
        save_figure(fig, paths.figures_dir, "fig_robustness")
        plt.close(fig)


def save_scenario_outputs(paths: ProjectPaths, scenario_outputs: dict[str, pd.DataFrame]) -> None:
    for scenario_name, df_scenario in scenario_outputs.items():
        df_scenario.to_csv(paths.tables_dir / f"scenario_{scenario_name}.csv", index=False)


def clean_output_artifacts(paths: ProjectPaths) -> None:
    obsolete_zip = paths.experiment_dir / "paper_v6_figures_tables.zip"
    if obsolete_zip.exists():
        obsolete_zip.unlink()
    for figure_path in paths.figures_dir.glob("*.png"):
        if figure_path.name not in PAPER_KEEP_FIGURES:
            figure_path.unlink()
    for table_path in paths.tables_dir.glob("*.csv"):
        if table_path.name not in PAPER_KEEP_TABLES:
            table_path.unlink()


def package_outputs(paths: ProjectPaths) -> None:
    clean_output_artifacts(paths)
    package_dir = paths.experiment_dir / "paper_submission_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    for relative in [
        "figures/main",
        "figures/supplementary",
        "tables/main",
        "tables/supplementary",
        "docs",
        "reproducibility",
    ]:
        (package_dir / relative).mkdir(parents=True, exist_ok=True)
    for name in PAPER_MAIN_FIGURES:
        source = paths.figures_dir / name
        if source.exists():
            shutil.copy2(source, package_dir / "figures" / "main" / name)
    for name in PAPER_SUPPLEMENTARY_FIGURES:
        source = paths.figures_dir / name
        if source.exists():
            shutil.copy2(source, package_dir / "figures" / "supplementary" / name)
    for name in PAPER_MAIN_TABLES:
        source = paths.tables_dir / name
        if source.exists():
            shutil.copy2(source, package_dir / "tables" / "main" / name)
    for name in PAPER_SUPPLEMENTARY_TABLES:
        source = paths.tables_dir / name
        if source.exists():
            shutil.copy2(source, package_dir / "tables" / "supplementary" / name)
    misc_files = [
        (paths.root / "README.md", package_dir / "reproducibility" / "README.md"),
        (paths.root / "pyproject.toml", package_dir / "reproducibility" / "pyproject.toml"),
        (paths.root / "scripts" / "stage_inputs.py", package_dir / "reproducibility" / "stage_inputs.py"),
        (paths.root / "scripts" / "reproduce_followup.py", package_dir / "reproducibility" / "reproduce_followup.py"),
        (paths.root / "src" / "social_eq" / "followup.py", package_dir / "reproducibility" / "followup.py"),
        (paths.root / "docs" / "jtg_figure_style_notes.md", package_dir / "docs" / "jtg_figure_style_notes.md"),
        (paths.root / "docs" / "paper_submission_recommendations.md", package_dir / "docs" / "paper_submission_recommendations.md"),
        (paths.input_dir / "input_manifest.json", package_dir / "docs" / "input_manifest.json"),
        (paths.input_dir / "bogota_official_ev_growth.csv", package_dir / "docs" / "bogota_official_ev_growth.csv"),
        (paths.output_dir / "run_manifest.json", package_dir / "docs" / "run_manifest.json"),
    ]
    for source, target in misc_files:
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    zip_path = paths.experiment_dir / "paper_submission_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(paths.experiment_dir))


def save_run_manifest(
    paths: ProjectPaths,
    inputs: InputFiles,
    config: ModelConfig,
    scenario_metrics: pd.DataFrame,
    robust_df: pd.DataFrame,
) -> None:
    manifest = {
        "inputs": {key: str(value) for key, value in asdict(inputs).items() if value is not None},
        "config": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in asdict(config).items()
            if key != "max_ports_matrix"
        },
        "scenario_metrics_rows": int(len(scenario_metrics)),
        "robustness_rows": int(len(robust_df)),
    }
    (paths.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def compare_with_reference(paths: ProjectPaths) -> dict[str, Any] | None:
    reference_dir = paths.reference_dir
    if reference_dir is None or not reference_dir.exists():
        return None
    comparison: dict[str, Any] = {"tables": {}}
    for table_name in [
        "table_gnn_performance_available.csv",
        "table_priority_summary.csv",
        "table_allocation_changes_S2_vs_S1.csv",
        "table_equity_metrics.csv",
        "table_equity_metrics_final.csv",
        "table_robustness_fast.csv",
    ]:
        produced = paths.tables_dir / table_name
        reference = reference_dir / "tables" / table_name
        if not produced.exists() or not reference.exists():
            continue
        produced_df = pd.read_csv(produced)
        reference_df = pd.read_csv(reference)
        info: dict[str, Any] = {
            "produced_shape": list(produced_df.shape),
            "reference_shape": list(reference_df.shape),
        }
        shared_numeric = [
            column
            for column in produced_df.columns
            if column in reference_df.columns
            and pd.api.types.is_numeric_dtype(produced_df[column])
            and pd.api.types.is_numeric_dtype(reference_df[column])
        ]
        if shared_numeric and produced_df.shape == reference_df.shape:
            aligned_produced = produced_df[shared_numeric].fillna(0.0).to_numpy(dtype=float)
            aligned_reference = reference_df[shared_numeric].fillna(0.0).to_numpy(dtype=float)
            info["max_abs_diff"] = float(np.max(np.abs(aligned_produced - aligned_reference)))
        comparison["tables"][table_name] = info
    (paths.output_dir / "reference_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def run_pipeline(
    root: Path,
    input_subdir: str = "data/inputs",
    output_subdir: str = "outputs/social_equity_followup_outputs",
    reference_subdir: str | None = "data/reference/paper_v6_experiments",
) -> ScenarioRun:
    paths = ProjectPaths.from_root(
        root=root,
        input_subdir=input_subdir,
        output_subdir=output_subdir,
        reference_subdir=reference_subdir,
    )
    paths.ensure_dirs()
    inputs = InputFiles.from_dir(paths.input_dir)
    config = ModelConfig()
    return run_refined_scenarios(paths=paths, inputs=inputs, config=config)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the Bogotá EV charging social equity follow-up outputs.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--input-subdir", default="data/inputs", help="Relative input directory.")
    parser.add_argument(
        "--output-subdir",
        default="outputs/social_equity_followup_outputs",
        help="Relative output directory.",
    )
    parser.add_argument(
        "--reference-subdir",
        default="data/reference/paper_v6_experiments",
        help="Relative reference directory. Use '' to disable.",
    )
    args = parser.parse_args(argv)
    reference_subdir = args.reference_subdir or None
    run = run_pipeline(
        root=args.root.resolve(),
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        reference_subdir=reference_subdir,
    )
    print("Outputs written to:", (args.root.resolve() / args.output_subdir))
    print(run.scenario_metrics.to_string(index=False))
    if run.reference_comparison:
        print("Reference comparison saved to:", args.root.resolve() / args.output_subdir / "reference_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
