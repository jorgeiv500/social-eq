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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")
plt.style.use("default")


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
    package_outputs(paths)
    save_run_manifest(paths, inputs, config, scenario_metrics, robust_df)

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


def export_figures(
    paths: ProjectPaths,
    utam: gpd.GeoDataFrame,
    df_inputs: pd.DataFrame,
    loss_hist: pd.DataFrame | None,
    scenario_outputs: dict[str, pd.DataFrame],
    protected_mask_s3: np.ndarray,
    robust_df: pd.DataFrame,
) -> None:
    utam_map = utam[["UTAM", "geometry"]].copy()
    if utam_map.crs is None:
        utam_map = utam_map.set_crs(4326)

    gdf_priority = utam_map.merge(df_inputs[["UTAM", "priority_norm", "vulnerability_index"]], on="UTAM", how="left")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    gdf_priority.plot(column="priority_norm", legend=True, cmap="viridis", linewidth=0.2, edgecolor="white", ax=ax)
    ax.set_title("Learned zone-level priority index")
    ax.set_axis_off()
    fig.savefig(paths.figures_dir / "fig_priority_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if loss_hist is not None and len(loss_hist):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for column in loss_hist.columns:
            if str(column).lower() == "epoch":
                continue
            if pd.api.types.is_numeric_dtype(loss_hist[column]):
                ax.plot(loss_hist[column], label=column)
        ax.set_title("GNN learning history")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss / metric")
        ax.legend(loc="best", fontsize=8)
        fig.savefig(paths.figures_dir / "fig_gnn_learning_history.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    attention = pd.read_csv(paths.input_dir / "temporal_attention_by_utam.csv")
    attention = attention.rename(columns={find_utam_col(attention): "UTAM"})
    hour_cols = [f"hour_{hour}" for hour in range(24) if f"hour_{hour}" in attention.columns]
    if hour_cols:
        ax.plot(range(len(hour_cols)), attention[hour_cols].mean(axis=0).to_numpy(dtype=float), marker="o")
        ax.set_xticks(range(len(hour_cols)))
        ax.set_xticklabels([column.replace("hour_", "") for column in hour_cols], rotation=0)
        ax.set_title("Average temporal attention profile")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean attention")
    fig.savefig(paths.figures_dir / "fig_attention_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_ports = []
    for scenario_name, df_scenario in scenario_outputs.items():
        if "ports_total" in df_scenario.columns:
            all_ports.extend(pd.to_numeric(df_scenario["ports_total"], errors="coerce").dropna().tolist())
    ports_vmin = float(np.min(all_ports)) if all_ports else 0.0
    ports_vmax = float(np.max(all_ports)) if all_ports else 1.0

    for scenario_name, df_scenario in scenario_outputs.items():
        gdf = utam_map.merge(df_scenario[["UTAM", "ports_total", "accessibility"]], on="UTAM", how="left")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        gdf.plot(
            column="ports_total",
            legend=True,
            cmap="YlOrRd",
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            vmin=ports_vmin,
            vmax=ports_vmax,
        )
        ax.set_title(f"{scenario_name} total port allocation")
        ax.set_axis_off()
        fig.savefig(paths.figures_dir / f"fig_allocation_{scenario_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if "S1" in scenario_outputs and "S2" in scenario_outputs:
        delta = scenario_outputs["S2"][["UTAM", "ports_total"]].merge(
            scenario_outputs["S1"][["UTAM", "ports_total"]],
            on="UTAM",
            suffixes=("_S2", "_S1"),
        )
        delta["delta_ports"] = delta["ports_total_S2"] - delta["ports_total_S1"]
        gdf_delta = utam_map.merge(delta[["UTAM", "delta_ports"]], on="UTAM", how="left")
        dmax = np.nanmax(np.abs(gdf_delta["delta_ports"].to_numpy(dtype=float)))
        dmax = float(dmax) if np.isfinite(dmax) and dmax > 0 else 1.0
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        gdf_delta.plot(
            column="delta_ports",
            legend=True,
            cmap="coolwarm",
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            vmin=-dmax,
            vmax=dmax,
        )
        ax.set_title("Net change in total ports: S2 minus S1")
        ax.set_axis_off()
        fig.savefig(paths.figures_dir / "fig_net_change_S2_minus_S1.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if "S2" in scenario_outputs and "S3" in scenario_outputs:
        delta32 = scenario_outputs["S3"][["UTAM", "ports_total"]].merge(
            scenario_outputs["S2"][["UTAM", "ports_total"]],
            on="UTAM",
            suffixes=("_S3", "_S2"),
        )
        delta32["delta_ports"] = delta32["ports_total_S3"] - delta32["ports_total_S2"]
        gdf_delta32 = utam_map.merge(delta32[["UTAM", "delta_ports"]], on="UTAM", how="left")
        dmax32 = np.nanmax(np.abs(gdf_delta32["delta_ports"].to_numpy(dtype=float)))
        dmax32 = float(dmax32) if np.isfinite(dmax32) and dmax32 > 0 else 1.0
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        gdf_delta32.plot(
            column="delta_ports",
            legend=True,
            cmap="coolwarm",
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            vmin=-dmax32,
            vmax=dmax32,
        )
        ax.set_title("Net change in total ports: S3 minus S2")
        ax.set_axis_off()
        fig.savefig(paths.figures_dir / "fig_net_change_S3_minus_S2.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    protected_df = pd.DataFrame({"UTAM": df_inputs["UTAM"], "protected_s3": protected_mask_s3.astype(int)})
    gdf_protected = utam_map.merge(protected_df, on="UTAM", how="left")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    gdf_protected.plot(column="protected_s3", legend=True, cmap="Set1", linewidth=0.2, edgecolor="white", ax=ax)
    ax.set_title("Protected set used in S3 soft-floor scenario")
    ax.set_axis_off()
    fig.savefig(paths.figures_dir / "fig_S3_protected_set.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for scenario_name, df_scenario in scenario_outputs.items():
        x, y = lorenz_curve(df_scenario["accessibility"])
        axes[0].plot(x, y, label=scenario_name)
        xw, yw = weighted_lorenz(df_scenario["accessibility"], 1.0 + df_scenario["vulnerability_index"])
        axes[1].plot(xw, yw, label=scenario_name)
    for axis, title in zip(axes, ["Unweighted Lorenz curves", "Vulnerability-weighted Lorenz curves"]):
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Population share")
        axis.set_ylabel("Accessibility share")
        axis.legend()
    fig.savefig(paths.figures_dir / "fig_lorenz_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if len(robust_df):
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
        y_pos = np.arange(len(robust_df))
        ax.scatter(robust_df["delta_weighted_gini"], y_pos, label="Delta weighted Gini")
        ax.scatter(robust_df["rank_stability"], y_pos, label="Rank stability")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(robust_df["alternative_specification"])
        ax.axvline(0, linestyle="--")
        ax.set_title("Robustness relative to S2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "fig_robustness.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_scenario_outputs(paths: ProjectPaths, scenario_outputs: dict[str, pd.DataFrame]) -> None:
    for scenario_name, df_scenario in scenario_outputs.items():
        df_scenario.to_csv(paths.tables_dir / f"scenario_{scenario_name}.csv", index=False)


def package_outputs(paths: ProjectPaths) -> None:
    zip_path = paths.experiment_dir / "paper_v6_figures_tables.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for subdir in [paths.figures_dir, paths.tables_dir]:
            for file_path in subdir.rglob("*"):
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
