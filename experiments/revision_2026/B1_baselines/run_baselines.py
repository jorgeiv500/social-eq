#!/usr/bin/env python3
"""Experimento B1 — baselines estáticos honestos frente al ST-GNN (R1.2).

Reconstruye baselines con features estáticas SOLAMENTE (sin las columnas
`trip_hour_*` que componen la etiqueta) y validación k-fold sobre las 142 UTAM.
Etiqueta: y = media(trip_hour_00..23), la misma pseudo-etiqueta del ST-GNN.

Salidas (en resultados/):
  - b1_diagnostico_insumos.csv    diagnóstico de etiqueta y features
  - b1_metricas_cv.csv            métricas OOF por modelo (+ por fold)
  - b1_predicciones_oof.csv       y, predicción OOF de cada modelo, priority_score
  - b1_spearman_prioridad.csv     Spearman de cada modelo contra priority_score

Ejecución (desde la raíz del repo):
  .venv/bin/python experiments/revision_2026/B1_baselines/run_baselines.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

SEMILLA = 42
K_FOLDS = 5
AQUI = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
RESULTADOS = AQUI / "resultados"
RESULTADOS.mkdir(exist_ok=True)

# Fuga de fuente: agregados del mismo conteo de viajes/etapas que compone la etiqueta.
FUGA_FUENTE = ["trips_total_w", "stages_total_w"]


def cargar_datos():
    feats = pd.read_csv(AQUI / "insumos" / "features_utam_full.csv")
    prior = pd.read_csv(
        REPO / "data" / "inputs" / "priority_utam_scores.csv",
        usecols=["UTAM", "priority_score"],
    )
    horas = [c for c in feats.columns if c.startswith("trip_hour_")]
    assert len(horas) == 24, f"se esperaban 24 columnas horarias, hay {len(horas)}"
    y = feats[horas].mean(axis=1).to_numpy()

    estaticas = [c for c in feats.columns if c not in horas and c != "utam"]
    constantes = [c for c in estaticas if feats[c].nunique() <= 1]
    informativas = [c for c in estaticas if c not in constantes]
    df = feats[["utam"] + informativas].merge(
        prior, left_on="utam", right_on="UTAM", how="inner", validate="1:1"
    )
    assert len(df) == 142, f"merge con priority_utam_scores dejó {len(df)} filas"
    return df, y, informativas, constantes


def ols_ajustar_predecir(X_tr, y_tr, X_va):
    """OLS con estandarización ajustada en el train del fold (sin fuga)."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    Xt = np.hstack([np.ones((len(X_tr), 1)), (X_tr - mu) / sd])
    Xv = np.hstack([np.ones((len(X_va), 1)), (X_va - mu) / sd])
    beta, *_ = np.linalg.lstsq(Xt, y_tr, rcond=None)
    return Xv @ beta


def evaluar_modelo(nombre, cols, tipo, df, y):
    """Devuelve (metricas_por_fold, pred_oof). tipo: 'media' | 'ols' | 'rf'."""
    X = df[cols].to_numpy(dtype=float) if cols else None
    pred_oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)
    filas = []
    for i, (tr, va) in enumerate(kf.split(np.arange(len(y)))):
        if tipo == "media":
            pred = np.full(len(va), y[tr].mean())
        elif tipo == "ols":
            pred = ols_ajustar_predecir(X[tr], y[tr], X[va])
        elif tipo == "rf":
            rf = RandomForestRegressor(
                n_estimators=500, random_state=SEMILLA, n_jobs=-1
            )
            rf.fit(X[tr], y[tr])
            pred = rf.predict(X[va])
        else:
            raise ValueError(tipo)
        pred_oof[va] = pred
        filas.append(
            {"modelo": nombre, "fold": i, "n_val": len(va),
             "mae": float(np.abs(pred - y[va]).mean())}
        )
    assert not np.isnan(pred_oof).any()
    return filas, pred_oof


def metricas_globales(nombre, y, pred, priority):
    err = pred - y
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "modelo": nombre,
        "mae_oof": float(np.abs(err).mean()),
        "rmse_oof": float(np.sqrt((err ** 2).mean())),
        "r2_oof": 1.0 - ss_res / ss_tot,
        "spearman_vs_y": float(spearmanr(pred, y).statistic),
        "spearman_vs_priority": float(spearmanr(pred, priority).statistic),
    }


def main():
    df, y, informativas, constantes = cargar_datos()
    priority = df["priority_score"].to_numpy()

    # --- diagnóstico de insumos -------------------------------------------
    diag = pd.DataFrame(
        [
            {"item": "n_utam", "valor": len(df)},
            {"item": "y_media", "valor": y.mean()},
            {"item": "y_sd", "valor": y.std()},
            {"item": "y_max", "valor": y.max()},
            {"item": "n_utam_y_cero", "valor": int((y == 0).sum())},
            {"item": "n_features_estaticas", "valor": len(informativas) + len(constantes)},
            {"item": "n_features_constantes_descartadas", "valor": len(constantes)},
            {"item": "n_features_informativas", "valor": len(informativas)},
            {"item": "spearman_y_vs_priority_score",
             "valor": float(spearmanr(y, priority).statistic)},
        ]
    )
    diag.to_csv(RESULTADOS / "b1_diagnostico_insumos.csv", index=False)

    # --- variantes de features --------------------------------------------
    var_A = informativas                                   # estática completa
    var_B = [c for c in informativas if c not in FUGA_FUENTE]  # estricta
    modelos = [
        ("media_train", None, "media"),
        ("ols_estatica_completa_A", var_A, "ols"),
        ("ols_estatica_estricta_B", var_B, "ols"),
        ("rf_estatica_completa_A", var_A, "rf"),
        ("rf_estatica_estricta_B", var_B, "rf"),
        # Diagnóstico de fuga: solo los dos agregados de volumen sospechosos.
        ("ols_solo_volumenes_fuga", FUGA_FUENTE, "ols"),
    ]

    pd.DataFrame(
        {
            "feature": informativas,
            "variante_A_completa": True,
            "variante_B_estricta": [c in var_B for c in informativas],
            "fuga_fuente": [c in FUGA_FUENTE for c in informativas],
        }
    ).to_csv(RESULTADOS / "b1_features_usadas.csv", index=False)

    por_fold, globales, preds = [], [], {"utam": df["utam"], "y": y,
                                         "priority_score": priority}
    for nombre, cols, tipo in modelos:
        filas, pred_oof = evaluar_modelo(nombre, cols, tipo, df, y)
        por_fold.extend(filas)
        globales.append(metricas_globales(nombre, y, pred_oof, priority))
        preds[f"pred_oof__{nombre}"] = pred_oof

    tabla_folds = pd.DataFrame(por_fold)
    tabla_glob = pd.DataFrame(globales)
    resumen_folds = (
        tabla_folds.groupby("modelo")["mae"].agg(["mean", "std"])
        .rename(columns={"mean": "mae_fold_media", "std": "mae_fold_sd"})
        .reset_index()
    )
    tabla_glob = tabla_glob.merge(resumen_folds, on="modelo")
    tabla_glob["k_folds"] = K_FOLDS
    tabla_glob["semilla"] = SEMILLA
    tabla_glob["n_features"] = [
        0 if cols is None else len(cols) for _, cols, _ in modelos
    ]

    tabla_glob.to_csv(RESULTADOS / "b1_metricas_cv.csv", index=False)
    tabla_folds.to_csv(RESULTADOS / "b1_metricas_por_fold.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTADOS / "b1_predicciones_oof.csv", index=False)
    tabla_glob[["modelo", "spearman_vs_priority", "spearman_vs_y"]].to_csv(
        RESULTADOS / "b1_spearman_prioridad.csv", index=False
    )

    pd.set_option("display.width", 200)
    print(tabla_glob.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
