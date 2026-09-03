# Experimento B1 — Baselines simples frente al ST-GNN (responde R1.2)

Revisión JTRG-D-26-00692. Todo corre en LOCAL (venv del repo, sin NEOS ni Colab).
Pregunta del Revisor 1 (R1.2): ¿se necesita realmente un ST-GNN, o modelos simples
capturan la misma señal de priorización?

## 1. Insumos

Copiados del Drive sincronizado con manifiesto md5 (`insumos/MANIFIESTO.md`).
Provienen de la corrida "enriched" del 5-abr-2026, la misma que generó los derivados
de `data/inputs/` del repo (verificado por md5 en
`insumos/MANIFIESTO.md`).

- `insumos/features_utam_full.csv` — 142 UTAM × 85 columnas: 61 features estáticas
  (EM-2023: hogares, personas, viajes, etapas) + 24 columnas `trip_hour_00..23`
  (tensor dinámico total por hora, aplanado).
- `insumos/baseline_metrics.csv` — métricas de los 3 baselines lineales que la corrida
  original ya evaluó (misma partición 70/15/15 del ST-GNN).
- `insumos/stgnn_run_protocol.json` — protocolo del ST-GNN: semilla 42, splits
  70/15/15 (99/21/22), early stopping 120, máx. 1200 épocas, k-NN k=6, n_utam=142.
- `insumos/model_metrics_split.csv` — métricas del ST-GNN híbrido por split
  (comparador directo).
- `data/inputs/priority_utam_scores.csv` (repo) — ranking de prioridad del ST-GNN.

### 1.1 Etiqueta (pseudo-etiqueta del ST-GNN)

`y = media(trip_hour_00..23)` por UTAM — demanda horaria media (proxy), idéntica a la
pseudo-etiqueta con que se entrenó el ST-GNN (verificación de insumos, secc. 1.2).
Reconstruida aquí desde las mismas columnas exportadas por la corrida.

Diagnóstico de insumos (calculado el 2026-09-01, script `run_baselines.py`, antes de
correr los baselines):

- `y`: media 19,290, sd 32,240, máx. 130,708; **98 de 142 UTAM tienen y = 0**
  (el tensor dinámico quedó en cero para esas zonas en la corrida original). La
  etiqueta es, por tanto, semidegenerada: dos tercios del problema es distinguir
  zonas con flujo tabulado de zonas sin él.
- Spearman(`y`, `priority_score`) = 0,7913 — coincide exactamente con el Spearman
  "all" del ST-GNN en `model_metrics_split.csv`, lo que confirma que
  `priority_score` es la predicción del ST-GNN sobre esta misma etiqueta.
- De las 61 features estáticas, **33 son constantes** (casi todas 0; fallos de match
  de la corrida original) y se descartan; quedan 28 informativas.

### 1.2 Evidencia ya corrida (se reporta tal cual, con su advertencia)

`insumos/baseline_metrics.csv` (partición 70/15/15, la misma del ST-GNN):

| Baseline original | Features | MAE val | R² val | Spearman val |
|---|---|---|---|---|
| `baseline_static_linear` | 61 estáticas (escaladas) | 1,298 | 0,991 | 0,998 |
| `baseline_dynamic_linear` | 24 `trip_hour_*` | 1,77e-05 | ≈1,000 | ≈1,000 |
| `baseline_combined_linear` | estáticas + `trip_hour_*` | 0,0027 | ≈1,000 | ≈1,000 |

**Advertencia de fuga de etiqueta (ya diagnosticada):** los baselines *dinámico* y
*combinado* ven las mismas 24 columnas cuya media ES la etiqueta — R²≈1 es tautológico
y no se puede citar como comparación con el ST-GNN. El baseline *estático* no contiene
esas columnas, pero sí `trips_total_w` y `stages_total_w`, agregados del mismo módulo
de viajes/etapas del que se compone la etiqueta (fuga parcial de fuente; Pearson con
`y`: 0,50 y 0,37). Además su métrica proviene de un único split con n_val = 21.
Por eso este experimento reconstruye baselines honestos con validación k-fold.

Comparador ST-GNN (`insumos/model_metrics_split.csv`; misma etiqueta):

| Split | n | MAE | R² | Spearman(pred, y) |
|---|---|---|---|---|
| train | 99 | 5,577 | 0,821 | 0,783 |
| **val** | 21 | **10,333** | 0,577 | 0,828 |
| test | 22 | 7,761 | 0,619 | 0,762 |
| all | 142 | 6,619 | 0,756 | 0,791 |

El MAE de validación 10,333 es el que cita el manuscrito
(manuscrito enviado en abril de 2026, §4.3: "best validation MAE of 10.333") y
coincide con `loss_mae_val` best = 10,3329 en
`data/reference/paper_v6_experiments/tables/table_gnn_performance_available.csv`.

## 2. Metodología de la reconstrucción honesta

Script: `run_baselines.py` (venv del repo: numpy 2.0.2, pandas 2.3.3, scipy 1.13.1,
scikit-learn 1.6.1 — instalado para este experimento).

- **Features:** solo estáticas; nunca `trip_hour_*`. Dos variantes:
  - **A — estática completa:** las 28 estáticas informativas (equivale al diseño del
    baseline original sin las columnas horarias).
  - **B — estática estricta:** variante A menos `trips_total_w` y `stages_total_w`
    (volúmenes derivados del mismo conteo de viajes/etapas que compone la etiqueta);
    26 features. Es la variante sin ninguna fuga de fuente.
- **Modelos:** (0) media del train (piso de referencia); (1) regresión lineal OLS
  (`np.linalg.lstsq`, features estandarizadas con media/sd del train de cada fold);
  (2) Random Forest (scikit-learn, 500 árboles, semilla 42) como ensamble de árboles.
  Cada uno en las variantes A y B.
- **Validación honesta:** k-fold con **k = 5**, barajado con **semilla 42**, sobre las
  142 UTAM. Toda predicción reportada es *out-of-fold* (OOF): ningún modelo vio la
  UTAM que predice. El escalado se ajusta dentro de cada fold (sin fuga de
  preprocesamiento).
- **Métricas:** MAE OOF global (y media ± sd por fold), RMSE, R² OOF,
  Spearman(pred OOF, y) y **Spearman(pred OOF, `priority_score`)** — cuánto reproduce
  cada baseline el ranking de prioridad del ST-GNN que alimenta la optimización.
- **Comparabilidad declarada:** el ST-GNN se evaluó en un único split 70/15/15
  (semilla 42); los baselines, en 5-fold OOF sobre las mismas 142 UTAM y la misma
  etiqueta. La escala del MAE es directamente comparable; el protocolo no es
  idéntico y se declara (OOF con n=142 es más exigente y estable que un val de n=21).

## 3. Hipótesis ex ante (registrada ANTES de ejecutar los baselines)

- **Rama (i) — el baseline simple alcanza al ST-GNN:** si algún baseline estático
  honesto logra MAE OOF ≤ 10,333 (el MAE de validación del ST-GNN citado en el
  manuscrito) y Spearman contra `priority_score` ≥ 0,75 (el nivel de acuerdo que el
  propio ST-GNN tiene con la etiqueta), la respuesta a R1.2 debe **conceder** que
  modelos simples capturan gran parte de la señal de priorización, y reencuadrar el
  aporte del ST-GNN: no la precisión puntual del proxy, sino la estructura espacial
  (grafo de contigüidad) y la atención temporal interpretable que alimenta el
  análisis por franjas horarias y vulnerabilidad.
- **Rama (ii) — el baseline es claramente peor:** si los baselines honestos quedan
  con MAE OOF claramente mayor (> 10,333) y/o Spearman contra `priority_score`
  claramente menor (< 0,6), la complejidad del ST-GNN queda justificada
  empíricamente y así se responde a R1.2.
- Resultados intermedios se reportan con matices (p. ej., MAE similar pero ranking
  peor, o viceversa). **Ambos desenlaces sirven; se reporta el que salga.**

Dado el diagnóstico de insumos (98/142 zonas con y = 0), se anticipa que cualquier
modelo razonable acertará las zonas nulas y el MAE quedará dominado por las 44 zonas
con flujo; la hipótesis no se ajusta por ello: los umbrales quedan fijados arriba.

## 4-bis. Resultados con los insumos CORREGIDOS (2026-09-02) — los que van al manuscrito

La sección 4 de arriba documenta la corrida del 2026-09-01 sobre los insumos de la versión
enviada, que tenían la etiqueta en cero en 98 de 142 zonas por el cruce de códigos fallido
(`../../auditoria/README.md`). Esos resultados quedan archivados en `resultados/publicado_2026-04/`
e `insumos/publicado_2026-04/` y **no se citan**. Esta sección reporta la re-corrida sobre
`insumos/features_utam_full.csv` y `insumos/model_metrics_split.csv` regenerados por
`../../auditoria/reentrenamiento/reentrenar_stgnn.py` (variante `peso_fexp_miles`: códigos
canónicos, pesos numéricos, viajes expandidos en miles). Mismo script `run_baselines.py`, k = 5,
semilla 42, n = 142; la etiqueta es ahora positiva en 140 zonas y hay 42 features estáticas
informativas (40 en la variante estricta B, que sigue excluyendo `trips_total_w` y
`stages_total_w`; la fuga por esas dos columnas sigue vigente: la OLS de solo volúmenes alcanza
R² 0.72).

| Modelo | n feat. | MAE OOF | MAE por fold | RMSE | R² | ρ vs y | ρ vs `priority_score` | ¿Citable? |
|---|---|---|---|---|---|---|---|---|
| Media del train (piso) | 0 | 5.613 | 5.60 ± 0.79 | 7.12 | -0.015 | -0.131 | -0.147 | sí (referencia) |
| OLS estática completa (A) | 42 | 2.959 | 2.96 ± 0.34 | 3.93 | 0.690 | 0.826 | 0.819 | **NO — fuga** |
| **OLS estática estricta (B)** | 40 | 3.156 | 3.16 ± 0.45 | 4.12 | 0.660 | 0.814 | 0.837 | sí |
| RF estática completa (A) | 42 | 2.756 | 2.75 ± 0.59 | 4.22 | 0.643 | 0.856 | 0.860 | **NO — fuga** |
| **RF estática estricta (B)** | 40 | 2.774 | 2.76 ± 0.83 | 4.19 | 0.649 | 0.863 | 0.856 | sí |
| OLS solo volúmenes (diagnóstico) | 2 | 2.472 | 2.47 ± 0.47 | 3.74 | 0.719 | 0.869 | 0.844 | solo diagnóstico |

Comparador ST-GNN reentrenado (`insumos/model_metrics_split.csv`, mismo protocolo 70/15/15):
val MAE 2.755 (R² 0.528, ρ 0.784); test MAE 1.998
(R² 0.738, ρ 0.862); all MAE 2.120 (R² 0.788, ρ 0.904). Unidades:
miles de viajes expandidos por hora (media de origen + destino).

**Lectura (rama i, igual que antes):** el bosque aleatorio estático estricto (MAE OOF
2.77) queda al nivel del ST-GNN en validación (2.75) y reproduce su ranking de
prioridad con ρ = 0.856; la OLS estricta es algo peor (MAE 3.16). La respuesta a R1.2
se mantiene: la precisión puntual no justifica por sí sola el ST-GNN; lo que la etapa
espacio-temporal aporta se mide en B2 (ablación de ω) y en B4 (contrafactual de prioridades
sobre la asignación), ambos re-corridos sobre los insumos corregidos.

## 4. Resultados (corrida del 2026-09-01 sobre insumos de la versión enviada — ARCHIVADA, no citar)

Corrida del 2026-09-01 (`run_baselines.py`; k = 5, semilla 42; salidas en
`resultados/`: `b1_metricas_cv.csv`, `b1_metricas_por_fold.csv`,
`b1_predicciones_oof.csv`, `b1_spearman_prioridad.csv`, `b1_features_usadas.csv`,
`b1_diagnostico_insumos.csv`).

### 4.1 Auditoría de la variante A — FUGA CONFIRMADA, NO CITABLE

La variante A (estática completa) difiere de la estricta B en exactamente **dos
features**: `trips_total_w` y `stages_total_w` (ver `b1_features_usadas.csv`).
Auditoría numérica (calculada sobre `insumos/features_utam_full.csv`):

- En las 44 UTAM con y > 0, **Pearson(`trips_total_w`, y) = 0,9986** y una OLS de
  solo volúmenes ajusta R² = 0,997 con coeficiente 0,0842 ≈ 2/24 = 0,0833 sobre
  `trips_total_w` (y coeficiente ~0 para `stages_total_w`). Es decir:
  **y ≈ 2·`trips_total_w`/24 donde la etiqueta no es cero** — identidad aritmética,
  no correlación: la etiqueta es la media horaria del flujo origen+destino y
  `trips_total_w` agrega esos mismos registros de viaje ponderados.
- Globalmente la correlación baja a 0,50 solo porque 98 zonas tienen y = 0 por
  fallos de match de la corrida original; las otras 26 features le indican al
  modelo cuáles son esas zonas y `trips_total_w` resuelve el resto. Así se explica
  el MAE OOF de 0,65 (OLS A) y también el MAE val 1,30 del `baseline_static_linear`
  original, que incluía estas columnas.
- El diagnóstico k-fold `ols_solo_volumenes_fuga` (2 features, MAE 25,15, R² 0,25)
  confirma que los volúmenes solos no bastan sin las features que señalan los ceros.

**Veredicto de auditoría: `ols_estatica_completa_A` y `rf_estatica_completa_A`
tienen fuga de etiqueta vía `trips_total_w` y NO SON CITABLES como comparación con
el ST-GNN — ni aquí ni en ningún borrador de respuesta o del manuscrito.** Lo mismo
aplica a los tres baselines de `insumos/baseline_metrics.csv` (el estático original
incluía esas columnas). Solo la variante estricta B es evidencia válida.

### 4.2 Métricas (OOF, k = 5, semilla 42, n = 142)

Fuente: `resultados/b1_metricas_cv.csv`.

| Modelo | n feat. | MAE OOF | MAE por fold (media ± sd) | RMSE | R² | Spearman vs y | Spearman vs `priority_score` | ¿Citable? |
|---|---|---|---|---|---|---|---|---|
| Media del train (piso) | 0 | 26,648 | 26,65 ± 0,75 | 32,15 | −0,001 | −0,04 | −0,03 | sí (referencia) |
| OLS estática completa (A) | 28 | 0,652 | 0,65 ± 0,20 | 1,54 | 0,998 | 0,824 | 0,643 | **NO — fuga** |
| **OLS estática estricta (B)** | 26 | **12,568** | 12,57 ± 2,26 | 31,14 | 0,060 | 0,700 | **0,586** | sí |
| RF estática completa (A) | 28 | 2,179 | 2,18 ± 1,03 | 5,93 | 0,966 | 0,997 | 0,792 | **NO — fuga** |
| **RF estática estricta (B)** | 26 | **6,772** | 6,76 ± 2,29 | 15,78 | 0,759 | 0,951 | **0,789** | sí |
| OLS solo volúmenes (diagnóstico) | 2 | 25,147 | 25,14 ± 0,63 | 27,78 | 0,252 | 0,252 | 0,136 | solo diagnóstico |

### 4.3 Comparabilidad con el MAE 10,333 del ST-GNN

- **Misma etiqueta y unidades:** verificado — y se reconstruye de las mismas
  columnas exportadas por la corrida del ST-GNN, y Spearman(y, `priority_score`)
  = 0,7913 coincide con la fila "all" de `model_metrics_split.csv`.
- **Esquema de validación distinto, se declara:** el ST-GNN se evaluó en un único
  split 70/15/15 (val n = 21, test n = 22); los baselines en 5-fold OOF (n = 142,
  todo fuera de muestra). La comparación fuera de muestra honesta es:
  RF-B 6,77 (OOF) frente a ST-GNN val 10,33 y test 7,76. El MAE "all" del ST-GNN
  (6,62) incluye las 99 zonas de train y es optimista; no se usa como vara.
  Con n = 21–22 las cifras del ST-GNN son ruidosas: **no se afirma que el RF sea
  mejor; sí que no es peor de forma discernible.**

### 4.4 Veredicto de la hipótesis ex ante

Se cumple la **rama (i), con un matiz importante**:

- El **RF estático estricto** cumple ambos umbrales fijados ex ante:
  MAE OOF 6,77 ≤ 10,333 y Spearman contra `priority_score` 0,789 ≥ 0,75
  (de hecho ≈ 0,791, el acuerdo de la propia etiqueta con el ranking del ST-GNN:
  el RF reproduce el ranking de prioridad casi al techo efectivo).
- La **OLS estricta** cae en la rama (ii): MAE 12,57 > 10,333 y Spearman
  0,586 < 0,6 — la relación no es lineal; la parte lineal honesta NO alcanza.

**Implicación para R1.2:** un modelo no lineal simple (ensamble de árboles) con 26
features socioeconómicas estáticas y validación honesta captura gran parte de la
señal de priorización del proxy, sin grafo ni componente temporal. La respuesta
debe conceder este punto y reencuadrar el aporte del ST-GNN donde el baseline no
llega: la estructura espacial explícita y la **atención temporal interpretable**,
que alimenta el análisis por franjas horarias/vulnerabilidad y que la ablación B2
cuantifica (quitar la ω aprendida cuesta hasta 22,2 puntos de cobertura vulnerable
a 15 min: 88,9% → 66,7% en S2; ver
`experiments/revision_2026/B2_ablacion_omega/salidas/tablas/resumen_ablacion_omega.csv`).
