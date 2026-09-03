# Manifiesto de insumos — Experimento B1 (baselines vs ST-GNN)

Fecha de copia: 2026-09-01. Copiados con `cp` desde la carpeta de Google Drive
sincronizada en disco; md5 verificado idéntico entre origen y copia. La corrida
que produjo estos archivos es la "enriched" del 5-abr-2026 (celda 18 de
`modelado/bogota_ev_charging_full_experiment_colab_enriched_utam.ipynb`), la misma
que generó los derivados de `data/inputs/` del repo (md5 idénticos de
`priority_utam_scores.csv` y `temporal_attention_by_utam.csv`; ver
la verificación de insumos resumida en este manifiesto).

Directorio de origen (Drive sincronizado):

```
/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes/parta_outputs/priority_outputs/
```

| Archivo | md5 | Fecha del origen | Tamaño (B) | Contenido |
|---|---|---|---|---|
| `features_utam_full.csv` | `757aa1fda7e3bf64f57c6b5ce8420fde` | 2026-04-05 16:46 | 107.737 | 142 UTAM × 85 columnas: 61 features estáticas + 24 columnas `trip_hour_00..23` |
| `baseline_metrics.csv` | `2908715066e0cdd49a849e3a226f6eb7` | 2026-04-05 16:46 | 1.810 | Métricas de los 3 baselines lineales de la corrida original (por split) |
| `stgnn_run_protocol.json` | `e48b1c8b7eb8828d6e1fc7e077d9d74c` | 2026-04-05 16:46 | 488 | Protocolo de la corrida del ST-GNN (semilla, splits, early stopping) |
| `model_metrics_split.csv` | `78f73fde8c986411abbc0b5eed6e69a7` | 2026-04-05 16:46 | 692 | Métricas del ST-GNN híbrido por split (complementario, para comparación) |

Insumo adicional usado por el experimento (ya versionado en el repo, no se copia):

- `/Users/jorgeiv500/Documents/social-eq/data/inputs/priority_utam_scores.csv`
  (md5 `2bf1301d4468b84a7f79feaa7c315f28`) — ranking de prioridad por UTAM del
  ST-GNN, referencia para la correlación de Spearman.
