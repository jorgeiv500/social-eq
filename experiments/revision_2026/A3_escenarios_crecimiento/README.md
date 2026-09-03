# A3 — Escenarios de crecimiento 2028 / 2030 (cuotas escaladas)

**Responde:** R2.2 (escenarios futuros de crecimiento VE) y refuerza R2.1: con cuotas mayores
el espacio de reasignación crece y la redistribución deja de ser marginal.
**Estado: COMPLETADO (2026-09-01)** — 6 corridas NEOS (S0/S1/S2 × 2028/2030), las seis
`optimal`, con los supuestos S1–S4 de abajo tal como están declarados; resultados en
`salidas/tablas/resumen_crecimiento.csv` e integrados en la Tabla 8 y §5.6 del manuscrito
(ver `../../TABLERO.md`).

## Derivación de los multiplicadores (aritmética desde el CSV, sin cifras nuevas)

Fuente única: `data/inputs/bogota_official_ev_growth.csv` (trayectoria oficial verificada,
Figura 1 del paper):

| Año | Corte | VE registrados |
|---|---|---|
| 2020 | 2020-11-30 | 2,039 |
| 2024 | 2024-11-30 | 8,899 |

- Crecimiento medio anual (el "+1,715/año" del plan):
  `(8,899 − 2,039) / (2024 − 2020) = 6,860 / 4 = 1,715 VE/año`.
- Proyección lineal de flota: `VE(h) = 8,899 + (h − 2024) × 1,715`
  - `VE(2028) = 8,899 + 4 × 1,715 = 15,759`
  - `VE(2030) = 8,899 + 6 × 1,715 = 19,189`
- Multiplicador de cuotas: `m(h) = VE(h) / VE(2024)`
  - `m(2028) = 15,759 / 8,899 ≈ 1.770873`
  - `m(2030) = 19,189 / 8,899 ≈ 2.156310`
- Cuotas escaladas `Q_v(h) = round(Q_v × m(h))`, con Q_v del paper (7,352 / 2,739 / 779,
  `ModelConfig.quotas` en `src/social_eq/followup.py`):

| Horizonte | residential | workplace | public | Total |
|---|---|---|---|---|
| 2024 (paper) | 7,352 | 2,739 | 779 | 10,870 |
| 2028 | 13,019 | 4,850 | 1,380 | 19,249 |
| 2030 | 15,853 | 5,906 | 1,680 | 23,439 |

El runner **recalcula todo esto desde el CSV en tiempo de ejecución** (no lleva cifras
codificadas) y guarda la derivación en `salidas/tablas/derivacion_cuotas.csv`; la tabla de
arriba es la misma aritmética hecha una vez para el lector.

## Supuestos de escenario (declarados, no hechos del paper)

- **S1 — Extrapolación lineal:** la flota crece a la media oficial 2020–2024 (+1,715 VE/año).
  Es un supuesto conservador; la serie real 2020–2024 se acelera, así que una alternativa de
  crecimiento compuesto daría flotas mayores. Queda como extensión opcional, no corrida.
- **S2 — Proporcionalidad puertos↔flota:** las cuotas EVI-Pro del paper se tratan como el
  nivel de infraestructura asociado a la flota de 2024 (8,899 VE, último corte del CSV) y se
  escalan proporcionalmente con la flota proyectada. El paper no afirma esta correspondencia;
  es el supuesto que convierte la trayectoria en cuotas.
- **S3 — Mezcla por sede constante:** las proporciones residential/workplace/public se
  mantienen (no hay dato oficial de cómo evolucionará la mezcla).
- **S4 — Redondeo:** al entero más cercano por sede (`round`); el total resultante puede
  diferir en ±1–2 puertos del producto exacto. Alternativas (piso/techo, ajustar el total)
  disponibles si el autor lo prefiere.

## Corridas que lanza

Por horizonte hay que re-resolver **S0** (u\* depende de las cuotas porque
`make_targets` escala los objetivos con Q_v — el u\* archivado del paper NO sirve aquí),
y luego S1 y S2 con sus ε del paper (0.03 y 0.08):

| # | Corrida | Cuotas | Especificación |
|---|---|---|---|
| 1 | `S0_2028` | Q(2028) | fidelidad pura (deriva u\*₂₀₂₈) |
| 2 | `S1_2028` | Q(2028) | `solve_equity`, ν = ninguno, ε = 0.03 |
| 3 | `S2_2028` | Q(2028) | `solve_equity_refined`, ν, κ = 4, ε = 0.08 |
| 4 | `S0_2030` | Q(2030) | fidelidad pura (deriva u\*₂₀₃₀) |
| 5 | `S1_2030` | Q(2030) | ídem #2 |
| 6 | `S2_2030` | Q(2030) | ídem #3 |

Total: **6 corridas NEOS**. Parámetros de solver: los de la especificación principal
(3600 s, mipgap 0.01). Tiempos: misma escala de modelo que el paper (142 UTAM, 10,011
pares); ver la nota de estimación en `../A1_barrido_epsilon/README.md`. Presupuesto
prudente: ≤ 6 h en el peor caso teórico, minutos por corrida en condiciones normales.

Extensión opcional (NO planificada, decidir con el autor): cruzar A1×A3 (p. ej. ε = 0.20 con
cuotas 2030) añadiría corridas adicionales por celda del cruce.

## Comando exacto

```bash
cd /Users/jorgeiv500/Documents/social-eq
export NEOS_EMAIL="correo_del_autor@dominio"

# Corrida en seco: imprime la derivación de cuotas y el plan, sin NEOS
.venv/bin/python experiments/revision_2026/A3_escenarios_crecimiento/run_a3.py

# Lanzamiento real (6 corridas NEOS)
.venv/bin/python experiments/revision_2026/A3_escenarios_crecimiento/run_a3.py --lanzar

# Un solo horizonte (3 corridas)
.venv/bin/python experiments/revision_2026/A3_escenarios_crecimiento/run_a3.py --lanzar --horizontes 2028
```

## Salidas (en `salidas/tablas/`)

- `derivacion_cuotas.csv` — flota proyectada, multiplicador y cuotas por sede por horizonte
  (recalculado desde el CSV oficial).
- `asignacion_<esc>_<año>.csv` — asignación por UTAM de cada corrida (S0/S1/S2 × 2028/2030).
- `resumen_crecimiento.csv` — por horizonte y escenario: cuotas usadas, u\* del horizonte,
  `objective_abs_dev`, pérdida de fidelidad (%), Gini ponderado/no ponderado a 6 decimales,
  cobertura 15 min de UTAM vulnerables, zonas cambiadas y puertos movidos S2 vs S1 **del
  mismo horizonte** (la cifra que responde a R2.1) y, como contexto, la estabilidad de rangos
  (Spearman de `ports_total`) frente al S2 del paper — contra el paper no se cuentan "puertos
  movidos" porque las cuotas totales difieren y la resta dejaría de medir reasignación.
  Escritura incremental tras cada corrida.
- `manifiesto_<esc>_<año>.json` — precisión completa, terminación y parámetros por corrida.
