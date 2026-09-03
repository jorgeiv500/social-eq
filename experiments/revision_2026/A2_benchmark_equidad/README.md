# A2 — Benchmark de equidad directa (sin acople a prioridad)

**Responde:** R1.6 — el revisor pide comparar con un "enfoque directo" que maximice la equidad
de accesibilidad sujeto solo a cuotas, para situar cuánta equidad se sacrifica por mantener la
fidelidad a la prioridad del ST-GNN.
**Estado: COMPLETADO (2026-09-01) — formulación cerrada, 2 corridas NEOS lanzadas y
terminadas `optimal`. Resultados abajo; borrador de respuesta en
`../../propuestas/r1/A2_respuesta_R1_6.md`.**

## Formulación final (linealización L1 por pares)

Modelo nuevo y pequeño sobre la misma infraestructura Pyomo + NEOS/CPLEX del paper.
Se **elimina** todo el acople a prioridad: no hay objetivo/restricción lexicográfica
(no hay `target`, ni `u`, ni u\*, ni ε).

```
min   Σ_{p=(i,j)∈P} w_p · z_p                          (diferencias absolutas por pares)
s.a.  Σ_i n_iv = Q_v                     ∀ v           (solo cuotas)
      A_i = Σ_v Σ_j ω_iv · e^{-t_ij/λ_v} · n_jv  ∀ i   (misma accesibilidad del paper)
      z_p ≥ A_i − A_j,  z_p ≥ A_j − A_i  ∀ p=(i,j)     (linealización del valor absoluto)
      n_iv ∈ Z⁺,  A_i ≥ 0,  z_p ≥ 0
```

Se corren **dos variantes** (2 corridas NEOS, el máximo presupuestado):

| Variante | Pesos de par w_p | Comparable con |
|---|---|---|
| `no_ponderado` | w_p = 1 | Gini **no ponderado** publicado y objetivo de S1 (`solve_equity` con `nu=None`) |
| `ponderado_k4` | w_p = 1 + κ(ν_i+ν_j)/2, κ = `s2_kappa` = 4, ν = `vulnerability_index` | Gini **ponderado** publicado y objetivo de S2 (`solve_equity_refined`, misma fórmula de pesos, líneas 714–719 de `src/social_eq/followup.py`) |

Cada variante es exactamente el término de equidad que S1/S2 minimizan, sin el ancla
lexicográfica: eso hace la comparación "cuánta equidad cuesta el acople" directa sobre el
objetivo que el paper realmente optimiza. Nota de honestidad métrica: el Gini normaliza esa
suma por la media de accesibilidad (que también depende de la asignación), así que el modelo
minimiza el numerador del Gini, no el Gini exacto; ambas cifras de Gini se reportan ex post
con las mismas funciones del paper (`gini`, `weighted_gini`). Escala: 142 UTAM → 10,011
pares, idéntica a S1/S2 (que ya corrieron por esta vía, ver `../A1_barrido_epsilon/`).

## Decisiones de formulación CERRADAS (2026-09-01)

1. **Objetivo: linealización L1 por pares** (min Σ w_p|A_i−A_j| sujeta solo a cuotas, sin
   restricción de fidelidad). **Razón:** la ruta Pyomo→NEOS de este repo solo se ha
   ejercitado con LP/MILP; el QP directo (min Σ(A_i−Ā)²) que menciona R1.6 exigiría validar
   el envío MIQP y queda **descartado y declarado** como equivalente conceptual no corrido.
   Además la L1 por pares es el mismo término de equidad del objetivo publicado, lo que da
   comparabilidad 1:1.
2. **Variante per cápita: NO se corre.** **Razón (dato, verificado 2026-09-01 y
   re-verificado antes de lanzar):** 22 de las 142 UTAM no tienen `dane_population`
   utilizable en `data/inputs/utam_social_vulnerability.csv` (21 vacías del borde rural
   5xx–7xx y UTAM590 en 0). No se imputa población — imputar fabricaría el denominador del
   objetivo. Queda como **limitación declarada**. En su lugar se corren las dos variantes de
   arriba, ambas con datos completos.
3. **Topes: los mismos `caps = config.max_ports_matrix` del pipeline** (hoy `None` →
   sin topes, `ModelConfig` línea 144 de `src/social_eq/followup.py`). **Razón:** sin dato
   que justifique topes nuevos, se replica exactamente la restricción del modelo publicado
   (la corrida oficial de S2/S3 pasa ese mismo `caps`, líneas 1007 y 1033). Si el óptimo
   concentra puertos de forma extrema, ese extremo ES parte del mensaje del benchmark y se
   reporta tal cual.
4. **Enteros: n_iv ∈ Z⁺ se mantiene** (mismo dominio que el modelo publicado). **Razón:**
   una relajación LP daría un benchmark "ideal" no alcanzable y rompería la comparación con
   las asignaciones enteras archivadas de S1/S2.

## Hipótesis ex ante (escrita ANTES de lanzar, 2026-09-01)

Evidencia previa del repo: todo el camino S0 → S2 → ε=0.30 movió el Gini no ponderado apenas
de 0.201468 (S0) a 0.201111 (S2) y 0.200670 (ε=0.30), y el ponderado de 0.197249 a 0.196896 y
0.196477, pagando +8 % y +30 % de fidelidad respectivamente
(`outputs/social_equity_followup_outputs/paper_v6_experiments/tables/table_equity_metrics_final.csv`;
`../A1_barrido_epsilon/salidas/tablas/manifiesto_eps_0.300.json`).

**Esperamos:** (i) el benchmark directo logra un Gini algo menor que S2 pero del mismo orden
(mejora en el tercer decimal, no un colapso de la desigualdad — la geografía de tiempos de
viaje acota cuánto puede igualarse A_i con cuotas fijas); (ii) a cambio de una pérdida de
fidelidad ex post muy superior al 8 % de S2 (decenas o cientos de %) y una caída clara del
Spearman puertos–prioridad (S2: 0.910472).

**Qué refutaría qué:** si el Gini del benchmark ≈ Gini de S2 (diferencia comparable o menor
que la mejora S0→S2, ~0.0004), el acople de prioridad **no cuesta equidad** y S2 queda
vindicado. Si, al contrario, el benchmark logra una mejora de Gini mucho mayor (≥ un orden de
magnitud sobre la mejora S0→S2) con pérdida de fidelidad moderada, la hipótesis (i)–(ii) queda
refutada y debilitaría el argumento del paper de que el enfoque lexicográfico equilibra bien;
se reportaría tal cual. Los resultados se publican como salgan, favorables o no.

## Datos del repo que usa cada término

| Término | Dato | Ruta |
|---|---|---|
| Cuotas Q_v = (7,352; 2,739; 779) | `ModelConfig.quotas` | `src/social_eq/followup.py` (líneas 107–113) |
| Pesos de sede ω_iv (atención temporal agregada a residential/workplace/public) | `build_venue_weights` sobre el CSV de atención | `data/inputs/temporal_attention_by_utam.csv` |
| Tiempos t_ij y decaimiento e^{-t_ij/λ_v} (λ = 15/25/10) | `build_access_coeffs` + `ModelConfig.lambda_v` | `data/inputs/utam_travel_time_matrix.csv` (completada por cierre de grafo si la cobertura < 95 %, como en el paper) |
| ν_i (pesos de par de `ponderado_k4`; también ex post) | `vulnerability_index` (normalizado 0–1) | `data/inputs/utam_social_vulnerability.csv` vía `load_minimal_inputs` |
| Prioridad (solo ex post: Spearman y fidelidad) | `priority_norm` | `data/inputs/priority_utam_scores.csv` |
| u\* = 62.8284379878768 (referencia de fidelidad) | fila S0, col. `objective_abs_dev` | `outputs/social_equity_followup_outputs/paper_v6_experiments/tables/table_priority_summary.csv` |
| Asignaciones S1/S2 de comparación | `scenario_S1.csv`, `scenario_S2.csv` | `outputs/.../paper_v6_experiments/tables/` |

## Métrica del trade-off (ex post, sin costo NEOS)

Con la asignación óptima n\* del benchmark se evalúa Σ|n\*_iv − target_iv| usando el mismo
`make_targets` del paper y se compara con u\* (el modelo NO restringe la fidelidad; esta es LA
cifra que pide R1.6). En el resumen aparece como `objective_abs_dev` y
`perdida_fidelidad_vs_S0_pct`. Da la frase para la carta de respuesta: "la equidad máxima
alcanzable con estas cuotas cuesta X % de fidelidad, frente al 8 % que concede S2".

## Comando

```bash
cd /Users/jorgeiv500/Documents/social-eq

# 1) Corrida en seco (sin NEOS): valida insumos e imprime el plan
.venv/bin/python experiments/revision_2026/A2_benchmark_equidad/run_a2.py

# 2) Lanzamiento real (2 corridas NEOS: no_ponderado y ponderado_k4)
.venv/bin/python experiments/revision_2026/A2_benchmark_equidad/run_a2.py --lanzar
```

Total: **2 corridas NEOS** (el máximo presupuestado para A2). Parámetros de solver: los de la
especificación principal (3600 s, mipgap 0.01, `ModelConfig`). Correo NEOS: el default del
config (`ModelConfig.neos_email`), que `solve_model` exporta si `NEOS_EMAIL` no está definida.
Evidencia de tiempos: las 3 corridas de A1 (misma escala) terminaron hoy en ~1–3 min cada una
(marcas de tiempo en `../A1_barrido_epsilon/salidas/tablas/`); el benchmark sin ancla
lexicográfica puede tardar más, con techo de 1 h de solver + cola NEOS.

## Salidas (en `salidas/tablas/`, escritura incremental)

- `asignacion_benchmark_<variante>.csv` — asignación por UTAM (puertos por sede y total,
  accesibilidad, vulnerabilidad), una por corrida.
- `resumen_benchmark.csv` — filas de contexto S1/S2 archivadas + una fila por corrida con:
  Gini ponderado y no ponderado a 6 decimales, cobertura 15 min de UTAM vulnerables,
  fidelidad ex post (`objective_abs_dev`) y su % sobre u\*, Spearman puertos–prioridad,
  zonas cambiadas y puertos movidos vs S1 y vs S2.
- `manifiesto_benchmark_<variante>.json` — parámetros completos, condición de terminación y
  métricas a precisión completa, uno por corrida.

## Resultados (2026-09-01; 2 corridas NEOS, ambas `optimal`)

Cifras de `salidas/tablas/resumen_benchmark.csv` y los manifiestos JSON (precisión completa);
filas S1/S2 desde las tablas archivadas del paper. Gini a 6 decimales; mipgap 0.01.

| Corrida | Σ\|n−target\| | vs u\* | Gini | Gini pond. | Cobertura 15' | ρ prioridad–puertos | Zonas ≠ vs S2 | Puertos movidos vs S2 |
|---|---|---|---|---|---|---|---|---|
| S1 (archivo) | 64.713291 | +3.0 % | 0.201248 | 0.197041 | 86.11 % | 0.916699 | 10 | 5 |
| S2 (archivo) | 67.854713 | +8.0 % | 0.201111 | 0.196896 | 88.89 % | 0.910472 | 0 | 0 |
| `benchmark_no_ponderado` | 17,324.403602 | **+27,474.1 %** | **0.055705** | 0.055686 | 91.67 % | **0.207787** | 104 | 7,222 |
| `benchmark_ponderado_k4` | 17,575.644178 | **+27,874.0 %** | 0.055360 | **0.053920** | 94.44 % | 0.210974 | 104 | 7,417 |

Notas: 7,222–7,417 puertos movidos equivalen al 66–68 % de los 10,870 del sistema; 104 zonas
son el 73 % de las 142 UTAM. Que el Gini no ponderado de `ponderado_k4` (0.055360) quede
levemente por debajo del de `no_ponderado` (0.055705) cabe dentro del mipgap 0.01 y de que el
modelo minimiza el numerador del Gini, no el Gini exacto (nota de honestidad métrica arriba).

### Veredicto de la hipótesis ex ante

- **(i) REFUTADA.** Esperábamos mejora de Gini "en el tercer decimal"; el benchmark reduce el
  Gini no ponderado de 0.201111 (S2) a 0.055705 (−72.3 %) y el ponderado de 0.196896 a
  0.053920 (−72.6 %): ~400 veces la mejora S0→S2 (0.000357 / 0.000353). El margen de equidad
  bajo cuotas fijas es mucho mayor de lo previsto; el criterio "si la mejora es ≥ un orden de
  magnitud sobre S0→S2" se cumplió con creces.
- **(ii) CONFIRMADA.** Capturar ese margen destruye la fidelidad a la prioridad: desviación
  ex post 276–280 × u\* (frente al +8 % que concede S2), Spearman puertos–prioridad de
  0.910472 → 0.21, y reasignación de 2/3 del sistema. La condición del criterio de refutación
  ("con pérdida de fidelidad moderada") NO se cumplió.
- La cobertura a 15 min de vulnerables también mejora bajo el benchmark (91.7 % / 94.4 % vs
  88.89 % de S2); se reporta tal cual.

### Resumen (≤150 palabras)

Se corrieron las dos variantes aprobadas del benchmark de equidad directa (L1 por pares,
solo cuotas, enteros, sin topes): `no_ponderado` (ν=1) y `ponderado_k4` (ν, κ=4, pesos de
S2). Ambas terminaron `optimal` en NEOS/CPLEX. Resultado: la equidad directa reduce el Gini
~72 % (0.201111→0.055705 no ponderado; 0.196896→0.053920 ponderado) y sube la cobertura de
vulnerables a 91.7–94.4 %, pero a costa de abandonar la prioridad del ST-GNN: la desviación
frente al target pasa de +8 % (S2) a +27,474–27,874 % de u\*, el Spearman puertos–prioridad
cae de 0.910 a 0.208–0.211 y se mueven 7,222–7,417 puertos (2/3 del sistema; 104/142 zonas).
La hipótesis (i) quedó refutada (el margen de Gini era mucho mayor que "tercer decimal") y la
(ii) confirmada (ese margen solo se captura destruyendo la fidelidad). El acople de prioridad
sí cuesta equidad, pero comprarla entera cuesta el modelo de demanda completo.
