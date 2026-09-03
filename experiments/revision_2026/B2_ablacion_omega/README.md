# B2 — Ablación de ω_iv: atención temporal aprendida vs. uniforme

**Responde:** R1.3 (la "pinza" sobre el ST-GNN: ¿qué aporta la etapa temporal al resultado
final de la optimización?). ω_iv (Eq. 10 del manuscrito, `main.tex` línea 453) es el puente
temporal→optimización y este experimento lo aísla.

## Dónde entra exactamente la atención temporal en el pipeline (verificado en código)

Cadena completa, verificada leyendo `src/social_eq/followup.py` (no se asumió nada):

1. `load_minimal_inputs` (líneas 881–884) lee `data/inputs/temporal_attention_by_utam.csv`
   y lo pasa por `build_venue_weights` (líneas 303–314): suma las columnas horarias
   `hour_h` en tres ventanas mutuamente excluyentes — residencial {22–23, 0–5},
   trabajo {8–17}, pública {18–21} — exactamente los $T_v$ de la Eq. 10.
2. `build_inputs` (líneas 448–468) normaliza esas tres columnas por zona para que sumen 1:
   esas son las ω_iv del modelo. La propia función define la salvaguarda "si la suma es 0,
   pesos uniformes 1/3" (línea 464), la misma que declara el manuscrito tras la Eq. 10.
3. ω_iv entra en la optimización por **exactamente dos** puntos, y por ninguno más:
   - **Targets** — `make_targets` (líneas 471–483): π_iv ∝ s_i·ω_iv (Eq. 11) y
     T_iv = π_iv·Q_v. Vía T, ω afecta a S0 (objetivo), y a S1/S2 (restricción lexicográfica).
   - **Accesibilidad** — `build_access_coeffs` (líneas 486–503): coeficiente
     ω_iv·exp(−t_ij/λ_v) de la Eq. 13, es decir, la A_i que S1/S2 igualan.
   El resto de la cadena S0→S1→S2 (`solve_s0`, `solve_equity`, `solve_equity_refined`,
   `summarize_scenario`) consume solo `target`, `coeffs`, vulnerabilidad y la matriz de
   tiempos; la atención no vuelve a aparecer (su otro uso, línea 2191, es solo para figuras).

## Construcción exacta de la variante "uniforme"

Sin tocar `src/`: se parte del mismo `df_inputs` del pipeline oficial, se reemplazan las tres
columnas de sede por 1/3 (la salvaguarda del propio código aplicada a todas las zonas) y se
recomputan `make_targets` y `build_access_coeffs` con las mismas funciones públicas. Con ω
uniforme, π_iv se reduce a s_i/Σ_k s_k para las tres sedes (misma distribución espacial en
todas, escalada por cuota) y A_i pierde la ponderación por sede (peso 1/3 en las tres).

Todo lo demás es idéntico a la corrida oficial (`run_refined_scenarios`, líneas 954–1016):

| Etapa | Función | Parámetros (idénticos al paper) |
|---|---|---|
| S0u | `solve_s0` | cuotas 7,352/2,739/779; deriva u*_unif |
| S1u | `solve_equity` | ν = None, ε = max(0.025, 0.03) = 0.03, u*_unif |
| S2u | `solve_equity_refined` | ν = vulnerabilidad, κ = 4.0, ε = 0.08, sin piso, u*_unif |

u* se **re-deriva** con S0u porque los targets uniformes cambian el óptimo de fidelidad;
usar el u* archivado (62.8284, de targets aprendidos) contaminaría la ablación.

## Doble evaluación de métricas (decisión declarada)

Con ω uniforme, la propia A_i cambia de definición, así que un Gini "auto-consistente" mezcla
dos efectos: la asignación distinta y la vara de medir distinta. Por eso cada asignación
uniforme se evalúa dos veces (ambas sin costo NEOS, es post-proceso):

- **métrica propia**: A_i con los coeficientes uniformes (lo que reportaría el pipeline ablacionado);
- **métrica aprendida**: A_i con los coeficientes ω aprendidos del paper — la comparación
  limpia contra las filas archivadas de `table_equity_metrics_final.csv`, que usan esa vara.

## Métricas reportadas (vs. corrida archivada del paper)

Por escenario (S0u vs S0, S1u vs S1, S2u vs S2, referencias en
`outputs/social_equity_followup_outputs/paper_v6_experiments/tables/`):
Gini ponderado y no ponderado a 6 decimales (ambas varas), cobertura ≤15 min de UTAM
vulnerables (cuartil superior), ρ de Spearman de la asignación entre variantes
(`ports_total` uniforme vs. paper), ρ de Spearman prioridad→puertos dentro de cada corrida,
zonas cambiadas (por total y por sede) y puertos movidos. Además, un diagnóstico sin NEOS:
ρ de Spearman entre los targets totales uniformes y aprendidos y el desplazamiento de target
en puertos-equivalentes (`diagnostico_omega.csv`). La "prioridad entre variantes" se
operacionaliza así (targets y ρ prioridad→puertos) porque `priority_norm` (s_i) es idéntica
por construcción en ambas variantes: lo único que cambia es ω.

## Hipótesis ex ante (escrita antes de lanzar corrida alguna)

La ω aprendida está lejos de uniforme: desviación media |ω−1/3| = 0.175, media de la sede
pública 0.164 frente a 0.333 (calculado de `data/inputs/temporal_attention_by_utam.csv` con
`build_venue_weights`; el runner lo re-deriva en `diagnostico_omega.csv`). Esperamos por tanto:

1. **Composición por sede**: cambios sustanciales — con ω uniforme las tres sedes reciben la
   misma distribución espacial, cosa que la ω aprendida rompe deliberadamente; muchas zonas
   con al menos una sede cambiada frente al paper.
2. **Totales por zona**: más estables (ρ de Spearman alta), porque s_i domina la suma por fila
   de los targets.
3. **Métricas finales de equidad** (Gini/cobertura con la métrica aprendida): cambio pequeño,
   dado que las diferencias S0→S2 del paper ya viven en el 3er–4to decimal del Gini.

Regla de decisión, fijada de antemano: si (3) se confirma — métricas finales casi iguales —
el beneficio marginal de la etapa temporal **sobre las métricas finales de equidad** es
acotado y así se reporta, con honestidad, apoyando la respuesta a R1.3 en (1): el papel de ω
está en la composición por sede de la asignación, no en el Gini. Si (3) se refuta — cambio
sustancial — la etapa temporal queda justificada por resultado final. Ninguno de los dos
desenlaces se maquilla.

## Comando exacto

```bash
cd /Users/jorgeiv500/Documents/social-eq
export NEOS_EMAIL="correo_del_autor@dominio"   # si no, cae al default de ModelConfig

# 1) Corrida en seco (sin NEOS): valida insumos, deriva el diagnóstico de ω, imprime el plan
.venv/bin/python experiments/revision_2026/B2_ablacion_omega/run_b2.py

# 2) Lanzamiento real (3 corridas NEOS: S0u, S1u, S2u)
.venv/bin/python experiments/revision_2026/B2_ablacion_omega/run_b2.py --lanzar
```

Total: **3 corridas NEOS** (≤ 6 autorizadas para B2; presupuesto conjunto B2+C1 ≤ 8).
S0u/S1u usan los parámetros de solver del config (3600 s, mipgap 0.01) igual que el pipeline
oficial; `--time-limit/--mip-gap` solo aplican a S2u (única etapa cuya firma los acepta).

## Salidas (en `salidas/tablas/`)

- `diagnostico_omega.csv` — comparación ω aprendida vs. uniforme y targets (sin NEOS).
- `asignacion_B2_<S0u|S1u|S2u>.csv` — asignación por UTAM de cada corrida uniforme.
- `resumen_ablacion_omega.csv` — filas de contexto archivadas (S0/S1/S2 del paper) + una
  fila por corrida uniforme, con las métricas de arriba; escritura incremental tras cada corrida.
- `manifiesto_B2_<corrida>.json` — parámetros, terminación, tiempos de pared, métricas a
  precisión completa (ambas varas) y comparaciones.

## Resultados (corridas del 2026-09-01; runner revisado y lanzado por la orquestación)

Las 3 corridas NEOS terminaron **óptimas** (duraciones: S0u 4.6 s, S1u 76.9 s, S2u 84.6 s de
pared). Toda cifra de esta sección sale de `salidas/tablas/resumen_ablacion_omega.csv`,
`diagnostico_omega.csv` y los `manifiesto_B2_*.json`; nada se cita de memoria.

| Métrica | Paper (ω aprendida) | Ablación (ω = 1/3) |
|---|---|---|
| u\* (óptimo de fidelidad, targets propios) | 62.828438 | 53.352176 |
| Gini no ponderado, vara aprendida (S2 vs S2u) | 0.201111 | **0.184625** |
| Gini ponderado, vara aprendida (S2 vs S2u) | 0.196896 | **0.181023** |
| Cobertura ≤15 min vulnerables (S2 vs S2u) — **comparación limpia** | 0.888889 (32/36) | **0.666667 (24/36)** |
| Cobertura a lo largo de la cadena | 0.7778 → 0.8611 → 0.8889 | 0.6667 → 0.6667 → 0.6667 (plana) |
| Acceso medio zonas alta vulnerabilidad, vara aprendida | 410.26 | 401.56 |
| ρ prioridad→puertos (S2 vs S2u) | 0.910472 | 0.888888 |
| ρ Spearman asignación total S2u vs S2 | — | 0.9225 |
| Zonas cambiadas S2u vs S2 (total = por sede) | — | 78 de 142 |
| Puertos movidos S2u vs S2 | — | 1,392 (12.8 % de 10,870) |

Diagnóstico previo (sin NEOS): ρ de targets totales 0.9926 pero desplazamiento de target de
1,714.5 puertos-equivalentes, concentrado en la sede residencial (1,542.2).

## Veredicto de la hipótesis ex ante

1. **Composición por sede — CONFIRMADA.** 78/142 zonas cambian al menos una sede frente a S2
   y se mueven 1,392 puertos (12.8 % del total). La ω aprendida decide *qué tipo* de puerto
   recibe cada zona.
2. **Totales por zona más estables — CONFIRMADA.** ρ de Spearman 0.92–0.94 entre variantes.
3. **Métricas finales casi iguales — REFUTADA.** La cobertura ≤15 min de zonas vulnerables cae
   22.2 pp (de 32/36 a 24/36 zonas), por debajo incluso del S0 del paper (28/36), y en la
   cadena uniforme la etapa de equidad **no la mueve en absoluto** (plana en 0.6667), mientras
   que con ω aprendida sube 0.7778→0.8889. La etapa temporal sí cambia el resultado final.

## Cautelas de lectura (se reportan sin maquillar)

- **Dos varas para el Gini.** Con ω uniforme cambia la *definición* de A_i (Eq. 13), así que el
  Gini auto-consistente no es comparable entre variantes; por eso ambas asignaciones se midieron
  también con la vara aprendida. Aun con esa vara común, la variante uniforme da un Gini
  **menor** (0.1846 vs 0.2011). Ese nivel ya está en S0u (0.184763): proviene del patrón de
  targets uniformes, no de la optimización de equidad (dentro de cada cadena el Gini solo se
  mueve en el 4º decimal: −0.000138 uniforme, −0.000357 paper). No se presenta como mejora: el
  escalar Gini es más bajo a costa de peor cobertura vulnerable (−22.2 pp) y menor acceso medio
  de las zonas de alta vulnerabilidad (401.6 vs 410.3, −2.1 %).
- **La cobertura sí es comparación limpia.** `coverage_within_threshold` (líneas 432–445 de
  `followup.py`) usa solo `n_sol`, la matriz de tiempos y la máscara de vulnerabilidad — es
  independiente de ω por construcción, en ambas variantes mide lo mismo.
- u\*_unif (53.35) y u\* (62.83) no son comparables como niveles de calidad: son óptimos contra
  targets distintos; se reportan como contexto.

## Resumen

Se corrió la cadena S0u→S1u→S2u con ω_iv = 1/3 (3 corridas NEOS, todas óptimas), idéntica en
todo lo demás al pipeline del paper, y se comparó con la corrida archivada. La hipótesis ex
ante se confirmó en composición (78/142 zonas con sede cambiada, 1,392 puertos movidos) y en
estabilidad de totales (ρ ≈ 0.92), pero se refutó en su parte central: las métricas finales
sí cambian. La cobertura ≤15 min de zonas vulnerables — la única métrica independiente de ω —
cae de 32/36 a 24/36 zonas y queda plana en toda la cadena uniforme. El Gini escalar resulta
menor en la ablación (0.1846 vs 0.2011, vara aprendida) y se reporta sin maquillar: viene del
patrón de targets, no de la equidad. Conclusión para R1.3: el puente ω_iv sí tiene efecto
material en el resultado final, vía composición por sede y cobertura vulnerable, no vía Gini.
