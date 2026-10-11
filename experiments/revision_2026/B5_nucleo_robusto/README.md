# B5 — Núcleo robusto y arrepentimiento cruzado entre señales de demanda

Revisión JTRG-D-26-00692, §5.5 (R1.2/R1.3). **Sin corridas NEOS nuevas.** Reutiliza las
asignaciones ya resueltas de B4 (`../B4_contrafactual_prioridad/salidas/tablas/`, señales
A = encuesta directa, B = lineal, C = bosque aleatorio) y la asignación publicada del ST-GNN
(D, leída con `comun.leer_asignacion_paper`). Los targets de cada señal se reconstruyen con
`make_targets` del pipeline oficial, con las mismas prioridades de B4
(`run_b4.prioridades_alternativas`). No se modificó `src/`.

## Qué se calcula

1. **Núcleo robusto.** Para cada zona *i* y sede *v*, `n_core = min_m n_iv^m` sobre las cuatro
   señales. Su suma son los puertos que cualquiera de los cuatro modelos asignaría ahí.
2. **Arrepentimiento cruzado.** La asignación de cada señal *m* se evalúa contra el target de
   cada señal *k*: `D_k(X_m) = Σ_iv |n_iv^m − t_iv^k|`. Se reporta como pérdida relativa a
   `u*_k` (lo pedido) y como **exceso en puertos** `(D_k − u*_k)/2`: como Σ_i n = Σ_i t por
   sede, esa cantidad son los puertos colocados fuera del ideal de *k*.
3. **Estabilidad de equidad.** Gini ponderado y no ponderado, cobertura a 15 min y provisión del
   cuartil vulnerable (puertos por 10.000 habitantes, `ports_per_capita_by_quartile`).
4. **Cota de robustez en fidelidad (local, HiGHS, no NEOS).** MIP pequeño (426 enteras):
   `min r` s.a. `D_k(X) ≤ u*_k (1 + r)` para las cuatro señales y cuotas por sede. Límite de
   240 s; se reporta la mejor solución y la cota inferior.

## Resultados (todos de `salidas/`)

**u\*.** El óptimo de S0 es separable por sede y se resuelve exactamente por redondeo. Para B, C
y D coincide con el u\* de NEOS (diferencia < 1e-6). Para A, el S0 de NEOS (110.3232) quedó
dentro del gap MIP del 1 % (el óptimo exacto es 109.4216; diferencia 0.82 %). B5 normaliza con
el óptimo exacto (`b5_u_star_verificacion.csv`). El texto del manuscrito dice que las corridas
"reached optimality"; lo correcto sería decir que son óptimas dentro de la tolerancia del 1 %.

**Núcleo robusto (S2)** — `b5_nucleo_resumen.csv`, `b5_nucleo_por_cuartil.csv`, `b5_zonas_S2.csv`
- 7.802 de 10.870 puertos (71,8 %) están en el núcleo: residencial 70,2 %, trabajo 75,1 %,
  público 74,8 %. A nivel de total por zona: 7.808 (71,8 %). En S0 y S1 la cifra es la misma
  (7.803).
- 133 de 142 zonas reciben puertos bajo las cuatro señales. Las 9 zonas que dependen del modelo
  están en los cuartiles 1 a 3; **las 36 zonas del cuartil más vulnerable reciben puertos bajo
  las cuatro señales.**
- Puertos en disputa (Σ max − min): 6.344. Rango por zona: mediana 39,5 puertos, 103 zonas con
  rango ≥ 25, máximo 196 (UTAM85, cuartil 4: entre 158 y 354).
- Cuartil vulnerable: núcleo de 2.694 puertos (7,4 por 10.000 hab.), entre el 70,8 % (D) y el
  79,4 % (B) de lo que cada modelo le asigna (3.392 a 3.805). Cuartiles 1/2/3: 15,6 / 8,1 /
  10,0 puertos por 10.000 hab. en el núcleo.

**Arrepentimiento cruzado (S2)** — `b5_matriz_exceso_puertos_S2.csv`, `b5_matriz_arrepentimiento_S2.csv`

| asignación \ ideal de | A | B | C | D |
|---|---|---|---|---|
| S2_A | 4,8 | 1.804,3 | 1.577,0 | 1.317,3 |
| S2_B | 1.789,3 | 3,9 | 1.883,5 | 1.680,6 |
| S2_C | 1.566,3 | 1.881,5 | 3,9 | 1.790,6 |
| S2_D | 1.310,4 | 1.693,5 | 1.802,0 | 4,2 |

(exceso en puertos sobre u\*_k). La etapa de equidad cuesta entre 3,9 y 4,8 puertos frente al
ideal del propio modelo; cambiar de modelo coloca entre 1.310 y 1.884 puertos (12,1 % a 17,3 %
del sistema) fuera del ideal del otro. Relativo a u\*_k la pérdida cruzada es de 2.395 % a
3.668 %: la normalización por u\* degenera porque u\* es solo el error de redondeo entero
(~100 sobre 10.870), por eso la lectura en puertos es la útil. **Minimax entre las cuatro S2:
la del ST-GNN** (peor caso 1.802,0 puertos, 16,6 %), prácticamente empatada con la encuesta
directa (1.804,3); lineal 1.883,5 y bosque 1.881,5.

**Compromiso minimax (local)** — `b5_asignacion_minimax_fidelidad_local.csv`, `b5_manifiesto.json`
- r = 17,9237 (cota inferior 17,9178; gap 0,03 %, límite de 240 s): ninguna asignación puede
  estar a menos de ~17,9·u\*_k de los cuatro ideales a la vez.
- En puertos: exceso máximo 980,6 (9,0 % del sistema) frente a ≥ 1.802 (16,6 %) de cualquier
  S2 de un solo modelo; por modelo 980,6 (A), 919,4 (B), 948,1 (C), 974,5 (D).
- Equidad (sin etapa de equidad): Gini ponderado 0,165574, cobertura 1,0, cuartil vulnerable
  10,1 puertos por 10.000 hab. (3.699 puertos).

**Estabilidad de equidad (S2)** — `b5_estabilidad_equidad.csv`. Gini ponderado 0,160881 (A),
0,166628 (B), 0,168320 (C), 0,164653 (D): coinciden con B4 a 6 decimales. No ponderado
0,165529–0,173598; cobertura 1,0 en las cuatro; cuartil vulnerable 9,29–10,42 puertos por
10.000 hab. (resto de zonas 15,29–15,93).

## S2 robusta (preparada, no lanzada)

`run_s2_robusta.py` replica la S2 publicada (κ = 4, sin piso ni topes) pero impone la
restricción de fidelidad para las cuatro señales a la vez:
`D_k(X) ≤ u*_k (1 + r*)(1 + ε)`, con r\* de `b5_manifiesto.json` y ε = 0,08. Presupuesto
resultante: ~997 a ~1.063 puertos fuera del ideal de cada modelo. Mismo tamaño que la S2
publicada más 3 × 426 desviaciones: **una** corrida NEOS. Sin `--lanzar` solo imprime el plan.

## Reproducir

```bash
.venv/bin/python experiments/revision_2026/B5_nucleo_robusto/run_b5.py           # ~5 min, local
.venv/bin/python experiments/revision_2026/B5_nucleo_robusto/run_s2_robusta.py    # plan, sin NEOS
```

`run_b5.py` escribe en `salidas/aux/` los auxiliares de `load_minimal_inputs` (shapefile
descomprimido); se pueden borrar. El MIP minimax depende del límite de tiempo: con otro
hardware la mejor solución puede variar ligeramente dentro del gap reportado.
