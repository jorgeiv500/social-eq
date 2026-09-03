# C1 — Variantes de piso vinculante (responde R1.5)

**Punto del revisor:** el S3 del paper (γ=0.95 sobre el percentil 10 de la accesibilidad S2
del conjunto protegido) no activa ninguna restricción sustantiva — 95.45 % de las zonas ya
superan el piso y S3 coincide con S2 al 100 %. R1.5 concluye que γ=0.95 es una calibración
conservadora y que S3 no debería presentarse como escenario central.

**Diseño (sin umbrales inventados):** mismo mecanismo de piso blando del S3 oficial
(`solve_equity_refined`, penalización de déficit 100, followup.py líneas 1019–1034), con el
piso derivado de la propia distribución archivada de accesibilidad S2 del conjunto protegido:

| Variante | Piso | Por construcción quedan bajo el piso |
|---|---|---|
| C1a | mediana del conjunto protegido (q=0.50, γ=1.0) | ~50 % de las protegidas |
| C1b | percentil 75 del conjunto protegido (q=0.75, γ=1.0) | ~75 % de las protegidas |

`floor_reference` = columna `accessibility` de `scenario_S2.csv` archivado (la misma a2 que
usa el S3 oficial); conjunto protegido con el criterio exacto de la línea 961 de followup.py.
El diagnóstico previo (`salidas/tablas/diagnostico_pisos.csv`) reproduce además el piso
oficial (γ=0.95, q=0.10) para verificar contra el 4.55 % bajo piso que reporta el manuscrito.

## Hipótesis ex ante (registrada antes de lanzar, 2026-09-01)

Cuando el piso muerde, el solver debe pagar algo por levantar las zonas protegidas hacia el
piso: pérdida de fidelidad (mayor `objective_abs_dev` dentro del presupuesto ε=0.08),
reasignación mayor que los 5 puertos de S2, y posiblemente déficits residuales d_k > 0 si la
penalización 100 no alcanza dentro del espacio factible. Resultados que informan la respuesta:

- Si el piso se satisface con poco movimiento → el mecanismo S3 es útil como instrumento de
  política y solo estaba mal calibrado (respalda mover S3 a robustez con γ documentado).
- Si quedan déficits grandes → el piso blando es diagnóstico, no garantía, dentro de ε=0.08:
  se reporta tal cual y refuerza la recomendación de R1.5 (S3 fuera del cuadro central).

Ambos resultados son publicables; el que salga se reporta sin maquillaje.

## Corridas

2 corridas NEOS (C1a, C1b), especificación S2 idéntica en todo lo demás (ν=vulnerabilidad,
κ=4, ε=0.08, cuotas del paper, sin topes). `python run_c1.py` = corrida en seco;
`python run_c1.py --lanzar` = envía a NEOS. Salidas en `salidas/tablas/` con manifiestos
JSON y escritura incremental.
