# A5 — Restricción de red eléctrica dentro del modelo S2 (R1.8, R2.1)

Experimento de la revisión JTRG-D-26-00692. A4 usó la holgura de subestación como
**cribado** ex post de la asignación. Aquí la misma holgura entra como **restricción del
modelo** (S0 y S2), en kW y de forma conjunta para las tres sedes, y se resuelve con
NEOS/CPLEX con los parámetros del manuscrito (`solver_time_limit = 3600 s`,
`solver_mip_gap = 0.01`).

Fecha de las corridas: 2026-10-10. Tres corridas S2 y una S0, todas `optimal`.

## Qué cambia respecto al modelo publicado (y nada más)

Para cada UTAM *i* con tope (cobertura de las zonas de servicio ≥ 50 %, regla de A4):

```
Σ_v kW_v · n[i,v]  ≤  1000 · cap_i_MVA        (restricción model.red_kw)
```

- `cap_i_MVA`: holgura de subestación repartida a la UTAM (`../A4_topes_capacidad/salidas/tablas/capacidad_utam.csv`,
  POT Decreto 555, SDP, CC BY 4.0; construcción y salvedades en el README de A4).
- 100 UTAM con tope; las 42 restantes (cobertura < 50 %) quedan sin tope, igual que en A4.
- Supuestos heredados de A4: factor de potencia ≈ 1 (1 MVA ≈ 1 MW), potencia nominal sin
  factor de simultaneidad y toda la holgura de 11.4 kV disponible para carga de VE (tope
  generoso).
- **Diferencia con A4**: A4 solo podía repartir el tope zonal entre sedes en proporción al
  target (`caps[i,v]` de `solve_equity_refined`). Aquí el tope es **conjunto** por zona y en kW,
  así que admite potencias distintas por sede y el modelo elige la mezcla.

**Implementación sin tocar `src/`.** `solve_s0` y `solve_equity_refined`
(`src/social_eq/followup.py`) arman el modelo y llaman a `solve_model` por nombre global del
módulo. Durante cada corrida, `run_a5.py` sustituye temporalmente `followup.solve_model` por
un envoltorio (`restriccion_red`) que añade la familia `model.red_kw` al modelo ya construido y
después llama al `solve_model` original. Cuotas, desviaciones absolutas, presupuesto
lexicográfico, accesibilidad, pares ponderados (ν = vulnerabilidad, κ = 4) y opciones del
solver son el código publicado sin cambios.

## Escenarios de potencia por puerto

Categorías de la clasificación AFIR (Reglamento (UE) 2023/1804), publicadas por el European
Alternative Fuels Observatory y ya citadas en A4: AC «lenta» < 7.4 kW, AC «media»
7.4–22 kW, DC «rápida» desde 50 kW.

| Escenario | Residencial | Trabajo | Público | Papel |
|---|---|---|---|---|
| AC7.4 | 7.4 | 7.4 | 7.4 | Confirmación (el tope más laxo) |
| MIXTO | 7.4 | 22 | 50 | Tecnología mixta realista: carga lenta nocturna en vivienda (AC monofásica, límite superior de la categoría lenta), AC trifásica de 22 kW en el lugar de trabajo (estancias largas de jornada) y DC de 50 kW en la red pública (estancias cortas, recarga de oportunidad) |
| DC50 | 50 | 50 | 50 | Cota de estrés: toda la flota de puertos como carga rápida DC |

## Regla para el presupuesto de fidelidad

Se eligió la opción **u\*_red**: la etapa 1 lexicográfica (S0) se resuelve **con la misma
restricción de red**, y S2 se resuelve con ε = 0.08 sobre ese u\*_red. Es la opción más
defendible porque conserva intacta la lógica del manuscrito (la fidelidad a la prioridad se
mide frente a lo mejor que se puede hacer *con la red*, y la tolerancia de equidad sigue siendo
el mismo 8 %) y no introduce ningún parámetro nuevo. La regla de A4
(`ε_relax = [u*·1.08 + 2D]·1.02/u* − 1`) añade un margen de integralidad arbitrario (+2 %) y
mezcla dos efectos: el desplazamiento forzado por la red y la relajación de equidad. Se
archiva solo como referencia en `diagnostico_red.json`.

Si el S0 archivado ya cumple la restricción de red, u\*_red = u\* archivado sin corrida nueva:
la restricción solo elimina soluciones factibles y el óptimo archivado sigue siendo factible.
Es el caso de AC7.4 y MIXTO.

## Diagnóstico determinista (sin NEOS)

`python run_a5.py` → `salidas/tablas/diagnostico_red.json`. Cota inferior de desviación con
tope conjunto: una zona cuya carga a target excede el tope en E_i kW debe perder al menos
E_i / max_v kW_v puertos bajo el target y, por conservación de masa en cada sede, esos puertos
reaparecen sobre el target en otra zona: Σu ≥ 2·D, con D = Σ_i E_i / max_v kW_v. Como
verificación cruzada, S0 con red se resuelve también en local con HiGHS (`scipy.optimize.milp`).

| Escenario | UTAM con carga a target > tope | UTAM donde S2 archivado viola | Utilización máx. de S2 archivado (UTAM) | Carga total S2 archivado | 2D | Presupuesto u\*·1.08 | S0 con red (HiGHS local) |
|---|---|---|---|---|---|---|---|
| AC7.4 | 0 | 0 | 33.2 % (UTAM48) | 80.4 MW | 0 | 117.54 | 108.8337 (= u\*) |
| MIXTO | 0 | 0 | 70.5 % (UTAM48) | 153.6 MW | 0 | 117.54 | 108.8337 (= u\*) |
| DC50 | 15 | 15 | 224.1 % (UTAM48) | 543.5 MW | 971.08 | 117.54 | 985.4478 |

Con DC50, S2 con red es infactible a ε = 0.08 sobre el u\* archivado (2D = 971.08 > 117.54),
como ya establecía A4.

## Resultados de las corridas NEOS (2026-10-10)

`python run_a5.py --lanzar --escenario {AC7.4|MIXTO|DC50}`; resumen en
`salidas/tablas/resumen_a5.csv`; detalle por corrida en `manifiesto_*.json` y asignaciones en
`asignacion_*.csv`; registros en `salidas/logs/`.

| | S2 archivado | AC7.4 | MIXTO | DC50 |
|---|---|---|---|---|
| Terminación S0 con red | — | no requerida | no requerida | `optimal` (10.5 s) |
| Terminación S2 con red | — | `optimal` (294 s) | `optimal` (254 s) | `optimal` (243 s) |
| u\* usado | 108.833700 | 108.833700 | 108.833700 | 985.447761 (+805.5 %) |
| Σu del solver (cota del presupuesto) | 117.540396 | 117.540396 | 117.540396 | 1 064.283582 |
| Desviación realizada Σ\|n−T\| (ex post)† | 117.242420 | 117.242420 | 117.242420 | 1 064.073434 |
| Gini no ponderado | 0.169942 | 0.169942 | 0.169942 | 0.161086 |
| Gini ponderado | 0.164653 | 0.164653 | 0.164653 | 0.156937 |
| Cobertura 15 min, cuartil vulnerable | 1.0 | 1.0 | 1.0 | 1.0 |
| Puertos / 10 000 hab., cuartil vulnerable | 10.42 | 10.42 | 10.42 | 9.16 |
| Puertos / 10 000 hab., resto | 15.62 | 15.62 | 15.62 | 16.81 |
| ρ prioridad–puertos | 0.925967 | 0.925967 | 0.925967 | 0.854059 |
| Zonas cambiadas (total) vs S2 | — | 0 | 0 | 100 |
| Puertos movidos vs S2 | — | 0 | 0 | 504 (res. 107, trab. 365, púb. 43) |
| Topes saturados* / violados | — | 0 / 0 | 0 / 0 | 15 / 0 |
| Saldo neto en el cuartil vulnerable | — | 0 | 0 | −458 (gana 15, pierde 473) |

† Recalculada desde la asignación con el mismo criterio que `scripts/realized_deviation.py`; para S2 archivado el manuscrito (Tabla `tab:tolerance_benchmark`) da 117.242419. La diferencia de 1·10⁻⁶ viene de construir T con `make_targets` en lugar de con la función del script.

\* «Saturado»: la holgura que queda es menor que la potencia del puerto más pequeño del
escenario (no cabe ni un puerto más).

**Confirmación (AC7.4 y MIXTO).** La solución con la red dentro del modelo es **idéntica,
puerto a puerto y sede a sede**, a la S2 archivada: 0 zonas con alguna sede distinta y
0 puertos movidos en cada una de las 142 × 3 celdas. Las métricas coinciden con las archivadas a la
precisión de `table_equity_metrics_final.csv` (Σu 117.54039615148103 frente a 117.540396151481).
Con la mezcla realista, la UTAM más cargada (UTAM48) usa el 70.5 % de su holgura.

**Cota de estrés (DC50).** La red muerde en 15 UTAM y **13 de ellas están en el cuartil
vulnerable** (Kennedy, Bosa, Suba, Ciudad Bolívar, entre otras); las otras dos son UTAM9 y
UTAM52. Esas 15 zonas pierden 487 puertos y quedan saturadas. Las mayores pérdidas son
UTAM84 (−89), UTAM48 (−78), UTAM82 (−58) y UTAM47 (−49). La capacidad se reubica en
80 zonas, de las que solo 11 son vulnerables, y 266 de los puertos ganados van a zonas
sin tope (sin dato de subestación). El resultado es que la provisión del cuartil
vulnerable cae de 10.42 a 9.16 puertos por 10 000 habitantes, mientras **el Gini de
accesibilidad baja** (0.164653 → 0.156937 ponderado). Esa bajada no es una mejora de
equidad atribuible a la red. Viene del presupuesto de desviación nueve veces mayor que
impone u\*_red: el ε equivalente sobre el u\* archivado es 8.78. En la frontera sin red de
A1 (`../A1_barrido_epsilon/salidas/tablas/resumen_barrido_epsilon.csv` y
`percapita_por_epsilon.csv`), ε = 3 ya da un Gini ponderado de 0.156830 con 10.21 puertos
por 10 000 en el cuartil vulnerable, y ε = 10 da 0.153305 con 9.73. Con un presupuesto
comparable, la red a 50 kW deja **menos equidad y menos provisión vulnerable** que la
relajación sin red. Es la misma separación entre Gini y provisión per cápita que discute el
manuscrito.

## Lectura para la respuesta a los revisores

1. Con la tecnología de carga que corresponde a cada sede (AC en vivienda y trabajo, DC de
   50 kW solo en la red pública), la holgura de subestación publicada **no muerde**. S2 con la
   restricción dentro del modelo es exactamente el S2 del manuscrito.
2. La restricción solo muerde si todos los puertos fueran DC de 50 kW. En ese caso muerde
   donde está la necesidad (13 de las 15 zonas saturadas son del cuartil vulnerable), y el
   modelo expulsa capacidad de ese cuartil. Por eso la red es una restricción sobre la
   **tecnología** y su despliegue en las zonas densas del suroccidente, no sobre la
   asignación espacial con la mezcla AC/DC prevista.

## Salvedades

Las del dato (A4): foto de diciembre de 2021, capacidad a nivel de subestación y no de
alimentador, holgura compartida con toda la demanda de la ciudad, 42 UTAM sin dato sin tope.
Los 266 puertos que DC50 reubica en zonas sin tope lo hacen precisamente porque allí no hay
dato. En DC50 el reparto de la holgura por área es la fuente principal de incertidumbre de
qué zonas saturan.

## Cómo reproducir

```bash
PY=/Users/jorgeiv500/Documents/social-eq/.venv/bin/python
$PY run_a5.py                             # diagnóstico, sin NEOS (~12 s)
$PY run_a5.py --lanzar --escenario AC7.4  # 1 corrida S2
$PY run_a5.py --lanzar --escenario MIXTO  # 1 corrida S2
$PY run_a5.py --lanzar --escenario DC50   # S0 con red + S2 con red
$PY run_a5.py --consolidar                # salidas/tablas/resumen_a5.csv
```

Correo NEOS: `NEOS_EMAIL` o `ModelConfig.neos_email` (ver `../README.md`).
