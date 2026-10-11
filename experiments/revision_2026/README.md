# Bloque experimental A — revisión JTRG-D-26-00692

Preparación técnica de las corridas aprobadas por el autor el 2026-09-01
(`../decisiones_autor.md`, decisión 2: "aprobado el bloque completo A1–A3; las corridas se
lanzan cuando el runner esté revisado"). **En esta preparación no se lanzó ninguna corrida
NEOS**: todos los runners hacen corrida en seco por defecto y solo envían trabajos con la
bandera explícita `--lanzar`.

Regla del bloque: ninguna cifra inventada — cada valor citado en estos README sale de un
archivo del repositorio con su ruta, y los runners recalculan las derivaciones en tiempo de
ejecución en lugar de llevar números codificados.

## Estado por experimento (actualizado 2026-09-02)

Todas las corridas del bloque se ejecutaron el **2026-09-01** (fechas de los CSV de
`salidas/`). Ninguna cifra de este README es nueva: cada estado remite al archivo que lo
sostiene. El detalle por experimento sigue en el README de cada carpeta.

| Carpeta | Qué es | Responde | Estado | Evidencia |
|---|---|---|---|---|
| `A1_barrido_epsilon/` | Barrido ε ∈ {0.15, 0.20, 0.30} sobre la especificación S2; curva tolerancia → zonas cambiadas → ΔGini/Δcobertura | R1.4, R2.1 | **Completado** — 3 corridas NEOS, las tres `optimal` | `salidas/tablas/resumen_barrido_epsilon.csv` |
| `A2_benchmark_equidad/` | Benchmark de equidad directa (min Σ\|A_i − A_j\| sujeto solo a cuotas, sin acople a prioridad) | R1.6, R2.1 | **Completado** — 2 corridas NEOS (`no_ponderado`, `ponderado_k4`), ambas `optimal` | `salidas/tablas/resumen_benchmark.csv` |
| `A3_escenarios_crecimiento/` | Cuotas escaladas a 2028/2030 con la trayectoria oficial; re-corre S0→S1→S2 por horizonte | R2.2, R2.1 | **Completado** — 6 corridas NEOS (S0/S1/S2 × 2028/2030), las seis `optimal` | `salidas/tablas/resumen_crecimiento.csv` |
| `A6_adopcion_heterogenea/` | Cuotas 2030 de A3 con adopción espacialmente diferenciada (tenencia de automóvil, EM-2023): H1 concentrada, H2 convergencia | R2.2, §5.6 | **Completado (2026-10-10)** — 6 corridas NEOS (S0/S1/S2 × H1/H2), las seis `optimal` | `A6_adopcion_heterogenea/salidas/tablas/resumen_adopcion_heterogenea.csv` |
| `A4_topes_capacidad/` | S2 con topes de capacidad eléctrica por UTAM (POT, 31 subestaciones) | R1.8, R2.1 | **Completado** — diagnóstico determinista, sin NEOS: con los topes tal como se construyen, S2 es infactible a ε = 0.08 por conservación de masa | `salidas/tablas/diagnostico_topes.csv`, `resumen_diagnostico.json` |
| `A7_uso_suelo/` | S2 con elegibilidad por (zona, sede) según las áreas de actividad del POT (Decreto 555, CC BY 4.0) | R2.1, Limitaciones | **Completado (2026-10-10)** — diagnóstico determinista + 4 corridas NEOS, las cuatro `optimal` | `A7_uso_suelo/salidas/tablas/resumen_uso_suelo.csv` |
| `A5_red_en_modelo/` | Holgura de subestación como restricción conjunta en kW **dentro** de S0/S2 (AC7.4, mezcla 7.4/22/50, DC50) | R1.8, R2.1 | **Completado** (2026-10-10) — 4 corridas NEOS (S2 ×3, S0 con red ×1), todas `optimal`; AC7.4 y mezcla reproducen S2 puerto a puerto | `salidas/tablas/resumen_a5.csv`, `manifiesto_*.json` |
| `B1_baselines/` | Baselines simples (regresión, árboles) frente al ST-GNN, validación cruzada local | R1.2 | **Completado** — local, sin NEOS | `resultados/b1_metricas_cv.csv` |
| `B2_ablacion_omega/` | S0–S2 con ω_iv uniforme frente a ω_iv aprendida | R1.3 | **Completado** — 3 corridas NEOS, las tres `optimal` | `salidas/tablas/resumen_ablacion_omega.csv` |
| `B3_tenencia/` | Validación del proxy de necesidad con tenencia en arriendo (EM 2021) | R1.1 | **Completado** — local | `salidas/tabla_correlaciones.csv` |
| `C1_piso_vinculante/` | Variantes de piso que sí muerden (mediana y p75 del conjunto protegido) | R1.5 | **Completado** — 2 corridas NEOS, ambas `optimal` | `salidas/tablas/resumen_piso_vinculante.csv` |
| `E3_mapas/` | Mapa de contexto (Figura 2) y primera pasada cartográfica | R2.m1, R3.5, R3.6 | **Completado** — local | `salidas/*.png` |
| `E6_lambdas/` | Duraciones de viaje por propósito (EODH) para anclar λ | R1.7 | **Completado** — local | `salidas/tablas/duracion_por_proposito.csv` |
| `E7_figuras_impresion/` | Las ocho figuras del manuscrito redibujadas al ancho al que se imprimen | R2.m2 | **Completado** — local; regenerado el 2026-09-02 tras corregir el rotulado de la Figura 7 | `salidas/*.png`, README de la carpeta |
| `comun.py` | Utilidades compartidas: carga de insumos vía `load_minimal_inputs`, lectura de referencias archivadas (u\*, S1, S2), comparación de asignaciones | — | Soporte, no lanza nada | — |

Cómo se integraron al manuscrito y a la carta: `../TABLERO.md`, sección "Corridas y análisis
ejecutados". Los runners siguen haciendo corrida en seco por defecto y solo envían trabajos
NEOS con la bandera explícita `--lanzar`; **no hace falta volver a lanzar nada** para el
reenvío.

`src/social_eq/followup.py` sí se modificó durante la revisión, pero solo en el código de
figuras (E3 y E7: capa urbano/rural/región, elementos cartográficos, rotulado con prueba de
colisión y tamaños de impresión). El modelo y el pipeline de optimización no se tocaron.

## Corridas NEOS ejecutadas

| Experimento | Corridas | Terminación |
|---|---|---|
| A1 | 3 (ε = 0.15, 0.20, 0.30) | `optimal` ×3 |
| A2 | 2 (`no_ponderado`, `ponderado_k4`) | `optimal` ×2 |
| A3 | 6 (S0/S1/S2 × 2028/2030) | `optimal` ×6 |
| B2 | 3 (S0/S1/S2 con ω uniforme) | `optimal` ×3 |
| C1 | 2 (piso mediana, piso p75) | `optimal` ×2 |
| **Total** | **16** | todas `optimal` |

Parámetros de solver en todas: los de la especificación principal del paper
(`solver_time_limit = 3600 s`, `solver_mip_gap = 0.01`, NEOS/CPLEX — `ModelConfig` en
`src/social_eq/followup.py`).

## `NEOS_EMAIL` (solo si se relanza algo)

`solve_model` (`src/social_eq/followup.py`) toma el correo de la variable de entorno
`NEOS_EMAIL`; si no está definida cae al valor por defecto
`ModelConfig.neos_email = jorgei.romerog@utadeo.edu.co`. Antes de lanzar:

```bash
export NEOS_EMAIL="correo_del_autor@dominio"
```

La corrida en seco de cada runner imprime qué correo se usaría. Advertencia:
`scripts/reproduce_followup.py` no interpreta argumentos; cualquier invocación (incluso
`--help`) arranca el pipeline completo y envía S0–S2 a NEOS.

## Validación realizada (2026-09-01, sin NEOS)

Los tres runners se ejecutaron en modo corrida en seco con el `.venv` del repo: cargan los
insumos por el mismo camino de código que el pipeline oficial (142 UTAM, matriz de tiempos
completada, 10,011 pares de equidad, 426 variables enteras), leen u\* = 62.8284379878768 de la
tabla archivada, derivan las cuotas 2028/2030 desde el CSV oficial e imprimen el plan de
corridas sin enviar nada.

## Impedimentos encontrados

**Variante per cápita de A2 (dato):** 22 de las 142 UTAM no tienen `dane_population`
utilizable en `data/inputs/utam_social_vulnerability.csv` (21 vacías y UTAM590 en 0; borde
rural 5xx–7xx). El runner aborta esa variante con diagnóstico en lugar de imputar población;
la decisión queda abierta en el README de A2. La variante principal (A_i absoluta) no se ve
afectada.

Ninguno de código para A1 y A3: `solve_equity(..., epsilon=...)` y
`solve_equity_refined(..., epsilon=...)` aceptan valores arbitrarios (el tope 0.10 solo vivía
en la propiedad `robustness_epsilon_fast`), y las cuotas son un diccionario del config que se
reemplaza con `dataclasses.replace`. Para A2, la única salvedad es que la ruta
Pyomo→NEOS→CPLEX de este repo solo se ha ejercitado con LP/MILP: el QP directo que menciona
R1.6 es viable en CPLEX pero exigiría validar el envío MIQP, por eso el borrador implementa la
linealización L1 (equivalente en espíritu y comparable con el objetivo publicado) y deja el QP
como decisión abierta.
