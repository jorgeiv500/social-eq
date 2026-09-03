# A4 — S2 con topes de capacidad eléctrica por UTAM (R1.8, R2.1)

Experimento de la revisión JTRG-D-26-00692. Pregunta: ¿qué pasa con la
asignación de equidad S2 cuando se impone la restricción del mundo real que
piden el Revisor 1 (punto 8) y el Revisor 2 (punto 1): la capacidad disponible
de la red eléctrica por zona?

## Fuente del dato (verificada, ver `propuestas/r2/verificacion_fuentes_A4_E2.md`)

- **"Capacidad eléctrica. POT Bogotá D.C."** — Secretaría Distrital de
  Planeación, Datos Abiertos Bogotá, licencia CC BY 4.0. Áreas de servicio de
  31 subestaciones AT/MT con capacidad instalada (MVA, red 11.4 kV) y factor de
  utilización. Corte: 29/12/2021 (Decreto 555, POT). Crudo:
  `propuestas/r2/datos_a4/capacidad_electrica_pot555.gpkg` (se usa el GPKG; el
  GeoJSON del portal declara mal el CRS).
- Holgura por subestación: `headroom = CAPACIDAD_ × (1 − CARGA_11)` (ambos
  atributos publicados). Total: 897.04 MVA en 31 subestaciones; de esa holgura,
  896.27 MVA quedan dentro del marco zonal UTAM tras el cruce espacial.
- Reparto a UTAM proporcional al área intersectada de cada zona de servicio
  (cruce espacial en EPSG:9377 con `UTAM2023.shp`, el mismo shapefile del
  pipeline). `preparar_topes.py` reproduce exactamente el demo archivado
  (`datos_a4/cap_i_demo_utam.csv`, dif. máx. 0.000000 MVA) y añade la **fracción
  de cobertura** de cada UTAM, de la que depende la regla de topes.

## De MVA a topes de puertos (todo declarado, nada inventado)

1. **Potencia por puerto — escenarios declarados y citables.** Se usan las
   categorías estándar de potencia de recarga de la clasificación europea
   (AFIR, Reglamento (UE) 2023/1804) publicadas por el European Alternative
   Fuels Observatory de la Comisión Europea
   (https://alternative-fuels-observatory.ec.europa.eu/general-information/recharging-systems,
   consultado 2026-09-01): recarga AC monofásica «lenta» P < 7.4 kW; AC
   trifásica «media» 7.4 ≤ P ≤ 22 kW; DC «rápida» desde 50 kW.
   - Escenario **AC7.4**: 7.4 kW nominales por puerto (el tope más laxo).
   - Escenario **AC22**: 22 kW nominales por puerto.
   - Escenario **DC50**: 50 kW por puerto — solo como **cota de estrés en el
     diagnóstico** (un puerto residencial DC de 50 kW no es representativo; no
     se optimiza).
2. **Conversión.** `tope_puertos(i) = cap_i_MVA × 1000 / kW`. Supuestos
   declarados: factor de potencia ≈ 1 (1 MVA ≈ 1 MW) y potencia **nominal** por
   puerto sin factor de simultaneidad. Nótese que el barrido 7.4→22 kW cubre el
   mismo orden de variación (×3) que cubriría un factor de coincidencia
   declarado; además toda la holgura de 11.4 kV se supone hipotéticamente
   disponible para carga de VE, lo que hace el tope **generoso**: si aun así
   muerde, el hallazgo es conservador.
3. **Regla de aplicación por cobertura (declarada).**
   - **Tope** solo donde las zonas de servicio mapeadas cubren **≥50 %** del
     área de la UTAM: 100 UTAM.
   - **Sin tope** para las **33 UTAM con cobertura <5 %** (códigos ≥500, las 5
     UPR rurales, UTAM64 y UTAM89): están fuera de las áreas de servicio
     mapeadas y no hay dato para acotarlas; imponerles 0 (o el valor de una
     astilla de borde) sería fabricar la restricción.
   - **Sin tope** también para las **9 UTAM con cobertura parcial 5–50 %**
     (refinamiento declarado del mismo principio: la mayor parte de la zona no
     está mapeada, y el tope de la fracción cubierta subestimaría la capacidad).
     Sensibilidad anotada: si se bajara el umbral a 5 %, la única zona
     adicional que podría morder es UTAM103 (Parque Salitre, cobertura 36 %) en
     el escenario AC22.
4. **Estructura del tope en el modelo.** `solve_equity_refined(..., caps=...)`
   (`src/social_eq/followup.py`, líneas 735–736) aplica el tope **por (zona,
   sede)**: `n[i,v] ≤ caps[i,v]`. Un tope conjunto zonal no es expresable sin
   modificar `src/` (prohibido), así que el tope zonal se **reparte entre las 3
   sedes proporcionalmente al target** (`caps[i,v] = tope_i · t[i,v]/Σ_v
   t[i,v]`; reparto igualitario si el target zonal es 0). Documentado: esto es
   ligeramente más restrictivo que un tope conjunto (la mezcla de sedes no
   puede desviarse del target dentro del tope zonal).
   - «Sin tope» se implementa como `caps[i,v] = cuota_v`, no restrictivo por
     construcción (la cuota es la suma de todas las zonas).

## Diagnóstico previo sin NEOS

`run_a4.py` (corrida en seco, por defecto) produce
`salidas/tablas/diagnostico_topes.csv` (142 UTAM × 3 escenarios) y
`salidas/tablas/resumen_diagnostico.json`, con:

- dónde **muerde** el tope: `tope < target` y `tope < asignación S2 archivada`;
- **factibilidad de cuotas**: Σ topes ≥ cuota por sede;
- **cota inferior de desviación**: por conservación de masa dentro de cada
  sede, todo puerto que el tope expulsa por debajo del target reaparece por
  encima del target en otra zona, luego `Σu ≥ 2·D` con
  `D = Σ max(0, target − tope)`. Si `2·D > u*·(1+0.08)` el modelo S2 con topes
  es **infactible a ε = 0.08 de forma demostrable, sin gastar NEOS**.

## Hipótesis ex ante (registradas antes de lanzar cualquier corrida NEOS)

- **H1 (muerde en zonas concentradas):** si los topes muerden, será en pocas
  UTAM donde el modelo de demanda concentra muchos puertos sobre subestaciones
  con poca holgura (candidatas por factor de carga alto: Fontibón 0.812, Bosa
  0.853, Techo 0.855, San Mateo 0.841). En ese caso la asignación con
  restricción de red se desplaza hacia zonas vecinas con holgura y eso
  **cuantifica el "real-world constraint" que pide R2.1**. Si el déficit
  forzado supera el presupuesto de fidelidad (`2·D > u*·1.08`), el resultado
  esperado de la corrida de confirmación es **infactible**, y la magnitud del
  conflicto (cuántos puertos no caben y dónde) es en sí misma la respuesta.
- **H0 (no muerde):** si con ningún escenario razonable el tope muerde, la
  respuesta honesta es que, a la escala actual de cuotas (10 870 puertos), la
  capacidad de subestación no es la restricción activa; se reporta igual, con
  una única corrida NEOS de confirmación.

Ambos resultados sirven; se reporta el que salga.

## Plan de corridas NEOS (máximo 3)

Especificación S2 idéntica en lo demás: ν = índice de vulnerabilidad, κ = 4,
cuotas del paper, u* archivado (62.8284…), CPLEX vía NEOS.

1. **Confirmación** — escenario AC7.4 (el tope más laxo) con ε = 0.08 exacto.
   Si es infactible, AC22 y DC50 lo son *a fortiori* (topes estrictamente más
   estrechos), por dominancia; no se gastan corridas en confirmarlo.
2. **Cuantificación AC7.4** — mismo modelo con el presupuesto de desviación
   ampliado exactamente en la cota inferior del desplazamiento forzado:
   `ε_relax = [u*·1.08 + 2·D_(7.4)]·1.02 / u* − 1` (margen +2 % por
   integralidad, declarado). Así el plan resultante es «S2 mínimamente ajustado
   para respetar la red», y las métricas vs S2 archivado (ΔGini a 6 decimales,
   cobertura vulnerable, puertos movidos, topes activos) miden el costo de la
   restricción, no una relajación gratuita.
3. **Cuantificación AC22** — ídem con `D_(22)`.

Corrida en seco obligatoria antes de `--lanzar`. Las corridas de A1 (ε = 0.15,
0.20, 0.30 sin topes) sirven de control parcial para separar el efecto del
tope del efecto de relajar ε.

## Cómo correr

```bash
# 1) cruce espacial (local, sin red):
/Users/jorgeiv500/Documents/social-eq/.venv/bin/python preparar_topes.py
# 2) diagnóstico en seco (sin NEOS):
/Users/jorgeiv500/Documents/social-eq/.venv/bin/python run_a4.py
# 3) corridas NEOS aprobadas:
/Users/jorgeiv500/Documents/social-eq/.venv/bin/python run_a4.py --lanzar
```

## Salvedades del dato (van también en la respuesta a los revisores)

- Foto a diciembre de 2021 (POT); no recoge expansiones posteriores de Enel.
- Capacidad a nivel de **subestación**, no de alimentador/circuito.
- La holgura es de toda la demanda de la ciudad, no un cupo reservado para VE.
- Como **tope relativo entre zonas** (lo que piden R1.8/R2.1) es defendible;
  como valor absoluto es una cota superior generosa.

---

## Resultados del diagnóstico en seco (2026-09-01, `run_a4.py`, sin NEOS)

Salidas: `salidas/tablas/diagnostico_topes.csv` (142 UTAM × 3 escenarios) y
`salidas/tablas/resumen_diagnostico.json`.

| Escenario | kW/puerto | UTAM con tope | Muerde vs. target | D (déficit forzado) | 2·D | Presupuesto (1+ε)u\* | ¿Infactible a ε=0.08? |
|---|---|---|---|---|---|---|---|
| AC7.4 | 7.4 | 100 | 1 | 453.18 | 906.36 | 67.85 | **sí, demostrable** |
| AC22 | 22 | 100 | 3 | 1 052.23 | 2 104.46 | 67.85 | **sí, demostrable** |
| DC50 | 50 | 100 | 6 | 1 633.81 | 3 267.61 | 67.85 | **sí, demostrable** |

**Veredicto: H1, en su rama de conflicto.** El tope muerde y el conflicto está concentrado:
en el escenario más laxo (AC7.4) basta **UTAM116** — target 944.5 puertos contra un tope de
491.4 — para forzar un déficit de 453 puertos. Por conservación de masa dentro de cada sede,
la desviación inducida es al menos 2·D, es decir más de un orden de magnitud por encima del
presupuesto de fidelidad. **No se gastan corridas NEOS**: la infactibilidad de S2 con topes a
ε = 0.08 queda demostrada aritméticamente, y por dominancia AC22 y DC50 lo son *a fortiori*.

Zonas donde muerde (AC22): UTAM116 (déficit 779.3), UTAM117 (187.2), UTAM115 (85.7).

Aplicado al manuscrito en §5.5 y en la §Limitations el 2026-09-01; responde R1.8 y cierra el
hueco declarado en `propuestas/r2/respuesta_R2_1_esqueleto.md` §5.
