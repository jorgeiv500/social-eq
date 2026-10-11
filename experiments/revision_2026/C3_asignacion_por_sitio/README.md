# C3 — De la zona al sitio: asignación de puertos a parqueaderos

Revisión de JTRG-D-26-00692. Se aplica el diseño fijado en [`preregistro.md`](preregistro.md)
(2026-10-11 11:56, con la Adenda 1 de las 12:03; sello `preregistro.sha256`). El preregistro no se
editó. Este experimento **no** toca `main.tex`, la carta, `src/` ni los resultados publicados, y
**no** lanza NEOS: todo corre en local.

## Cómo reproducir

```bash
PY=/Users/jorgeiv500/Documents/social-eq/.venv/bin/python
OMP_NUM_THREADS=1 $PY demanda_c3.py --K 5 --njobs 4   # ~18 min con 4 procesos (B6 corría en paralelo)
$PY asignar_c3.py                                      # ~2 min
$PY mapa_c3.py                                         # mapa a ancho de impresión
```

`EM2023_DIR` apunta a la carpeta con `ZAT2023.shp` y los módulos de la EM-2023 (la misma de
`scripts/retrain_stgnn.py`). Insumos: S2 publicado (`paper_v6_experiments/tables/scenario_S2.csv`,
vía `comun.leer_asignacion_paper`), capa de candidatos y llegadas de C2, cuartil y población de B5
(`b5_zonas_S2.csv`, cuartil 4 = vulnerable) y la caché y el código de B6.

## Método

**Nivel 1 (zona).** Puertos por UTAM y sede del S2 publicado, sin cambios (7.352 residenciales,
2.739 de trabajo y 779 públicos). Por A4, al sitio van solo los de **trabajo y públicos**: 3.518 puertos.

**Demanda por ZAT (A5, `demanda_c3.py`).** Las predicciones archivadas de B6 son del nivel de
*todos* los extremos de viaje del modo AUTO agrupado (conductores y pasajeros, salidas y llegadas),
no de llegadas de conductores. Por eso se reajustaron los mismos estimadores de B6, importando su
código (`run_b6.py`): EB (contracción bayesiana empírica), RF con rezago espacial, GCN y ST-GNN.
Se usaron las mismas covariables de hogares, el mismo vecindario (k = 8, IDW) y las mismas rejillas
de configuración. El objetivo pasa a ser los 8.110 viajes «Vehículo privado como conductor»
contados en su ZAT de destino. Ajuste cruzado: K = 5 particiones 50/50 de hogares. En cada una,
cada mitad hace de A (ajuste A1→A2 y A2→A1, validación A2→A1, igual que en B6), y la predicción
con entrada A estima el nivel poblacional, porque B6 reescala cada submuestra a la población. El
nivel suavizado es la media de las 10 predicciones. Regla A5: en las ZAT con ≥ 30 llegadas en
muestra se usa el valor expandido observado de C2. Salida: `salidas/demanda_zat_estimadores.csv`.
Spearman entre estimadores: RF–ST-GNN 0,95; RF–GCN 0,92; RF–EB 0,61.

**Candidatos.**
- Principal: parqueaderos fuera de vía, es decir, OSM `amenity=parking` fuera de vía más manzanas
  Catastro de «depósitos y parqueaderos».
- A2: se excluyen los OSM con `access=private/no`.
- A3: una manzana Catastro que contiene un punto OSM elegible se descarta y cuenta como OSM.
- Resultado: «ambas» = 2.412 sitios; «OSM» = 2.112; «Catastro» = 336.

**Capacidad.** c_s = ⌊f · cupos⌋, con mínimo 1 si hay cupos y f = 10 %. Cupos sin dato (A1):
mediana de los OSM fuera de vía con cupos declarados > 0, que es 30 (n = 50), así que c = 3 por
sitio imputado. Las ZPP usan sus cupos de auto.

**Nivel 2 (`asignar_c3.py`).** En cada UTAM se toman T_u = puertos de trabajo + públicos. Se
reparten entre sus sitios con peso w_s = D_z(s) / (nº de sitios elegibles en la ZAT de s), con tope
c_s, por *water-filling*: reparto proporcional, se fijan en su tope los sitios que lo superan y se
redistribuye el excedente hasta agotar T_u o la capacidad. Después se redondea al entero por mayor
resto, respetando los topes. Instalados I_u = mín(T_u, Σc_s). La brecha es T_u − I_u. La sede de
cada puerto en sitio se asigna por mayor resto, de modo que las filas y columnas cuadran.

**Métricas.**
1. Fracción de la cuota instalable, Σ I_u / Σ T_u, por cuartil de vulnerabilidad.
2. Distancia euclidiana (EPSG:9377) desde el punto representativo de cada ZAT de Bogotá al sitio
   más cercano con al menos un puerto, ponderada por población, por cuartil, con la brecha Q4 − resto
   y la razón Q4/resto.
3. En la reubicación: Gini ponderado de la accesibilidad (`comun.calcular_accesibilidad` y
   `weighted_gini` de `src/`, pesos 1 + ν) y provisión por 10.000 habitantes (misma regla que
   `scripts/intervention_size.py`). Control: con S2 publicado se reproduce **0,164653** y
   **10,42 frente a 15,62**.

## Desviaciones (además de la Adenda 1), con su motivo

1. **Demanda reajustada en vez de leída de B6.** B6 predice otro objeto: todos los extremos AUTO,
   no las llegadas de conductores. Para la selección de configuración se usa el error del
   log-nivel, porque ω no interviene aquí. El nivel suavizado es la media de expm1 de predicciones
   en log, lo que lo sesga un poco a la baja frente al observado. Solo afecta a las 44 ZAT con
   ≥ 30 llegadas, que usan el observado. Dentro de una UTAM solo importan los pesos relativos.
2. **OSM `parking=street_side/on_kerb/lane` (172) se tratan como «en vía».** Salen del principal
   («fuera de vía») y entran en la sensibilidad «en vía» junto con las ZPP.
3. **Los OSM con cupos declarados = 0 tienen capacidad 0.** Se aplica literalmente «mínimo 1 si hay
   cupos». La mediana de A1 se calcula sobre cupos > 0.
4. **Sitios en ZAT sin demanda.** Solo reciben puertos cuando los sitios con demanda ya están en su
   tope; entonces el reparto entre ellos es uniforme.
5. **26 UTAM de 20 municipios vecinos (Soacha, Chía, Mosquera, Funza, Facatativá, Zipaquirá, …).**
   Tienen cuota S2 pero quedan fuera de la capa de candidatos de C2, que solo cubre Bogotá. Su brecha
   (528 de 3.518 puertos) existe por construcción y no indica falta de parqueaderos. Por eso las
   fracciones se reportan con las 142 UTAM y con las **116 UTAM de Bogotá**. La lectura principal
   usa las de Bogotá.
6. **Reubicación.**
   - Contigüidad: polígonos UTAM que se tocan (búfer de 1 m).
   - «Más cercana»: distancia entre centroides.
   - Si las vecinas de primer orden no tienen capacidad libre, se pasa a anillos de contigüidad
     sucesivos.
   - Las UTAM se procesan de mayor a menor brecha.
   - Los puertos que no se pueden reubicar se dejan en su UTAM de origen para el Gini y la
     provisión, de modo que la cuota total sigue siendo 10.870.
   - La variante preregistrada mueve también la brecha de los municipios vecinos hacia Bogotá
     (desviación 5). Se añade una variante **no preregistrada**, `reubicacion_solo_bogota`, en la
     que esa brecha no se mueve. Es la versión interpretable.
7. **Población por ZAT.** La población DANE de cada UTAM se reparte entre sus ZAT en proporción a
   las personas expandidas de la EM-2023 (hogar × `perstotal_hg`), o por área si la UTAM no tiene
   hogares encuestados. Las ZAT se asignan a la UTAM que contiene su punto representativo.
8. **Comparación de estimadores.** El denominador es el total de puertos ubicados en sitio con RF
   (2.589 en «ambas»). La fracción instalable no depende del estimador por construcción: la
   capacidad se agota a escala UTAM. Lo que cambia es *en qué sitio* quedan los puertos.

## Resultados

### Métricas primarias — principal («ambas», 10 %, RF)

| Cuartil ν | Fracción instalable (116 UTAM de Bogotá) | Fracción instalable (142 UTAM) | Distancia ponderada al sitio con cargador (m) | Distancia a cualquier candidato (m) |
|---|---:|---:|---:|---:|
| 1 (menos vulnerable) | 0,968 | 0,877 | 241 | 208 |
| 2 | 0,878 | 0,864 | 464 | 416 |
| 3 | 0,904 | 0,538 | 347 | 333 |
| **4 (vulnerable)** | **0,790** | **0,790** | **338** | 335 |
| Resto (1–3) | 0,929 | 0,702 | 346 | 320 |
| Ciudad | 0,866 | 0,736 | — | — |
| Brecha Q4 − resto | **−13,9 pp** | +8,8 pp | **−8 m** (razón 0,98) | +14 m |

Brecha en puertos: 401 de 2.990 en Bogotá, de ellos 285 en el cuartil 4, que tiene 1.356 de cuota.
En todo el sistema son 929 de 3.518. La lista por UTAM está en `salidas/brecha_por_utam.csv` y el
mapa en `salidas/mapa_sitios_brecha.png` (2.400 px de ancho, 320 dpi, con norte, escala y nota de
límites).

### Control de sesgo de fuente (principal; Bogotá)

| Candidatos | Q1 | Q2 | Q3 | Q4 | Resto | Dist. Q4 / resto (m) |
|---|---:|---:|---:|---:|---:|---:|
| Ambas | 0,968 | 0,878 | 0,904 | 0,790 | 0,929 | 338 / 346 |
| Solo OSM | 0,964 | 0,854 | 0,845 | 0,704 | 0,900 | 360 / 401 |
| Solo Catastro | 0,342 | 0,289 | 0,261 | 0,175 | 0,301 | 761 / 855 |

### Sensibilidades (candidatos «ambas»; fracción instalable en Bogotá)

| Escenario | Q4 | Resto | Brecha Q4 − resto | Dist. Q4 / resto (m) |
|---|---:|---:|---:|---:|
| Principal (10 %) | 0,790 | 0,929 | −13,9 pp | 338 / 346 |
| Capacidad 5 % | 0,373 | 0,729 | −35,6 pp | 337 / 337 |
| Capacidad 20 % | 0,923 | 0,972 | −4,9 pp | 340 / 348 |
| Cupos imputados p25 (10) | 0,381 | 0,736 | −35,5 pp | 337 / 337 |
| Cupos imputados p75 (58,5) | 0,904 | 0,963 | −5,9 pp | 340 / 348 |
| + ZPP y OSM en vía | 0,816 | 0,935 | −11,9 pp | 332 / 343 |
| A4: residenciales de Q4 también en parqueaderos | **0,403** | 0,929 | −52,6 pp | 336 / 346 |

En A4 solo cabe el **40 %** de los 2.449 puertos residenciales del cuartil vulnerable; quedan 1.466
sin sitio. Con solo OSM cabe el 35 % y con solo Catastro el 6 %. Las mismas sensibilidades con
OSM y con Catastro están en `salidas/resultados_escenarios.csv`.

**Reubicación de la brecha a la UTAM vecina (Δ frente a S2 publicado: Gini 0,164653; provisión Q4 10,42).**

| Variante | Candidatos | Reubicados | Sin reubicar | Gini pond. | ΔGini | Provisión Q4 / resto | ΔProvisión Q4 | Distancia media (km) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Preregistrada | ambas | 749 | 180 | 0,166233 | +0,0016 | 10,61 / 16,13 | +0,19 | 6,1 |
| Solo Bogotá (no prerreg.) | ambas | 401 | 528 | 0,164033 | −0,0006 | **10,00 / 16,05** | **−0,42** | 4,3 |
| Preregistrada | OSM | 914 | 180 | 0,1661 | +0,0015 | 10,29 / 16,45 | −0,13 | 6,5 |
| Solo Bogotá | OSM | 566 | 528 | 0,1636 | −0,0010 | 10,04 / 16,00 | −0,38 | 4,6 |
| Ambas variantes | Catastro | 279 | 2.510 | 0,1609 | −0,0038 | 9,65 / 16,39 | −0,76 | 14,1 |

En la variante solo Bogotá («ambas»), el cuartil vulnerable **exporta 179 puertos** a UTAM menos
vulnerables y recibe 26. En la preregistrada, el +0,19 se debe a que entran a Q4 puertos de los
municipios vecinos, que no tienen capa de candidatos (desviación 5). Flujos completos en
`salidas/reubicacion_movimientos.csv`.

### Comparación de estimadores (puertos que cambian de sitio frente a RF con rezago; ½ Σ|Δ| / total del nivel 2)

| Candidatos | EB | GCN | ST-GNN |
|---|---:|---:|---:|
| Ambas (2.589 puertos en sitio) | 6,1 % (159) | 4,7 % (123) | **3,4 % (87)** |
| OSM | 6,3 % | 4,0 % | 2,8 % |
| Catastro | 1,1 % | 1,4 % | 1,0 % |

## Lectura

- **Qué no cambia:** el patrón zonal. El cuartil vulnerable queda por debajo del resto, y el nivel
  de sitio lo **acentúa**. Con la regla del 10 % cabe en parqueaderos el 79 % de su cuota de
  trabajo y públicos, frente al 93 % del resto de Bogotá. La dirección se mantiene en todas las
  sensibilidades y en las tres fuentes. La magnitud depende mucho del supuesto de capacidad: la
  diferencia va de 5 a 36 pp entre 20 % y 5 %. La elección del estimador de demanda no altera la
  fracción instalable y mueve solo entre el 3 % y el 6 % de los puertos entre sitios.
- **Qué cambia:** la brecha de sitios no es de proximidad. El cuartil vulnerable es denso y tiene
  un parqueadero con cargador a la misma distancia ponderada que el resto, unos 340 m. Lo que le
  falta es capacidad: pocos cupos por habitante y casi ninguno conocido. Si la brecha se resuelve
  reubicando puertos en la UTAM vecina, el Gini ponderado apenas se mueve (|Δ| ≤ 0,004). La
  provisión del cuartil vulnerable baja de 10,42 a 10,00 por 10.000 habitantes, porque Q4 exporta
  puertos.
- **La carga residencial es el cuello de botella:** si los hogares vulnerables sin garaje tuvieran
  que cargar en parqueaderos, solo cabría el 40 % de su cuota residencial.
- **Límites:** la capacidad es imputada en el 98 % de los sitios OSM y en todos los de Catastro.
  Catastro solo cubre manzanas de uso predominante de parqueaderos, lo que explica que su brecha sea
  mucho mayor. La distancia es euclidiana, no por red. No hay restricción eléctrica por sitio.

## Archivos

| Archivo | Contenido |
|---|---|
| `demanda_c3.py`, `asignar_c3.py`, `mapa_c3.py` | scripts |
| `salidas/asignacion_por_sitio_principal.csv` | los 2.412 sitios del principal: cupos (e imputación), capacidad, demanda de su ZAT, peso, puertos por sede, coordenadas EPSG:9377 |
| `salidas/asignacion_por_sitio_todos_escenarios.csv.gz` | sitios con puertos en cada escenario × fuente |
| `salidas/brecha_por_utam.csv` | cuota, capacidad, instalados y brecha por UTAM y sede, escenario y fuente |
| `salidas/resultados_escenarios.csv` | todas las métricas, por escenario y fuente (sufijo `_bogota` = 116 UTAM de Bogotá) |
| `salidas/asignacion_utam_reubicacion*_*.csv` | asignación UTAM × sede tras reubicar (entrada del Gini) |
| `salidas/reubicacion_movimientos.csv` | origen → destino, sede, puertos, anillo y distancia |
| `salidas/demanda_zat_estimadores.csv`, `demanda_predicciones_crudas.csv.gz`, `demanda_meta.csv` | demanda por ZAT y estimador; configuraciones elegidas |
| `salidas/zat_poblacion_cuartil.csv` | población, cuartil y punto de cada ZAT de Bogotá |
| `salidas/referencias_c3.json` | Gini y provisión de S2 recalculados; cupos imputados |
| `salidas/mapa_sitios_brecha.png` / `.pdf` | mapa a ancho de impresión |
| `salidas/log_*.txt` | registros de ejecución |
