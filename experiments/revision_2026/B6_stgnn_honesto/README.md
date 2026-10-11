# B6 — ¿Supera un ST-GNN sin fuga a los competidores clásicos? (prueba de falsación rápida)

Revisión JTRG-D-26-00692. Experimento **para decidir**, no para el manuscrito: no toca `main.tex`,
la carta, `src/` ni ningún resultado publicado, y no lanza NEOS (S0 se resuelve exacto por
redondeo, como en B5). Protocolo fijado en [`preregistro.md`](preregistro.md) a las 10:54 del
2026-10-11, sellado con SHA-256 a las 10:55:31 (`preregistro.sha256`), antes de escribir el código
de modelado y de ver cualquier resultado.

> Ubicación canónica: `experiments/revision_2026/B6_stgnn_honesto/` (los scripts resuelven rutas
> desde ahí). `salidas/cache_b6.pkl` contiene microdatos de la encuesta y no se publica.

## Veredicto: **NO-GO**

En la escala de decisión (ZAT, régimen principal A/B por hogares, viajes en AUTO), el ST-GNN
**no mejora** al mejor competidor clásico (RF con rezago espacial). Queda prácticamente empatado
y, de hecho, ligeramente peor:

| | wMAE de ω (ZAT, AUTO, K = 10) |
|---|---|
| RF con rezago espacial (mejor de EB y RF) | 0,1160 |
| ST-GNN inductivo | 0,1169 |
| Diferencia ST-GNN − RF (IC 95 %, bootstrap por localidad) | +0,0009 (−0,0004; +0,0019) |
| Mejora relativa (IC 95 %) | −0,8 % (−1,6 %; +0,3 %) |

El criterio exigía un IC que excluyera 0 **y** una mejora ≥ 10 %. No se cumple ninguna de las
dos condiciones. (`salidas/veredicto.json`)

## Tabla de resultados — régimen principal (K = 10 particiones A/B de hogares)

wMAE de ω = error absoluto medio de la participación por ventana de sede (residencial 22–5,
trabajo 8–17, público 18–21) contra la mitad B, ponderado por los viajes expandidos de B. Entre
paréntesis, la DE entre particiones (suelo de ruido). Puertos S0 = puertos mal ubicados en S0
exacto (cuotas 2024: 7.352 / 2.739 / 779 = 10.870), a escala UTAM, frente a las metas de la mitad
B (media ± DE sobre las 10 particiones).

| Escala | Modo | Competidor | wMAE ω | wMAE log-nivel | Spearman nivel | Puertos S0 |
|---|---|---|---|---|---|---|
| ZAT | AUTO | (i) media ciudad | 0,1236 (0,0020) | 1,379 | — | 4.071 ± 184 |
| ZAT | AUTO | (ii) crudo A | 0,1489 (0,0062) | 0,706 | 0,535 | 3.388 ± 203 |
| ZAT | AUTO | (iii) EB / James-Stein | 0,1182 (0,0029) | 0,582 | 0,489 | 3.443 ± 241 |
| ZAT | AUTO | (iv) RF + rezago | **0,1160** (0,0024) | 0,598 | 0,653 | **2.965** ± 196 |
| ZAT | AUTO | (v) IDW / grafo | 0,1215 (0,0022) | 0,611 | 0,493 | 3.482 ± 195 |
| ZAT | AUTO | (vi) GCN | 0,1172 (0,0029) | 0,642 | 0,564 | 3.217 ± 257 |
| ZAT | AUTO | (vii) ST-GNN | 0,1169 (0,0029) | **0,563** | 0,636 | 3.105 ± 144 |
| UTAM | AUTO | (i) media ciudad | 0,0694 (0,0031) | 1,010 | — | 4.349 ± 82 |
| UTAM | AUTO | (ii) crudo A | 0,0746 (0,0031) | 0,319 | 0,892 | 3.418 ± 188 |
| UTAM | AUTO | (iii) EB | 0,0613 (0,0030) | **0,305** | 0,898 | 3.010 ± 139 |
| UTAM | AUTO | (iv) RF + rezago | 0,0591 (0,0031) | 0,338 | **0,900** | **2.917** ± 162 |
| UTAM | AUTO | (v) IDW | 0,0602 (0,0033) | 0,456 | 0,776 | 3.530 ± 160 |
| UTAM | AUTO | (vi) GCN | **0,0587** (0,0041) | 0,364 | 0,852 | 3.091 ± 141 |
| UTAM | AUTO | (vii) ST-GNN | 0,0614 (0,0040) | 0,359 | 0,867 | 3.060 ± 233 |
| ZAT | todos | (i) media ciudad | 0,0588 (0,0008) | 1,184 | — | 3.272 ± 59 |
| ZAT | todos | (ii) crudo A | 0,0607 (0,0014) | 0,310 | 0,897 | 1.354 ± 92 |
| ZAT | todos | (iii) EB | 0,0515 (0,0009) | **0,296** | 0,841 | 1.508 ± 107 |
| ZAT | todos | (iv) RF + rezago | **0,0495** (0,0013) | 0,314 | **0,924** | 1.328 ± 89 |
| ZAT | todos | (v) IDW | 0,0518 (0,0012) | 0,421 | 0,645 | 1.652 ± 117 |
| ZAT | todos | (vi) GCN | 0,0502 (0,0015) | 0,336 | 0,903 | 1.411 ± 102 |
| ZAT | todos | (vii) ST-GNN | 0,0502 (0,0016) | 0,329 | 0,910 | **1.326** ± 73 |
| UTAM | todos | (i) media ciudad | 0,0321 (0,0007) | 0,608 | — | 3.504 ± 46 |
| UTAM | todos | (ii) crudo A | 0,0260 (0,0023) | 0,133 | 0,972 | 1.269 ± 82 |
| UTAM | todos | (iii) EB | 0,0242 (0,0019) | 0,133 | 0,972 | **1.215** ± 80 |
| UTAM | todos | (iv) RF + rezago | **0,0229** (0,0013) | 0,165 | 0,967 | 1.256 ± 77 |
| UTAM | todos | (v) IDW | 0,0236 (0,0012) | 0,367 | 0,764 | 2.076 ± 55 |
| UTAM | todos | (vi) GCN | 0,0244 (0,0011) | 0,167 | 0,961 | 1.298 ± 84 |
| UTAM | todos | (vii) ST-GNN | 0,0237 (0,0012) | 0,184 | 0,959 | 1.243 ± 82 |

**ST-GNN frente al mejor de EB y RF** (bootstrap por bloques de localidad, 21 localidades,
2.000 réplicas; `salidas/bootstrap_stgnn_vs_mejor.csv`):

| Escala | Modo | Régimen | Mejor clásico | Δ wMAE (ST-GNN − mejor) [IC 95 %] | Mejora relativa [IC 95 %] |
|---|---|---|---|---|---|
| **ZAT** | **AUTO** | **principal (decisión)** | RF | +0,0009 [−0,0004; +0,0019] | −0,8 % [−1,6 %; +0,3 %] |
| ZAT | todos | principal | RF | +0,0006 [−0,0003; +0,0014] | −1,3 % [−3,0 %; +0,5 %] |
| UTAM | AUTO | principal | RF | +0,0024 [+0,0010; +0,0039] | −4,0 % [−6,9 %; −1,7 %] |
| UTAM | todos | principal | RF | +0,0008 [−0,0001; +0,0016] | −3,6 % [−7,7 %; +0,5 %] |
| ZAT | AUTO | *kriging* por localidades | RF | +0,0029 [+0,0016; +0,0038] | −2,4 % [−3,2 %; −1,4 %] |
| ZAT | todos | *kriging* por localidades | RF | +0,0014 [+0,0001; +0,0025] | −2,5 % [−4,6 %; −0,1 %] |
| UTAM | AUTO | *kriging* por localidades | RF | +0,0035 [+0,0002; +0,0070] | −5,5 % [−10,7 %; −0,4 %] |
| UTAM | todos | *kriging* por localidades | EB | +0,0009 [+0,0002; +0,0016] | −3,3 % [−5,9 %; −0,6 %] |
| ZAT | AUTO | tiempo (→ junio) | RF | +0,0021 [−0,0022; +0,0069] | −1,8 % [−7,3 %; +1,7 %] |
| ZAT | AUTO | tiempo (→ octubre) | EB | +0,0034 [−0,0020; +0,0080] | −2,9 % [−6,5 %; +1,5 %] |
| ZAT | todos | tiempo (→ octubre) | RF | −0,0033 [−0,0046; +0,0021] | +4,0 % [−1,9 %; +5,8 %] |

(Filas restantes del régimen temporal, todas con IC que incluye 0: `salidas/bootstrap_stgnn_vs_mejor.csv`.)

En **ninguna** de las 16 combinaciones de escala, modo y régimen el ST-GNN supera al mejor
clásico con IC que excluya 0, y su mejor caso (+4,0 %, ZAT-todos contra octubre, 685 hogares)
queda lejos del 10 %. En UTAM-AUTO (principal) y en *kriging* (las cuatro combinaciones) es
**significativamente peor** que el mejor clásico.

## Suelo de ruido: por qué el margen para ganar es casi nulo

Fiabilidad entre mitades (correlación ponderada de ω entre A y B, zonas con dato en ambas;
`salidas/fiabilidad_entre_mitades.csv`):

| Escala | Modo | r residencial | r trabajo | r público | wMAE A vs B | Cota aprox. alcanzable (÷√2) |
|---|---|---|---|---|---|---|
| ZAT | AUTO | 0,16 | 0,18 | 0,12 | 0,145 | ≈ 0,103 |
| ZAT | todos | 0,48 | 0,41 | 0,22 | 0,061 | ≈ 0,043 |
| UTAM | AUTO | 0,45 | 0,38 | 0,16 | 0,075 | ≈ 0,053 |
| UTAM | todos | 0,80 | 0,69 | 0,33 | 0,026 | ≈ 0,018 |

- En ZAT-AUTO la mitad de la muestra apenas predice a la otra mitad (r ≈ 0,15). El dato crudo
  (0,149) es **peor que la media de la ciudad** (0,124): con 6–48 viajes en auto por ZAT, el
  perfil de cada zona es casi todo ruido. Todos los métodos que contraen hacia algo (EB, RF, GCN,
  ST-GNN) quedan entre 0,116 y 0,122, a menos de 6 % de la media de la ciudad.
- Incluso un predictor perfecto de la verdad poblacional tendría un error contra B de alrededor
  de 0,103 (aproximación gaussiana: el ruido de B solo). El RF ya está en 0,116: **el margen
  máximo teórico sobre el RF es ≈ 11 %**, así que el umbral del 10 % solo lo alcanzaría un oráculo.
  Esto es una propiedad de los datos, no del modelo.
- En la decisión (S0), la DE entre particiones es de 140–260 puertos (1,3–2,4 % del sistema).
  RF y ST-GNN difieren en 140 puertos en ZAT-AUTO (a favor del RF), dentro de ese ruido. Elegir
  cualquier modelo razonable frente a otro cambia menos puertos que repetir la encuesta con otra
  mitad de hogares (crudo A frente a B: 3.388 puertos mal ubicados, 31 % del sistema).

## Régimen de *kriging* (localidades enteras sin dato en A; K = 5, 5 pliegues de localidades)

| Escala | Modo | media ciudad = crudo | EB | RF + rezago | IDW | GCN | ST-GNN |
|---|---|---|---|---|---|---|---|
| ZAT | AUTO | 0,1246 | 0,1281 | **0,1184** | 0,1281 | 0,1199 | 0,1213 |
| UTAM | AUTO | 0,0714 | 0,0645 | 0,0639 | 0,0648 | **0,0612** | 0,0674 |
| ZAT | todos | 0,0598 | 0,0603 | **0,0546** | 0,0604 | 0,0549 | 0,0559 |
| UTAM | todos | 0,0334 | **0,0277** | 0,0279 | 0,0279 | 0,0291 | 0,0286 |

Con la localidad entera oculta, EB e IDW coinciden por construcción (el prior de EB es el rezago
IDW cuando la localidad no tiene dato y la zona no tiene dato propio). El ST-GNN entrenado con
enmascaramiento estilo IGNNK —el régimen en el que la literatura de *kriging* con grafos reporta
ventajas— queda por detrás del mejor clásico en las cuatro combinaciones (IC 95 % que excluye 0). El GCN sin módulo temporal es el
mejor en UTAM-AUTO (0,0612 frente a 0,0639 del RF); no se hizo bootstrap de GCN frente a RF porque
no forma parte del criterio.

## Régimen secundario fuera de tiempo (adenda 1)

Entrenar con hogares de marzo–mayo (16.200) y evaluar contra junio (5.866) y octubre (685). Ver
tablas completas en `salidas/tabla_resultados.csv` (`regimen = tiempo_junio / tiempo_octubre`).
ZAT-AUTO: RF 0,1137, EB 0,1138, GCN 0,1096, ST-GNN 0,1157 (junio); EB 0,1168, GCN 0,1093,
ST-GNN 0,1202 (octubre). El ST-GNN no supera al mejor de EB/RF en ninguna escala ni modo
(bootstrap en la tabla de arriba y en el CSV).

**Advertencias, verificadas en los datos:**

1. **`fecha` es la marca de la entrevista, no el día del viaje.** Lleva hora (mediana 17 h; 25 %
   antes de las 13 h), y los hogares entrevistados en sábado o domingo (26 %) reportan **más**
   viajes por hogar que los entrevistados entre semana (4,58 frente a 4,34; domingo 4,65). Si
   `fecha` fuera el día del viaje, el fin de semana tendría menos viajes. Lo coherente es que los
   viajes se refieran a un día hábil anterior a la entrevista. Por eso **no** se construyeron los
   perfiles separados por día hábil y fin de semana (opcionales): con esta columna no se pueden
   separar. No encontré el diccionario de datos en la carpeta de la EM-2023; conviene confirmarlo
   ahí.
2. **El trabajo de campo avanzó por zonas.** 78 de 142 UTAM tienen más del 80 % de sus hogares en
   marzo–mayo y 17 tienen menos del 20 %; junio cubre 77 UTAM con ≥ 10 hogares y octubre solo 10.
   La partición temporal es sobre todo una extrapolación **espacial** con mitades desbalanceadas,
   no una prueba de estabilidad en el tiempo. Lo confirma la fiabilidad: en ZAT la correlación de
   ω entre marzo–mayo y junio es ≈ 0 (`fiabilidad_entre_mitades.csv`), y el Spearman del nivel
   cae a 0,1–0,4 porque el factor de expansión de cada submuestra no la representa a escala de
   zona.
3. Por lo anterior, los puertos S0 del régimen temporal (≈ 6.000 contra junio y ≈ 9.400 contra
   octubre, sobre 10.870) miden sobre todo que la verdad de junio u octubre deja muchas zonas
   vacías. La media de la ciudad gana ahí. No es informativo para elegir modelo.

## Qué implica para el artículo

- **R1.2 ("¿por qué un ST-GNN?") queda respondida en contra del ST-GNN como estimador**: con la
  EM-2023, un ST-GNN reentrenado sin fuga y con el protocolo de la literatura (enmascaramiento de
  nodos, salida multihora, pérdida ponderada) no mejora a un RF con rezago espacial ni a una
  contracción bayesiana empírica, ni para rellenar zonas observadas ni para estimar zonas sin dato.
  El techo lo pone el ruido de la encuesta, no la arquitectura.
- La salida honesta para el manuscrito es **la cadena de decisión con un estimador simple y
  transparente** (EB o RF con rezago) y reportar la incertidumbre muestral de ω y del nivel como
  insumo de robustez (B5 ya muestra que el núcleo robusto es ~72 % de los puertos). El ST-GNN
  puede mencionarse como alternativa evaluada y descartada, con este experimento en el suplemento.
- La escala ZAT para AUTO **no** sostiene perfiles horarios por zona: cualquier modelo devuelve
  algo cercano a la media de su entorno. Si se quiere ω por zona, conviene estimarlo en UTAM.
- La decisión S0 es más sensible al muestreo (3.400 puertos entre mitades) que a la elección de
  modelo (≈ 150–500 puertos entre los cuatro mejores). Eso refuerza el argumento de B5 de
  presentar un núcleo robusto en lugar de una asignación única.
- Este resultado **no** dice nada sobre los resultados publicados más allá de lo ya sabido: la
  etiqueta del ST-GNN publicado tenía fuga (sección 0.2 de la propuesta) y el defecto de códigos
  (memoria del proyecto). B6 solo responde si un ST-GNN bien hecho valdría la pena: no.

## Desviaciones y recortes declarados (el criterio no cambió)

1. **ZAT = 1.124**, no 1.156: se usan los polígonos de `ZAT2023.shp` cuyo centroide cae dentro de
   las 142 UTAM (como fija el preregistro). El 97,3 % de los extremos de viaje cae en esas ZAT.
2. ***Kriging*: K = 5** particiones (no 10), 5 pliegues de localidades cada una. Se corrió
   primero con AUTO y luego con todos los modos (7 procesos). No se calculan puertos S0 en
   *kriging* (solo hay predicción para las zonas ocultas de cada pliegue).
3. **Régimen temporal**: una sola partición por mes (no K). No se hicieron perfiles día hábil /
   fin de semana (ver advertencia 1).
4. La adenda dice "10:56", pero el archivo completo (con la adenda) se escribió de una vez y se
   selló a las 10:55:31; la hora real de la adenda es ≤ 10:55:31. Ningún resultado existía.
5. Redes: el reajuste final con ambas direcciones (A1→A2 y A2→A1) usa el número de épocas de la
   mejor validación, con un mínimo de 20. Las 4 configuraciones por método se eligen con la
   validación A2→A1. EB elige k (perfil) y k_L (nivel) por separado, cada uno entre 4 valores.
6. En *kriging*, el "crudo A" no existe para las zonas ocultas y se sustituye por la media de la
   ciudad (por eso aparecen iguales), como fijó el preregistro.
7. El bootstrap usa un único generador (semilla 20261011) recorrido en orden de combinación; al
   añadir *kriging* con todos los modos, los IC se recalcularon y cambiaron en la cuarta cifra
   decimal (p. ej., UTAM-todos-principal pasó de [0,0000; 0,0016] a [−0,0001; 0,0016]). El
   veredicto no cambia; las cifras del README son las de la última ejecución.
8. La primera corrida (principal) se lanzó sin limitar los hilos BLAS y tardó 20,5 min; las
   siguientes usan `OMP_NUM_THREADS=1`. Solo afecta al tiempo.
9. Observación sobre la validación de las redes: el ST-GNN eligió a menudo pocas épocas
   (20–110) por parada temprana. Es lo esperable con objetivos tan ruidosos, y se aplicó el mismo
   presupuesto a todos los métodos; no se ajustó nada después de ver resultados.

## Tiempo de cómputo (MacBook, 8 núcleos, CPU)

| Corrida | Trabajos | Reloj |
|---|---|---|
| Datos (`datos_b6.py`) | 1 | 8 s |
| Principal, K = 10, 2 escalas × 2 modos | 40 | 20,5 min |
| Tiempo (junio, octubre), 2 × 2 | 8 | 4,7 min |
| *Kriging* AUTO, K = 5 × 5 pliegues × 2 escalas | 50 | 20,5 min |
| *Kriging* todos los modos (7 procesos) | 50 | 41,6 min |
| Análisis + bootstrap (2.000 réplicas × 16 combinaciones) | 1 | ≈ 1,5 min (≈ 10 min con la CPU ocupada) |

Reloj total de la sesión, desde la lectura del contexto hasta el veredicto: ≈ 1 h 40 min (10:51 → 12:30); cómputo efectivo ≈ 1 h 30 min.

## Archivos

- `preregistro.md`, `preregistro.sha256` — protocolo y sello.
- `datos_b6.py` — zonas (UTAM, ZAT dentro de UTAM, localidad), hogares (incluye `fecha` y autos
  por hogar del módulo vehículos) y viajes por extremo. Reutiliza `canon`, `leer_modulo`,
  `parse_hour` y `to_num` de `scripts/retrain_stgnn.py`. Escribe `salidas/cache_b6.pkl` (no se
  versiona; se regenera en segundos).
- `run_b6.py principal|kriging|tiempo` — los 7 competidores, el ajuste anidado y S0 exacto.
- `analizar_b6.py` — tablas, fiabilidad, bootstrap y `veredicto.json`.
- `salidas/`: `tabla_resultados.csv`, `bootstrap_stgnn_vs_mejor.csv`,
  `fiabilidad_entre_mitades.csv`, `s0_puertos_mal_ubicados.csv` (por partición en `s0_*.csv`),
  `pred_*.csv.gz` (predicción por zona, partición y método), `meta_*.csv` (hiperparámetros
  elegidos y segundos por trabajo), `veredicto.json`, `log_*.txt`.

Reproducir (≈ 1 h en 8 núcleos):

```bash
cd experiments/revision_2026/B6_stgnn_honesto
export OMP_NUM_THREADS=1
../../../.venv/bin/python datos_b6.py
../../../.venv/bin/python run_b6.py principal --K 10
../../../.venv/bin/python run_b6.py tiempo
../../../.venv/bin/python run_b6.py kriging --K 5 --modos auto
../../../.venv/bin/python run_b6.py kriging --K 5 --modos todos --etiqueta _todos
../../../.venv/bin/python analizar_b6.py
```
