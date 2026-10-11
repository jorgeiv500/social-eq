# B7 — La brecha de provisión: denominador, índice, geografía e incertidumbre

Revisión JTRG-D-26-00692, segunda ronda (R1.B y R1.D de `revisor1_metodos.md`; «¿es pública la
brecha?» y «hallazgo de lugar» de `revisor2_politica.md`; R3.1 y R3.2 de `revisor3_geografia.md`).
**Sin NEOS y sin modelos nuevos.** Todo se calcula en local sobre las asignaciones publicadas
(`outputs/.../paper_v6_experiments/tables/scenario_S0.csv`, `scenario_S2.csv`) y los insumos
que ya existen. No se tocó `src/`, `latex/` ni ningún resultado publicado.

```bash
cd experiments/revision_2026/B7_brecha_denominador
../../../.venv/bin/python b7_brecha.py --B 200     # ~9 s, un proceso
```

Necesita `EM2023_DIR` (módulo de viajes de la EM-2023, igual que `scripts/retrain_stgnn.py`) y la
caché de hogares de B6 (`../B6_stgnn_honesto/salidas/cache_b6.pkl`, que no se publica).

**Verificación.** Con el cuartil vulnerable reconstruido (cuantil 0,75 del índice sobre las 142
zonas; coincide zona a zona con `high_vulnerability` del CSV publicado) y la población DANE, el
script reproduce exactamente `table_provision_intensity.csv`: S0 10,430 / 15,607 y 941,7 / 1.890,8
puertos; S2 10,417 / 15,623 y 947,2 / 1.901,8 (`salidas/b7_resumen.json`, clave
`verificacion_tabla_publicada`).

## Tabla resumen

| Punto | Indicador | S0 publicado | S2 publicado |
|---|---|---|---|
| 1 | Total por 10.000 residentes, vulnerable / resto (razón) | 10,43 / 15,61 (1,50) | 10,42 / 15,62 (1,50) |
| 1 | Residencial por 10.000 residentes | 6,71 / 10,85 (1,62) | 6,71 / 10,85 (1,62) |
| 1 | Trabajo por 10.000 viajes atraídos con propósito trabajo (todos los modos) | 14,01 / 6,16 (0,44) | 13,96 / 6,18 (0,44) |
| 1 | Trabajo por 10.000 viajes atraídos de trabajo en auto | 144,7 / 44,5 (0,31) | 144,1 / 44,6 (0,31) |
| 1 | Público por 10.000 viajes atraídos sin regreso al hogar ni trabajo | 1,80 / 1,26 (0,70) | 1,80 / 1,26 (0,70) |
| 1 | Público por 10.000 de esos viajes en auto | 32,9 / 8,0 (0,24) | 32,9 / 8,0 (0,24) |
| 2 | Razón resto/vulnerable con cuartil por personas por vivienda (13 de 36 zonas salen) | 1,26 | 1,26 |
| 2 | … por densidad (10 de 36 salen) | 1,43 | 1,43 |
| 2 | … por % en arriendo, 80 zonas (25 de las 34 con dato salen) | 1,01 | 1,01 |
| 2 | … por el índice PCA de la batería (11 de 36 salen) | 1,76 | 1,76 |
| 2 | … por estrato medio invertido (20 de 36 salen) | 1,15 | 1,15 |
| 3 | Provisión por anillos de distancia al centro (0–5 / 5–10 / 10–15 / >15 km) | — | 17,7 / 13,0 / 11,3 / 8,8 |
| 4 | Bootstrap, meta directa (s y ω de la encuesta): vulnerable / resto, mediana [IC 95 %] | 11,60 [11,23; 11,97] / 13,54 [13,23; 13,84] | no se re-resuelve |
| 4 | … diferencia; puertos a la media; a la paridad | 1,93 [1,30; 2,59]; 350 [237; 471]; 703 [476; 945] | — |
| 4 | Bootstrap, s de la encuesta y ω de atención publicada fija: vulnerable / resto | 9,95 [9,75; 10,14] / 15,92 [15,75; 16,10] | — |
| 4 | … diferencia; puertos a la media; a la paridad | 5,97 [5,65; 6,33]; 1.087 [1.028; 1.152]; 2.182 [2.065; 2.313] | — |

«Vulnerable / resto» en puertos por 10.000 del denominador indicado. La razón es resto/vulnerable:
mayor que 1 significa que el cuartil vulnerable recibe menos; menor que 1, que recibe más.

## 1. Brecha por sede (`salidas/b7_1_brecha_por_sede.csv`, `b7_viajes_atraidos_por_utam.csv`)

Denominadores:
- **residencial:** residentes DANE (120 zonas con población, como en la tabla publicada);
- **trabajo:** viajes expandidos (`fexp_vj`) de la EM-2023 cuyo destino es la UTAM y cuyo
  `motivo_viaje` es «A trabajar»;
- **público:** viajes expandidos atraídos cuyo motivo no es «A regresar al hogar» ni «A trabajar»
  (estudio, trámites, compras, salud, visitas, ocio…). Es la definición más defendible que permite
  la encuesta: un viaje hacia una actividad que no es la casa ni el trabajo, que es lo que cubre la
  carga pública de destino. Como sensibilidad se da también «sin regreso al hogar, incluido el
  trabajo».
- Cada denominador de viajes se da con todos los modos y solo con AUTO (modo principal), y en dos
  universos: zonas con población DANE (comparable con el titular) y todas las zonas con viajes. El
  universo cambia poco las tasas (p. ej. trabajo, resto 6,16 frente a 6,63).
- El 99,1 % de los 16,7 millones de viajes expandidos tiene destino en una de las 142 UTAM.

Lectura: **toda la brecha publicada es residencial.** Con residentes en el denominador, el cuartil
recibe 6,71 frente a 10,85 puertos residenciales (razón 1,62). Con viajes atraídos en el
denominador, la brecha de trabajo y la de carga pública **se invierten**: el cuartil vulnerable
recibe 2,3 veces más puertos de trabajo por viaje de trabajo atraído (14,0 frente a 6,2) y 1,4 veces
más puertos públicos por viaje atraído no residencial (1,80 frente a 1,26). Con viajes en auto la
inversión es mayor (3,2 y 4,1 veces), porque esas zonas atraen pocos viajes en auto. S2 y S0 son
indistinguibles a dos decimales en todas las sedes. La sede pública es idéntica en S0 y S2.

## 2. Sensibilidad al índice (`salidas/b7_2_sensibilidad_indice.csv`)

Cuartil = 25 % superior de cada indicador. Las asignaciones S0 y S2 publicadas no cambian; solo
cambia qué zonas forman el grupo. «Salen» = de las 36 zonas vulnerables publicadas, cuántas no
están en el nuevo cuartil.

| Índice | Zonas en el universo | Salen de las 36 | ρ con el publicado | S2 vuln. / resto | Razón | Puertos a la media / paridad |
|---|---|---|---|---|---|---|
| Publicado (personas por vivienda + densidad) | 142 | 0 | 1 | 10,42 / 15,62 | 1,50 | 947 / 1.902 |
| (a) Personas por vivienda | 142 | 13 | 0,74 | 11,30 / 14,25 | 1,26 | 524 / 905 |
| (b) Densidad | 142 | 10 | 0,79 | 10,71 / 15,30 | 1,43 | 835 / 1.666 |
| (c) % en arriendo (B3) | 80 (cuartil de 20) | 25 de 34 con dato | 0,33 | 11,08 / 11,17 | 1,01 | 11 / 13 |
| (d) PCA de la batería de robustez | 142 | 11 | 0,87 | 10,07 / 17,70 | 1,76 | 1.314 / 3.412 |
| (e) Estrato medio invertido (EM-2023) | 142 | 20 | 0,41 | 11,86 / 13,64 | 1,15 | 296 / 460 |

- (d) replica exactamente el código de la batería (`followup.py`, «PCA weights for vulnerability
  index»): primer componente de **las nueve** columnas numéricas de `utam_social_vulnerability.csv`,
  entre ellas `dane_population`, viviendas, hogares y densidad. Por eso contiene el denominador en
  mayor medida que el índice publicado y agranda la brecha. No es un índice independiente.
- (e) no lo pidieron los revisores con esas palabras, pero R2 pide probar el estrato medio. Se
  añade porque el dato ya existe (`avg_stratum`).
- Correlación de Spearman entre cada indicador y los puertos por habitante de S2 (120 zonas):
  publicado −0,56; personas por vivienda −0,35; densidad −0,49; PCA −0,68; estrato invertido −0,17;
  arriendo +0,03 (B3 daba 0,037 con otra muestra de zonas).

## 3. Geografía de la brecha (`salidas/b7_3_*.csv`)

Diez zonas con menor provisión per cápita en S2 (población ≥ 10.000, para que no dominen la
UPR de Sumapaz con 236 habitantes y Guaymaral con 2.001; la lista sin filtro está en el CSV):
Modelia (Fontibón) 4,16; San Cristóbal Norte (Usaquén) 4,54; Garcés Navas (Engativá) 5,83;
Diana Turbay (Rafael Uribe Uribe) 6,28; Casa Blanca Suba (Suba) 6,35; La Gloria (San Cristóbal)
6,69; Bolivia (Engativá) 6,69; Fontibón (Fontibón) 6,92; Marruecos (Rafael Uribe Uribe) 6,92;
Marco Fidel Suárez (Rafael Uribe Uribe) 7,52. Cinco de las diez están en el cuartil vulnerable.

Provisión media por localidad en S2 (puertos por 10.000 residentes), de menor a mayor: San
Cristóbal 8,7; Rafael Uribe Uribe 8,7; Engativá 8,8; Suba 9,9; Tunjuelito 10,6; Usaquén 10,7;
Ciudad Bolívar 11,5; Kennedy 11,6; Bosa 13,0; Antonio Nariño 14,0; Usme 14,1; Puente Aranda 14,4;
Fontibón 14,5; … Teusaquillo 33,1; Sumapaz 60,4; La Candelaria 69,2.

Gradiente (102 zonas con ≥ 10.000 habitantes; distancia del centroide al de La Candelaria):
ρ de Spearman −0,39 entre provisión per cápita y distancia; por anillos 17,7 (0–5 km, 19 zonas),
13,0 (5–10 km, 34), 11,3 (10–15 km, 36) y 8,8 (> 15 km, 13). Lectura: **hay un centro muy bien
provisto y un descenso con la distancia, pero la cola baja no es la periferia sur.** Bosa, Usme y
Ciudad Bolívar quedan en la mitad o por encima. Las zonas con menos puertos por residente forman un
arco occidental y norte (Fontibón, Engativá, Suba, Usaquén) más el sur pericentral (Rafael Uribe
Uribe y San Cristóbal). Esto concilia la §5.3 («no hay un gradiente centro–periferia simple») con la
Discusión: el gradiente existe en la media, pero el déficit no se concentra en la periferia sur.

## 4. Incertidumbre por muestreo (`salidas/b7_4_*.csv`, `b7_resumen.json`)

**Procedimiento.** Bootstrap de hogares de la EM-2023 (22.755 hogares), B = 200, semilla
20261011, **estratificado por UTAM del hogar** (142 estratos; cada hogar arrastra todos sus viajes
con su `fexp_vj`). En cada réplica se reconstruye la meta con el procedimiento de la cadena
(`make_targets`: t_iv = Q_v · s_i ω_iv / Σ_j s_j ω_jv, cuotas 7.352 / 2.739 / 779). S0 se resuelve
exacto por redondeo por sede (piso más remanente a las fracciones mayores, como en B5 y B6), y se
calculan la provisión del cuartil vulnerable publicado y del resto, su diferencia, los puertos a la
media y a la paridad con la misma aritmética de la tabla publicada.

**Aproximación declarada.** Re-entrenar el ST-GNN en cada réplica no es viable en este plazo. La
prioridad s_i se sustituye por la **meta directa de la encuesta**: el nivel de extremos de viaje
expandidos (salidas por hora de inicio más llegadas por hora de fin, todos los modos), normalizado
min–max como `priority_norm`. Su ρ de Spearman con la media horaria de la etiqueta del pipeline es
0,99999. Para el reparto por sede ω se dan dos variantes:
- **directa:** ω observado en la encuesta, como participación de los extremos de viaje en las
  ventanas 22–5, 8–17 y 18–21 (como en B6);
- **ω de atención fija:** ω = atención temporal publicada del ST-GNN (`build_venue_weights`), que
  no se puede remuestrear sin re-entrenar; solo varía s_i.

**Por qué no se re-resuelve S2.** S2 publicado se aparta de S0 en 18 puertos entre zonas (de
10.870; la tolerancia de fidelidad es de unos 8,7 puertos de desviación), y su provisión del
cuartil (10,42 frente a 10,43) coincide con la de S0 a dos decimales. La incertidumbre de muestreo
de S2 es, a esa precisión, la de S0. Re-resolver el MIP de S2 en 200 réplicas exigiría NEOS.

**Descomposición previa (`b7_4_descomposicion_meta_S0.csv`, muestra completa).** Qué parte de la
meta produce la brecha:

| Prioridad s_i | Reparto ω | Vulnerable / resto | Razón | A la media / paridad | Residencial vuln. / resto | Puertos movidos frente a S0 publicado |
|---|---|---|---|---|---|---|
| ST-GNN publicada | atención publicada | 10,43 / 15,61 | 1,50 | 942 / 1.891 | 6,71 / 10,84 | 0 (reproduce S0) |
| ST-GNN publicada | encuesta | 12,26 / 13,15 | 1,07 | 162 / 326 | 8,67 / 8,23 | 1.802 |
| encuesta | atención publicada | 9,93 / 15,93 | 1,60 | 1.090 / 2.189 | 6,51 / 10,90 | 1.367 |
| encuesta | encuesta | 11,60 / 13,54 | 1,17 | 352 / 707 | 8,26 / 8,40 | 2.113 |

**La mayor parte de la brecha la produce el reparto por sede tomado de la atención del ST-GNN, no
la prioridad.** Con el ω observado en la encuesta, la brecha residencial desaparece (8,26 frente a
8,40) y la total baja de 5,2 a 1,9 puertos por 10.000 residentes.

**Resultados del bootstrap** (mediana [IC 95 %]; percentiles 2,5 y 97,5 de 200 réplicas):

| | Directa (s y ω de la encuesta) | ω de atención fija |
|---|---|---|
| Vulnerable (por 10.000 hab.) | 11,60 [11,23; 11,97] | 9,95 [9,75; 10,14] |
| Resto | 13,54 [13,23; 13,84] | 15,92 [15,75; 16,10] |
| Diferencia | 1,93 [1,30; 2,59] | 5,97 [5,65; 6,33] |
| Razón resto / vulnerable | 1,17 [1,11; 1,23] | 1,60 [1,56; 1,65] |
| Puertos a la media (análogo a 947) | 350 [237; 471] | 1.087 [1.028; 1.152] |
| Puertos a la paridad (análogo a 1.902) | 703 [476; 945] | 2.182 [2.065; 2.313] |
| Residencial, vulnerable / resto | 8,27 [7,93; 8,62] / 8,38 [8,12; 8,67] | 6,51 [6,37; 6,63] / 10,90 [10,78; 11,03] |
| Puertos que cambian de zona-sede frente a la muestra completa | 533 [473; 613] | 270 [224; 306] |

Lectura: con la composición por sede fija, el muestreo de hogares mueve la brecha en ±0,3 puertos
por 10.000 y los «947» en unos ±60 puertos. La brecha es estable ante el muestreo, pero **no ante la
fuente de ω**. Cambiar la atención del ST-GNN por el perfil horario observado la reduce a un tercio,
mucho más que cualquier intervalo de muestreo. Ninguna variante del bootstrap reproduce la cifra
publicada (que usa la s_i del ST-GNN). Los intervalos describen la incertidumbre de la meta directa,
no de la cadena publicada.

## Archivos

| Archivo | Contenido |
|---|---|
| `b7_brecha.py` | Todo el cálculo (puntos 1–4 y verificación) |
| `salidas/b7_viajes_atraidos_por_utam.csv` | Viajes expandidos por UTAM de destino y propósito, todos los modos y AUTO |
| `salidas/b7_1_brecha_por_sede.csv` | Punto 1 |
| `salidas/b7_2_sensibilidad_indice.csv` | Punto 2 |
| `salidas/b7_3_diez_zonas_menor_provision_S2.csv`, `b7_3_provision_por_localidad_S2.csv` | Punto 3 |
| `salidas/b7_4_descomposicion_meta_S0.csv`, `b7_4_bootstrap_replicas.csv` | Punto 4 |
| `salidas/b7_resumen.json` | Verificación, correlaciones, gradiente, resumen del bootstrap |
| `salidas/b7_log.txt` | Salida de la última corrida |
