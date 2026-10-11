# C2 — Cimientos de datos para la extensión «de la zona al sitio»

Revisión de JTRG-D-26-00692. Este experimento **no optimiza nada** ni lanza NEOS. Prepara dos
insumos para una futura asignación de puertos por sitio:

1. una **capa de parqueaderos candidatos** en Bogotá (OSM + Zonas de Parqueo Pago + Catastro);
2. **llegadas y permanencias** de conductores de auto por ZAT/UTAM y hora (EM-2023).

No toca `main.tex`, la carta, `src/` ni resultados publicados. Fuentes, URL, fechas, licencias y
CRS: `datos/MANIFIESTO.md`. Todos los cruces espaciales se hacen en EPSG:9377.

## Cómo reproducir

```bash
PY=/Users/jorgeiv500/Documents/social-eq/.venv/bin/python
$PY parte1_parqueaderos.py --pbf RUTA/colombia-261009.osm.pbf   # ~9 min, 1 proceso
$PY parte1b_vulnerabilidad.py      # candidatos por cuartil de ν (usa salidas de B5)
$PY parte2_llegadas_permanencias.py  # ~25 s
$PY cruce_partes.py                # llegadas en zonas sin candidatos
```

`EM2023_DIR` (variable de entorno) apunta a la carpeta de Drive con `ZAT2023.shp` y los módulos
de la EM-2023, igual que en `scripts/retrain_stgnn.py`.

---

## Parte 1 — Capa de parqueaderos candidatos

### Fuentes y qué funcionó

| Fuente | Cómo se obtuvo | Elementos en Bogotá | Capacidad |
|---|---|---:|---|
| OSM `amenity=parking` | **Overpass falló** (406/504/500 en tres servidores, 6 intentos espaciados); se usó el **extracto Geofabrik de Colombia del 2026-10-09** leído con el controlador OSM de GDAL (pyogrio) | 2 566 (403 nodos, 2 163 polígonos) | etiqueta `capacity` (solo enteros explícitos) |
| Zonas de Parqueo Pago (Terminal de Transporte, 2026) | servicio REST oficial enlazado en la ficha CKAN (la descarga directa del portal expiró) | 862 tramos | `CUP_AUT_` (cupos de auto) |
| Catastro UAECD, uso predominante por manzana, vigencia 2026 | servicio REST de Catastro | 336 manzanas | no tiene |

**Catastro sí tiene una categoría de parqueaderos, pero combinada:** «DEPÓSITOS Y PARQUEADEROS»,
que además es el uso *predominante* por área construida de la manzana. Se extrajeron esas 336
manzanas (vigencia 2026), con la advertencia de que una parte serán depósitos, no parqueaderos.

### Esquema de la capa (`salidas/parqueaderos_candidatos.gpkg`, capa `candidatos`; y `.csv`)

`id` (`osm_<tipo>_<id>`, `zpp_<OBJECTID>`, `cat_<MANCODIGO>`), `fuente`, `nombre`, `capacidad`
(numérica o vacía; **nunca imputada**), `capacidad_texto` (valor original), `tipo` (OSM:
`parking=*`; ZPP: `via_publica_tarifada`; Catastro: `manzana_depositos_y_parqueaderos`), `acceso`,
`tarifa`, `operador`, `area_m2` (polígonos OSM y manzanas), `cupos_electricos` (ZPP),
`geometria_original`, `estado_zpp`, `ZAT`, `UTAM`, `UTAMNombre`, `osm_en_manzana_catastral`,
geometría **punto** en EPSG:9377 (nodo; punto representativo del polígono; punto medio del tramo
ZPP). El CSV añade `x_9377, y_9377, lon, lat`.

### Diagnóstico (`salidas/diagnostico_parte1.json`)

| Indicador | Valor |
|---|---:|
| Candidatos totales | **3 764** (OSM 2 566 · ZPP 862 · Catastro 336) |
| Con capacidad numérica | **927 (24,6 %)** — ZPP 100 %, OSM 2,5 % (65), Catastro 0 % |
| Cupos conocidos | 10 975 (ZPP 6 868 de auto; OSM 4 107) |
| Cupos ZPP para eléctricos | 185, en 94 tramos |
| OSM sin etiqueta `parking` / `access` / `fee` | 1 662 / 1 957 / 2 263 de 2 566 |
| OSM con acceso restringido explícito (`private`, `customers`, `no`, …) | 381 |
| OSM que cae dentro de una manzana catastral de parqueaderos (posible duplicado) | 50 |
| ZAT de Bogotá sin ningún candidato | **316 de 940** (357 si se ignora Catastro; 796 sin ningún candidato con capacidad) |
| UTAM sin ningún candidato | **31 de 142** (casi todas UPR/borde rural: UTAM5xx–7xx, UPR3, UPR4; además UTAM60, UTAM89, UTAM96) |
| UTAM sin ningún candidato con capacidad | 90 de 142 |

Coberturas por zona: `salidas/cobertura_por_zat.csv`, `salidas/cobertura_por_utam.csv`.

**Comparación con los 42 puntos de carga OSM** (`salidas/carga_existente_vs_candidatos.csv`):
los 42 caen en ZAT de Bogotá (24 UTAM). Distancia al candidato más cercano: mediana 73,5 m;
11 a ≤ 50 m, 29 a ≤ 100 m, 41 a ≤ 250 m (solo parqueaderos OSM: 9 / 26 / 40). 3 puntos de carga
están en ZAT sin candidato. Es decir, los puntos de carga actuales están casi siempre junto a un
parqueadero mapeado, lo que respalda usar la capa como conjunto de sitios.

### Por cuartil de vulnerabilidad ν (`parte1b_vulnerabilidad.py`)

Cuartil y población DANE tomados de `B5_nucleo_robusto/salidas/b5_zonas_S2.csv` (cuartil 4 =
vulnerable, verificado allí contra la máscara del artículo). Área de la UTAM en EPSG:9377.
Tabla completa: `salidas/candidatos_por_cuartil_vulnerabilidad.csv`.

| Cuartil ν | UTAM | Población | km² | Candidatos /10k hab | Candidatos /km² | Cupos conocidos /10k hab | Cupos /km² | UTAM sin candidato |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (menos vulnerable) | 36 | 869 554 | 985,1 | 22,75 | 2,01 | 87,7 | 7,74 | 4 |
| 2 | 24 | 752 551 | 644,9 | 6,74 | 0,79 | 24,7 | 2,89 | 3 |
| 3 | 46 | 2 002 625 | 254,0 | 3,09 | 2,43 | 5,2 | 4,12 | 23 |
| **4 (vulnerable)** | 36 | 3 652 765 | 130,9 | **1,81** | **5,05** | **1,2** | 3,36 | **1** |

Lectura: el cuartil vulnerable es denso, así que por km² tiene la mayor densidad de candidatos,
pero **por habitante tiene 12 veces menos candidatos y 73 veces menos cupos conocidos que el
cuartil 1**. Los cupos conocidos dependen sobre todo de ZPP, que es un programa de zonas
centrales y del norte: el 75,5 % de los tramos ZPP está en el cuartil 1 y el 1,9 % en el 4.

**UTAM vulnerables (36) sin ningún candidato, por fuente:** todas las fuentes 1 (UTAM96);
OSM 1; ZPP **32**; Catastro **9**. Sin ningún cupo conocido: **28 de 36**.

**Contraste de sesgo de mapeo OSM frente a Catastro** (participación de cada cuartil en el total
de la fuente; cociente < 1 = OSM sub-representa):

| Cuartil | % OSM | % Catastro | Cociente OSM/Catastro |
|---|---:|---:|---:|
| 1 | 45,3 | 49,4 | 0,92 |
| 2 | 13,8 | 7,4 | 1,86 |
| 3 | 18,9 | 19,6 | 0,96 |
| 4 | 22,1 | 23,5 | **0,94** |

En agregado, OSM **no** sub-representa de forma marcada al cuartil vulnerable (0,94). Entre UTAM
pobladas, en cambio, la densidad OSM por km² baja con ν (Spearman −0,30) y la catastral no
(+0,10), y ambas densidades se parecen poco entre sí (Spearman 0,30). La conclusión prudente:
hay un sesgo moderado dentro de los cuartiles, no un vacío sistemático del cuartil 4. El contraste
es indicativo: Catastro mide uso *predominante* de depósitos/parqueaderos, no presencia de
parqueaderos.

---

## Parte 2 — Llegadas y permanencias de conductores de auto (EM-2023)

### 2.1 Verificaciones previas

- **`fecha` de hogares es la marca de la ENTREVISTA, no el día de viaje.** No se halló el
  diccionario (ver manifiesto), así que se verificó con los datos: (i) lleva hora de reloj, y el
  67,3 % cae entre 15 y 23 h; (ii) hay 5 866 hogares con `fecha` en sábado o domingo; (iii) el
  perfil de los viajes es el mismo cualquiera que sea el día de `fecha`: trabajo+estudio 28,8–31,7 %
  y salidas entre 5 y 8 h 28,1–30,4 %, también cuando `fecha` es sábado o domingo
  (`salidas/verificacion_fecha_hogares.csv`). Si `fecha` fuera el día de viaje, los fines de semana
  tendrían muchos menos viajes de trabajo. Los viajes describen, por tanto, un día de referencia
  típico entre semana, presumiblemente el día hábil anterior. Esa regla exacta **no se pudo
  confirmar** sin el cuestionario. Para el perfil horario esto no importa; para unirlo con fechas
  concretas (clima, festivos), sí.
- **Horas decimales:** `hora_ini`/`hora_fin` son horas decimales (18,58 = 18:35). (fin − inicio)·60
  coincide con `duracion_min` a ±2 min en el 99,97 % de los viajes; la lectura hh,mm no encaja (12 %).
- **Conductor vs pasajero:** `modo_principal_desagrupado` lo distingue. Se usan solo los viajes
  «Vehículo privado como conductor»: **8 110 viajes** (1 275 432 expandidos con `fexp_vj`;
  3 913 personas). Quedan fuera del grupo AUTO: como pasajero 3 758, auto compartido 511,
  auto alquilado 177 y auto eléctrico 30, porque no declaran si la persona conducía.
- **Destinos:** 6 622 llegadas en ZAT de Bogotá; 1 488 con destino fuera de Bogotá se excluyen de
  las tablas zonales, pero no del cálculo de la permanencia.

### 2.2 Regla de permanencia (explícita)

- **Observada:** hora_ini del siguiente viaje de la misma persona (en cualquier modo) − hora_fin
  de la llegada.
- **Último viaje del día:** si el destino es el hogar (motivo «A regresar al hogar» o ZAT destino =
  ZAT del hogar), se imputa la **pernocta** = (24 − hora_fin) + hora_ini del primer viaje del día de
  esa persona. Si el destino no es el hogar, se **excluye**.
- **Inconsistentes** (siguiente salida anterior a la llegada): se excluyen (1 caso).
- **Sensibilidad «vehículo»:** hasta la siguiente salida como conductor desde la misma ZAT
  (`permanencia_vehiculo_por_proposito_sensibilidad.csv`). Media de 9,81 h frente a 9,66 h: casi
  idéntica.

| Estado de la permanencia (8 110 viajes de conductor) | Muestra | % muestra | % expandido |
|---|---:|---:|---:|
| Observada | 4 331 | 53,4 | 55,6 |
| **Imputada (pernocta)** | 3 619 | **44,6** | **42,8** |
| Excluida (último viaje, no al hogar) | 159 | 2,0 | 1,7 |
| Excluida (inconsistente) | 1 | 0,0 | 0,0 |

La imputación afecta casi solo a los viajes de regreso al hogar (95,5 % de ellos imputados); en
los demás propósitos es ≤ 2,7 %.

**Validación externa:** el módulo de etapas trae el tiempo de parqueo declarado (`t_parqueo_min`).
En 2 468 viajes con ambos datos, Spearman = 0,77 entre la permanencia calculada y la declarada;
medianas 3,0 h y 2,5 h; el 70 % difiere en menos de 1 h. Lugar de parqueo en destino (expandido):
garaje propio 47,7 %, espacio público no pago 19,4 %, parqueadero privado 19,4 %, parqueadero
público 6,8 %, garaje arrendado 3,8 %, espacio público pago 1,4 %.

### 2.3 Permanencia por propósito (destinos en Bogotá; ponderada por `fexp_vj`)

| Propósito | Llegadas (muestra) | Expandidas | Media h | Mediana h | p25–p75 h |
|---|---:|---:|---:|---:|---|
| Regresar al hogar | 3 079 | 507 277 | 15,19 | 15,33 | 12,0–19,5 |
| Trabajar | 1 763 | 283 503 | 7,34 | 8,00 | 5,0–9,7 |
| Trámite personal | 558 | 98 024 | 2,41 | 1,50 | 0,7–3,3 |
| Acompañar sin remuneración | 293 | 55 524 | 1,43 | 0,50 | 0,1–1,6 |
| Compras | 270 | 43 685 | 1,99 | 1,50 | 0,6–2,4 |
| Asuntos médicos | 212 | 33 555 | 2,26 | 1,75 | 1,0–2,7 |
| Visitar a alguien | 112 | 17 165 | 3,51 | 2,79 | 1,2–4,5 |
| Recreación y cultura | 106 | 16 939 | 3,70 | 3,01 | 2,0–5,6 |
| Estudiar | 86 | 12 124 | 5,99 | 5,71 | 4,7–7,6 |
| **Todos** | 6 622 | 1 097 764 | 9,66 | 9,47 | 2,8–15,0 |

Tabla completa (incluye % imputado y % excluido por propósito): `salidas/permanencia_por_proposito.csv`.

### 2.4 Salidas por zona y hora

| Archivo | Contenido |
|---|---|
| `llegadas_permanencia_por_zat.csv` / `_por_utam.csv` | por zona (las 940 ZAT / 142 UTAM, ceros incluidos): llegadas en muestra y expandidas, % imputado/excluido, permanencia media/mediana/p25/p75, ocupación máxima y su hora, % de llegadas por trabajo y al hogar |
| `llegadas_zat_hora.csv` / `llegadas_utam_hora.csv` | llegadas por zona × hora de llegada (⌊hora_fin⌋), expandidas y muestra |
| `ocupacion_zat_hora.csv` / `ocupacion_utam_hora.csv` | **O_zh**: vehículos (expandidos) llegados a z que siguen estacionados a las h:30, en reloj circular de 24 h |
| `perfil_horario_bogota.csv` | llegadas y ocupación de toda la ciudad por hora |
| `viajes_conductor_auto_permanencia.csv` | microdato por viaje de conductor; **no se publica** (excluido por `.gitignore`): se regenera con `parte2_llegadas_permanencias.py` a partir de la EM-2023 |

Perfil de la ciudad: las llegadas tienen su pico a las 7–8 h (98 000–102 000 por hora) y otro a
las 17–18 h; la ocupación esperada va de ≈ 383 000 vehículos (7 h) a ≈ 471 000 (0 h). La
madrugada la domina la pernocta en el hogar.

### 2.5 Escasez muestral (llegadas de conductores de auto sin expandir)

| Zona | Universo | Sin llegadas | < 10 | < 30 | Mediana | % de llegadas en zonas con ≥ 30 |
|---|---:|---:|---:|---:|---:|---:|
| ZAT | 940 | 218 | **746** | **896** | 3 | 34,7 |
| UTAM | 142 | 28 | **38** | **72** | 27 | 89,7 |

A escala ZAT el perfil horario es ruido: solo 44 ZAT tienen ≥ 30 llegadas. Por ZAT sirve el
**nivel** de llegadas, y conviene combinarlo con la permanencia por propósito, que sí es estable.

### 2.6 Cruce con la capa de candidatos (`cruce_partes.py`)

- 183 de las 316 ZAT sin candidato reciben llegadas de conductores; concentran el **13,0 %** de las
  llegadas expandidas. El **63,8 %** de las llegadas ocurre en ZAT sin ningún cupo conocido.
- Cupos conocidos por 1 000 llegadas expandidas de conductores: cuartil 1 = 15,2; 2 = 17,0;
  3 = 4,2; **4 = 1,9**. Candidatos por 1 000 llegadas: 3,9 / 4,6 / 2,5 / 2,8
  (`salidas/cruce_llegadas_candidatos_por_cuartil.csv`).

---

## Limitaciones

1. **Capacidad:** solo el 24,6 % de los candidatos tiene capacidad, y casi toda viene de ZPP
   (estacionamiento en vía). En OSM solo el 2,5 % la tiene; Catastro no trae capacidad. No se
   imputó ninguna.
2. **OSM incompleto y heterogéneo:** el 65 % no tiene etiqueta `parking`, el 76 % no tiene `access`
   y 381 son de acceso restringido explícito. Hay parqueaderos residenciales privados mezclados
   con parqueaderos públicos.
3. **Catastro es una aproximación gruesa:** la categoría mezcla depósitos y parqueaderos y solo
   aparece cuando ese uso predomina en la manzana. Los parqueaderos dentro de edificios de otro uso
   (centros comerciales, oficinas) no aparecen.
4. **Posibles duplicados:** 50 puntos OSM caen dentro de manzanas catastrales de la capa. Están
   marcados (`osm_en_manzana_catastral`), no eliminados.
5. **Overpass no respondió;** el extracto Geofabrik (2026-10-09) es equivalente en contenido, pero
   no es una consulta en vivo.
6. **Permanencia:** el 44,6 % (muestra) es pernocta imputada, con el supuesto de que el día siguiente
   repite la primera salida. Las llegadas excluidas (1,8 % en Bogotá) no ocupan en O_zh, así que
   O_zh se subestima un poco. O_zh solo cuenta vehículos que llegan en un viaje registrado: los
   autos que no se mueven en el día no aparecen.
7. **`fecha`** no identifica el día de viaje; no se pudo confirmar la regla de referencia (p. ej.,
   día hábil anterior) sin el cuestionario oficial.
8. **Escasez:** con 8 110 viajes de conductor en total, la escala ZAT no permite perfiles horarios
   fiables.

## Qué falta para la optimización por sitio

1. **Capacidad por sitio:** imputación declarada para OSM y Catastro (p. ej., área/25 m² por plaza
   para polígonos en superficie, o por número de pisos en `multi-storey`), o una fuente oficial de
   parqueaderos con cupos (registro de la Secretaría Distrital de Movilidad o de la
   Superintendencia de Industria y Comercio). Hoy el 64 % de las llegadas cae en ZAT sin capacidad
   conocida.
2. **Filtro de elegibilidad:** decidir qué acceso entra (público / clientes / privado residencial)
   y si los tramos ZPP en vía cuentan como sitio de carga.
3. **Deduplicación OSM ↔ Catastro ↔ ZPP** con una regla de distancia o de superposición.
4. **Asignar la demanda de las llegadas a los sitios** (dentro de la ZAT o con radio de caminata), con
   la permanencia por propósito para separar carga lenta (trabajo, hogar) y rápida
   (trámites, compras).
5. **Red eléctrica por sitio** (la capa de capacidad eléctrica del POT usada en A4 está a escala de
   zona).
6. **Agregar la demanda a UTAM o a grupos de ZAT** donde la muestra sea suficiente (≥ 30 llegadas), o
   suavizarla (p. ej., con un modelo jerárquico) antes de bajarla a ZAT.
