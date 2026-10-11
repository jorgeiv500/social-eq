# A7 — S2 con elegibilidad por uso del suelo (POT, Decreto 555 de 2021)

Experimento de la revisión JTRG-D-26-00692 (2026-10-10). Pregunta: ¿qué le pasa a la asignación
de equidad S2 si el uso del suelo entra **dentro** del modelo como restricción de elegibilidad
por (zona, sede)? Responde a la limitación declarada en el manuscrito ("does not internalize
zone-level land-use ... constraints") y complementa A4 (red eléctrica, cribado) y A5 (red dentro
del modelo, carpeta de otro agente).

Regla del bloque: ninguna cifra inventada. Cada número de este README sale de un archivo de
`salidas/tablas/` con su nombre.

## Dato

Capa oficial **"Área de actividad. Bogotá. POT Decreto 555. 2021. 1:2000"** y capa **"Sectores
incompatibles al uso residencial"** del servicio de cartografía oficial del POT (CC BY 4.0),
descargadas el 2026-10-10. URL, licencia, CRS, conteos y SHA-256 en `datos/MANIFIESTO.md`.

El POT zonifica el suelo urbano y de expansión en tres áreas de actividad (art. 240):
Estructurante (AAE; zonas receptoras de vivienda de interés social y de actividades económicas),
Proximidad (AAP; "zonas cuyo uso principal es el residencial") y Grandes Servicios
Metropolitanos (AAGSM; usos económicos y servicios metropolitanos). El art. 241 añade los
sectores incompatibles con el uso residencial.

## Cruce espacial (`preparar_uso_suelo.py`, local)

Superposición en EPSG:9377 con `UTAM2023.shp` (el shapefile del pipeline, igual que A4). Por UTAM:
área de cada código, área zonificada (unión de la capa) y fracción del **suelo zonificado** que
es compatible con cada sede. Se usa el suelo zonificado como denominador porque la capa no
cubre espacio público (vías, parques, estructura ecológica). Salida:
`salidas/tablas/uso_suelo_utam.csv`. 108 de 142 UTAM tienen cobertura ≥ 5 %; las 34 restantes
(UPR rurales y bordes no zonificados) no tienen dato.

## Regla declarada

Correspondencia área de actividad → sede (arts. 240-241):

| Sede | Suelo compatible |
|---|---|
| residential | AAP (ambas zonas) + AAE receptora de VIS, menos el suelo en sectores incompatibles con el uso residencial |
| workplace | AAE (ambas zonas) + AAGSM |
| public | AAE (ambas zonas) + AAGSM |
| (las tres) | PEMP de BIC nacional (se rige por su propia resolución; tratarlo como incompatible sería fabricar una restricción) |

**Elegibilidad:** la sede v no es elegible en la UTAM i (`caps[i, v] = 0`) si la fracción
compatible es menor que el umbral τ. Las UTAM con cobertura < 5 % quedan **sin tope**
(`caps = cuota`, no restrictivo), por el mismo principio de A4: sin dato no se impone
restricción. Umbral principal τ = 0.10 (una zona en la que menos de una décima parte del suelo
zonificado admite el uso de la sede no recibe puertos de esa sede); sensibilidad en
τ ∈ {0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50} (diagnóstico) y τ ∈ {0.05, 0.10, 0.20} (NEOS).

Implementación: `solve_equity_refined(..., caps=...)` de `src/social_eq/followup.py`, sin
modificar `src/`.

## Diagnóstico determinista (`run_a7.py`, sin NEOS)

Salida: `salidas/tablas/diagnostico_elegibilidad.csv` y `elegibilidad_tau_*.csv`.

- **Cota de conservación de masa.** Dentro de cada sede, la masa de target en pares excluidos
  (D) tiene que reaparecer en otra zona, así que Σu ≥ 2·D. Con cualquier τ ≥ 0.01, 2·D supera
  el presupuesto del S2 archivado (u*·1.08 = 117.54): ya con τ = 0.01 la exclusión del uso
  residencial en UTAM117 (Aeropuerto El Dorado, 100 % AAGSM, target residencial 109.7) da
  2·D = 219.31. **S2 con topes es infactible a ε = 0.08 sobre el u* archivado.**
- **Re-solución de S0 con topes (u\*_cap), sin NEOS.** S0 es separable por sede
  (min Σ|n − t| con Σn = cuota, 0 ≤ n ≤ tope, n entero) con costo convexo por zona; el voraz
  por costo marginal es exacto. Validación: sin topes reproduce el u* archivado
  (108.8337007840 frente a 108.8337001403; diferencia < 1e-6). Con τ = 0.05 / 0.10 / 0.20:
  u\*_cap = 398.06 / 618.97 / 1321.69.

| τ | pares (zona, sede) excluidos | residential / workplace / public | puertos de S2 en pares excluidos | u\*_cap |
|---|---|---|---|---|
| 0.01 | 1 | 1 / 0 / 0 | 110 | 288.97 |
| 0.05 | 6 | 2 / 2 / 2 | 170 | 398.06 |
| 0.10 | 13 | 3 / 5 / 5 | 301 | 618.97 |
| 0.20 | 34 | 6 / 14 / 14 | 660 | 1321.69 |
| 0.30 | 61 | 11 / 25 / 25 | 1225 | 2451.87 |
| 0.40 | 106 | 18 / 44 / 44 | 2222 | 4447.11 |
| 0.50 | 144 | 26 / 59 / 59 | 3076 | 6153.68 |

## Corridas NEOS (4, las cuatro `optimal`)

S2 con ν = índice de vulnerabilidad, κ = 4, cuotas del paper, sin piso, ε = 0.08 **sobre
u\*_cap** (diseño lexicográfico coherente: la etapa 1 y la etapa 2 ven las mismas
restricciones). CPLEX vía NEOS, 3600 s, gap 0.01, lanzadas el 2026-10-10. Una cuarta corrida
de **control** repite τ = 0.10 con la misma holgura absoluta que el S2 archivado
(presupuesto = u\*_cap + 0.08·u\*, ε efectivo = 0.0141), para separar el efecto del tope del
efecto de una holgura mayor.

Tabla `salidas/tablas/resumen_uso_suelo.csv` (generada por `resumir_a7.py`):

| Corrida | Terminación | Topes activos | Puertos movidos vs S2 | Zonas cambiadas (total / sede) | Gini pond. | Gini no pond. | Cobertura 15 min vulnerables | Puertos/10 000 cuartil vulnerable | Puertos/10 000 resto |
|---|---|---|---|---|---|---|---|---|---|
| S2 archivado | — | 0 | 0 | 0 / 0 | 0.164653 | 0.169942 | 1.0 | 10.42 | 15.62 |
| τ = 0.05 | optimal | 6 | 176 | 59 / 59 | 0.161731 | 0.166790 | 1.0 | 10.44 | 15.56 |
| **τ = 0.10** | optimal | 13 | 303 | 70 / 71 | 0.159255 | 0.163923 | 1.0 | 10.29 | 15.71 |
| τ = 0.10, holgura absoluta | optimal | 13 | 300 | 96 / 96 | 0.160733 | 0.165589 | 1.0 | 10.34 | 15.63 |
| τ = 0.20 | optimal | 34 | 692 | 115 / 116 | 0.151979 | 0.156018 | 1.0 | 9.79 | 16.09 |

"Topes activos": pares (zona, sede) con tope 0 donde el S2 archivado tenía puertos (todos los
excluidos, en las cuatro corridas). Ninguna corrida viola un tope (`violaciones_de_tope = 0` en los
manifiestos). Detalle de cambios: `salidas/tablas/detalle_cambios_vs_S2.csv`.

### Lectura

1. **El uso del suelo muerde, y muerde donde el modelo de demanda no mira el suelo.** El caso
   más claro es UTAM117 (Aeropuerto El Dorado): 100 % AAGSM y 110 puertos residenciales en S2.
   Con τ = 0.10 salen también UTAM61 (Ciudad Usme, 98 % AAE receptora de actividades
   económicas) y UTAM76 (Fontibón San Pablo) del uso residencial, y cinco UTAM de Proximidad
   (UTAM24, 52, 55, 66, 79) de los usos workplace y public.
2. **El efecto en equidad es pequeño y en la dirección de menor Gini**, porque el presupuesto de
   fidelidad crece con u\*_cap: Gini ponderado 0.164653 → 0.159255 (τ = 0.10). El control de
   holgura absoluta da 0.160733: la mayor parte de la baja persiste con la misma holgura que S2.
3. **La provisión per cápita del cuartil vulnerable apenas se mueve en τ ≤ 0.10**
   (10.42 → 10.29 puertos/10 000) y cae a 9.79 con τ = 0.20, cuando la regla excluye de los usos
   no residenciales a más zonas de Proximidad: el AAP es el 63 % del suelo zonificado de las 36
   UTAM del cuartil vulnerable (todas zonificadas) frente al 42 % en las otras 72 UTAM zonificadas
   (cálculo sobre `uso_suelo_utam.csv`).
4. **Salvedad: los puertos desplazados se concentran en zonas sin dato.** UTAM89 (San Isidro -
   Patios, Chapinero; cobertura 0 % de la capa, sin tope por regla) gana 109 / 140 / 272 puertos
   con τ = 0.05 / 0.10 / 0.20, y UTAM97 (Chicó Lago) gana 12 / 91 / 152. El tope de las zonas no
   zonificadas queda abierto: la capa del POT no las cubre y no se les impuso nada.

## Cómo correr

```bash
PY=/Users/jorgeiv500/Documents/social-eq/.venv/bin/python
$PY preparar_uso_suelo.py                       # cruce espacial (local)
$PY run_a7.py                                   # diagnóstico en seco (sin NEOS)
$PY run_a7.py --lanzar --umbrales 0.10          # una corrida NEOS por umbral
$PY run_a7.py --lanzar --umbrales 0.10 --holgura-absoluta   # control
$PY resumir_a7.py                               # tabla resumen
```

Registros de las corridas: `salidas/logs/`. Manifiestos por corrida:
`salidas/tablas/manifiesto_*.json`.

## Salvedades del dato

- Norma, no uso observado: el área de actividad dice qué usos admite el POT, no qué hay hoy en
  cada predio (eso lo daría el uso predominante catastral, no usado aquí).
- La correspondencia área de actividad → sede es una regla declarada; workplace y public
  comparten el mismo suelo compatible.
- La capa es la cartografía oficial del Decreto 555 de 2021 con fecha de captura 2021-12-29;
  no se verificó aquí su vigencia jurídica posterior.
