# B3 — Validación del proxy de necesidad con tenencia de la vivienda (EM 2021)

Experimento de validación para la revisión JTRG-D-26-00692, opción **(b)** del informe
la verificación de tenencia en arriendo (resumida aquí): correlacionar el
porcentaje de hogares en arriendo o subarriendo (Encuesta Multipropósito 2021, DANE–SDP)
con las variables del modelo (índice de vulnerabilidad ν_i, prioridad del ST-GNN y
asignación de puertos del escenario S2) para validar —o matizar— el proxy de necesidad de
carga pública que cuestiona el comentario R1.1. Todo el cálculo es local; no se usa NEOS
ni se re-corre ningún escenario.

Fecha del experimento: 2026-09-01.

---

## Hipótesis ex ante (registrada ANTES de calcular)

Escrita el 2026-09-01, antes de ejecutar cualquier correlación:

- **H1 (refuerzo del proxy).** Si el % de hogares en arriendo correlaciona
  **positivamente** con ν_i y/o con la prioridad normalizada del ST-GNN (Spearman ρ > 0,
  con significancia razonable dado el n), la tenencia **refuerza** el proxy actual: la
  necesidad que el modelo ya capta apunta hacia donde vive la población arrendataria, que
  es la que depende de la carga pública. En ese caso la variable se integra como
  **validación** en la respuesta a R1.1 y en un párrafo del texto, sin tocar ν_i.
- **H0/H2 (dimensión no capturada).** Si la correlación es **nula o negativa**, la
  tenencia es una dimensión de necesidad **no capturada** por el modelo. Se propone
  entonces como ν aumentado (corrida de sensibilidad a′ del informe base) o se declara
  como limitación explícita. **Ambos resultados se reportan tal cual salgan.**

Sobre la asignación de puertos (S2): la expectativa ex ante es más débil, porque la
asignación óptima combina prioridad con accesibilidad, demanda y restricciones de
capacidad; una correlación positiva pero más baja que la de ν_i/prioridad sería coherente
con el diseño del modelo. No se ajustará la hipótesis después de ver los datos.

Nota ecológica declarada de antemano: todas las correlaciones son entre **agregados
espaciales** (UPZ/localidad → UTAM), no entre hogares; no se hará inferencia individual.

---

## Datos fuente (con manifiesto)

| Archivo | Fuente y URL | Consultado | MD5 |
|---|---|---|---|
| `datos_fuente/20220901em2021_sincv_0.zip` | SDP, "Cuadros de salida 2021 (sin CV)", Encuesta Multipropósito 2021 (DANE–SDP), <https://www.sdp.gov.co/sites/default/files/encuestas-multiproposito/20220901em2021_sincv_0.zip> (página índice: <https://www.sdp.gov.co/content/resultados-encuesta-multiproposito>) | 2026-09-01 | `9ed8071048d44279175c95faf227b9f5` |
| `datos_fuente/20220901em2021_sinCV.xlsx` (extraído del zip) | ídem; hoja **"Tablas 62"** = pregunta C-1 "Tenencia de vivienda" (código `nhccp1`), con tablas oficiales por Bogotá, municipios, **localidades** y **UPZ** | 2026-09-01 | `75c18493b7e30d9a05f87baa6e377f42` |
| `datos_fuente/upz_bogota_ideca.geojson` | IDECA/Catastro Bogotá, capa oficial "Unidad Planeamiento Zonal" (112 UPZ, Dec. 190/POT), servicio ArcGIS REST `ordenamientoterritorial/unidadplaneamientozonal`, <https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/ordenamientoterritorial/unidadplaneamientozonal/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson> | 2026-09-01 | `4535931d6cf43bd2a9bf6a57484b6481` |

Extractos derivados (por el script, sin edición manual):

- `datos_fuente/extraidos/em2021_tenencia_localidades.csv` — tabla "Hogares según
  tenencia de la vivienda - Localidades" (20 localidades; hoja Tablas 62, filas 78–98).
- `datos_fuente/extraidos/em2021_tenencia_upz.csv` — tabla "Hogares según tenencia de la
  vivienda - UPZ" (95 UPZ con dato publicado; hoja Tablas 62, filas 137–233).

La variable usada es `pct_arriendo_subarriendo` = "Porcentaje. En arriendo o
subarriendo" (hogares expandidos). Cifra de control contra la tabla oficial: Bogotá
total = **50,11 %** (2 407 349 hogares expandidos, de los cuales 1 408 218 en arriendo o
subarriendo). Nota del cuadro oficial: en los resultados de la pregunta solo se incluyen
los hogares que respondieron; datos expandidos con proyecciones de población.

Datos del modelo (ya en el repo, sin modificar):

- `data/inputs/utam_social_vulnerability.csv` → `vulnerability_index` (ν_i) y
  `dane_population` (para per cápita).
- `data/inputs/priority_utam_scores.csv` → `priority_norm` (prioridad ST-GNN).
- `outputs/social_equity_followup_outputs/paper_v6_experiments/tables/scenario_S2.csv`
  → `ports_total` (asignación S2).
- `outputs/social_equity_followup_outputs/utam_shp/UTAM2023/UTAM2023.shp` → geometría y
  atributos (`MUNNombre`, `LOCNombre`) de las 142 UTAM.

## Niveles de agregación y cruce espacial (declarados)

1. **Principal — UPZ → UTAM (interpolación areal).** Las 116 UTAM bogotanas del modelo se
   cruzan con las 112 UPZ oficiales (IDECA) mediante **intersección por área** en
   geopandas (overlay en CRS proyectado EPSG:3116). El % de arriendo de cada UTAM es el
   **promedio ponderado por área de intersección** de las UPZ con dato EM 2021 (95 de
   112). Supuesto declarado: homogeneidad de tenencia dentro de cada UPZ (interpolación
   areal clásica). Las UTAM cuya área queda cubierta en menos del **50 %** por UPZ con
   dato se **excluyen y se listan** en `salidas/utam_excluidas.csv`.
2. **Chequeo de escala — localidad (n = 20).** `LOCNombre` del shapefile empalma directo
   con la tabla oficial por localidad. Advertencia declarada: con 20 observaciones el
   poder estadístico es bajo; se reporta como consistencia, no como evidencia principal.
3. **Sensibilidad — 142 UTAM.** A las 26 UTAM de municipios vecinos se les asigna el %
   municipal urbano (tabla oficial "Municipios urbano (Cabecera)" de la misma hoja).
   Choachí no hace parte de la cobertura EM 2021 → UTAM710 queda sin dato y se declara.

## Método estadístico

Correlaciones de **Spearman** (principal: no supone linealidad ni normalidad) y
**Pearson** (complementaria), con n explícito por fila, entre `pct_arriendo_subarriendo`
y: (i) `vulnerability_index` (ν_i), (ii) `priority_norm`, (iii) `ports_total` S2,
(iv) puertos S2 por 10 000 habitantes (`ports_total / dane_population × 10⁴`).
Script: `codigo/analisis_tenencia.py` (reproducible: `python3 codigo/analisis_tenencia.py`).

## Salidas

- `salidas/tabla_correlaciones.csv` — tabla de resultados (nivel, variable, n, Spearman ρ,
  p, Pearson r, p).
- `salidas/fig_tenure_validation.png` / `.pdf` — figura de dispersión **en inglés**
  (apta para manuscrito/suplemento).
- `salidas/utam_tenencia.csv` — % arriendo por UTAM con nivel de fuente y cobertura.
- `salidas/utam_excluidas.csv` — unidades excluidas y motivo.

## Resultados

Corrida del 2026-09-01 (`codigo/analisis_tenencia.py`, determinista, sin NEOS).
Cobertura: **80 de 142 UTAM** con interpolación areal válida (36 excluidas por cobertura
de UPZ con dato < 50 %, 26 sin intersección con la capa UPZ — el anillo de municipios
vecinos; listado en `salidas/utam_excluidas.csv`). Chequeo de escala: 19 de 20 localidades
(Sumapaz queda fuera al no tener UTAM válida).

| Nivel | Variable | n | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|---|---|
| UTAM | `vulnerability_index` (ν_i) | 80 | **0.3276** | **0.0030** | 0.3483 | 0.0015 |
| UTAM | `priority_norm` (ST-GNN) | 80 | 0.0505 | 0.6562 | 0.0261 | 0.8185 |
| UTAM | `ports_total` (S2) | 80 | −0.0610 | 0.5910 | −0.0112 | 0.9212 |
| UTAM | puertos S2 por 10 000 hab. | 80 | −0.0678 | 0.5503 | −0.0271 | 0.8113 |
| Localidad | ν_i | 19 | 0.0474 | 0.8473 | 0.0414 | 0.8665 |
| Localidad | prioridad ST-GNN | 19 | 0.0246 | 0.9205 | 0.2000 | 0.4118 |
| Localidad | puertos S2 | 19 | −0.2518 | 0.2984 | −0.1261 | 0.6068 |

**Veredicto frente a las hipótesis ex ante: resultado mixto, reportado tal cual.**

- **H1 se cumple solo para ν_i**: la tenencia correlaciona positiva y significativamente con
  el índice de vulnerabilidad. Salvedad de interpretación declarada: ν_i ya incluye un
  componente censal de tenencia (Tabla 1 del manuscrito), así que esto es **validez
  convergente entre dos fuentes independientes** del mismo constructo, no una validación
  externa pura.
- **H0/H2 se cumple para la prioridad aprendida**: ρ = 0.05 (p = 0.66). La tenencia es una
  dimensión de necesidad que el proxy de movilidad **no capta**. No se ajusta ν_i: el
  hallazgo se reporta como argumento a favor del diseño en dos etapas (necesidad derivada de
  movilidad y dependencia de carga pública por vivienda son dimensiones empíricamente
  distintas en esta ciudad) y como limitación declarada.
- La correlación con la asignación S2 es nula, coherente con la expectativa ex ante más
  débil registrada arriba.

Aplicado al manuscrito en §5.5 (*Design benchmarks and model-necessity checks*) el
2026-09-01.

## Reglas cumplidas

- Ninguna cifra fabricada: todos los porcentajes provienen de los cuadros oficiales
  descargados con manifiesto MD5.
- No se toca `src/`, `latex/` ni `comentarios/`; no se usa NEOS.
- La hipótesis quedó registrada antes de calcular y no se edita a posteriori.
